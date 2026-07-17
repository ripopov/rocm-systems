namespace rocjitsu {

enum class ConSanMoiDiagnosticKind : uint32_t {
  None = 0,
  AccessConflict = 1,
  MetadataFull = 2,
  BarrierDivergence = 3,
};

struct ConSanMoiWorkgroupShadowLayout {
  uint32_t base = 0;
  uint32_t size = 0;
  uint32_t required_group_segment_size = 0;
};

/// One eight-byte shadow slot covers one four-byte guest-LDS granule. Keeping
/// the mirror in the owning workgroup's LDS makes the storage bounded by the
/// kernel descriptor and prevents independent workgroups from contending on a
/// process-global slot. Dynamic-only LDS and kernels that cannot fit the
/// mirror in the architectural 64 KiB workgroup limit are not applicable.
[[nodiscard]] constexpr std::optional<ConSanMoiWorkgroupShadowLayout>
plan_consan_moi_workgroup_shadow(uint32_t original_group_segment_size) {
  constexpr uint32_t kAlignment = 8u;
  constexpr uint32_t kGuestGranuleBytes = 4u;
  constexpr uint32_t kMaxWorkgroupLdsBytes = 64u * 1024u;
  if (original_group_segment_size == 0u)
    return std::nullopt;
  const uint64_t base = (static_cast<uint64_t>(original_group_segment_size) + kAlignment - 1u) &
                        ~(static_cast<uint64_t>(kAlignment) - 1u);
  const uint64_t slot_count =
      (static_cast<uint64_t>(original_group_segment_size) + kGuestGranuleBytes - 1u) /
      kGuestGranuleBytes;
  const uint64_t size = slot_count * sizeof(uint64_t);
  const uint64_t required = base + size;
  if (required > kMaxWorkgroupLdsBytes)
    return std::nullopt;
  return ConSanMoiWorkgroupShadowLayout{.base = static_cast<uint32_t>(base),
                                        .size = static_cast<uint32_t>(size),
                                        .required_group_segment_size =
                                            static_cast<uint32_t>(required)};
}

enum class ConSanMoiInlineAtomicSupport : uint8_t {
  Supported,
  NonFlatAddress,
  NonzeroOffset,
  UnsupportedScope,
  UnsupportedWidth,
  CompareExchangeOutcomeUnavailable,
  UnsupportedEncoding,
  MissingOperands,
  MissingOrderingMetadata,
};

/// Consumer-neutral effective-address recipe for MOI atomic instrumentation.
/// Inline-shadow and record/replay must use the same recipe so that they key
/// synchronization metadata by exactly the address consumed by the guest
/// instruction.
enum class ConSanMoiAtomicAddressKind : uint8_t {
  Unsupported,
  FlatGuestPair,
  VglobalGuestPair,
  VglobalGuestPairMaterialized,
  VglobalMaterialized,
};

enum class ConSanMoiAtomicAddressSupport : uint8_t {
  Supported,
  UnsupportedArchitecture,
  UnsupportedAddressKind,
  UnsupportedWidth,
  UnsupportedEncoding,
  MissingAddressOperands,
  UnsupportedInputWidth,
  UnsupportedOffset,
  UnsupportedScope,
  UnsupportedResourcePlan,
  UnsupportedScratchShape,
  ResultAddressAlias,
  ScratchOperandAlias,
};

struct ConSanMoiAtomicAddressPlan {
  ConSanMoiAtomicAddressKind kind = ConSanMoiAtomicAddressKind::Unsupported;
  ConSanMoiAtomicAddressSupport support = ConSanMoiAtomicAddressSupport::UnsupportedAddressKind;
  uint16_t input_address_vgpr = 0;
  uint16_t input_address_vgpr_count = 0;
  std::optional<uint16_t> scalar_base_sgpr;
  int32_t signed_byte_offset = 0;
  uint16_t result_address_vgpr = 0;
  uint16_t result_address_vgpr_count = 0;
  uint16_t scratch_vgpr = 0;
  uint16_t scratch_vgpr_count = 0;
  ConSanRegisterAllocationSource resource_source = ConSanRegisterAllocationSource::Unsupported;

  [[nodiscard]] constexpr bool supported() const {
    return support == ConSanMoiAtomicAddressSupport::Supported;
  }
  [[nodiscard]] constexpr bool requires_materialization() const {
    return kind == ConSanMoiAtomicAddressKind::VglobalGuestPairMaterialized ||
           kind == ConSanMoiAtomicAddressKind::VglobalMaterialized;
  }
};

[[nodiscard]] std::string_view
consan_moi_atomic_address_support_name(ConSanMoiAtomicAddressSupport support);

/// Build a typed effective-address plan from decoded atomic operands and an
/// already resolved scratch-resource plan. FLAT aliases its guest 64-bit
/// address. VGLOBAL reserves the last two VGPRs of a five-VGPR private window
/// for `saddr + zero_extend(vaddr) + signed24(ioffset)`.
[[nodiscard]] ConSanMoiAtomicAddressPlan
plan_consan_moi_atomic_address(const ConSanAtomicSite &site, uint16_t scratch_vgpr,
                               uint16_t scratch_vgpr_count,
                               ConSanRegisterAllocationSource resource_source, rj_code_arch_t arch);

/// Emit the consumer-neutral VGLOBAL materialization. The returned sequence
/// snapshots and restores both VCC and SCC; @p vcc_save_sgpr names a pair and
/// @p scc_save_sgpr names one scalar register. A supported FLAT alias produces
/// an empty sequence.
[[nodiscard]] std::optional<std::vector<uint32_t>>
build_consan_moi_atomic_address_materialization(const ConSanMoiAtomicAddressPlan &plan,
                                                uint16_t vcc_save_sgpr, uint16_t scc_save_sgpr,
                                                rj_code_arch_t arch);

[[nodiscard]] constexpr ConSanMoiInlineAtomicSupport
classify_consan_moi_inline_atomic_support(const ConSanAtomicSite &site,
                                          ConSanMoiAtomicEventKind event_kind) {
  const bool is_flat = site.mnemonic.starts_with("flat_atomic");
  const bool is_vglobal = site.mnemonic.starts_with("global_atomic");
  if (!is_flat && !is_vglobal)
    return ConSanMoiInlineAtomicSupport::NonFlatAddress;
  const bool is_compare_exchange = site.mnemonic.find("cmpswap") != std::string::npos ||
                                   site.mnemonic.find("cmpxchg") != std::string::npos;
  (void)event_kind;
  if (site.width_bits != 32)
    return ConSanMoiInlineAtomicSupport::UnsupportedWidth;
  if (site.size != 3u * sizeof(uint32_t) ||
      (is_flat && (!site.raw_saddr || *site.raw_saddr != 0x7cu)))
    return ConSanMoiInlineAtomicSupport::UnsupportedEncoding;
  if (!site.raw_ioffset || (is_flat && *site.raw_ioffset != 0))
    return ConSanMoiInlineAtomicSupport::NonzeroOffset;
  if (!site.addr_vgpr || *site.addr_vgpr >= 255 || !site.data_vgpr ||
      (is_compare_exchange && (!site.dst_vgpr || *site.data_vgpr >= 255)))
    return ConSanMoiInlineAtomicSupport::MissingOperands;
  if (!site.raw_scope || !site.raw_th || !site.returns_old_value)
    return ConSanMoiInlineAtomicSupport::MissingOrderingMetadata;
  // Wave scope is not an inter-wave handoff. Workgroup, agent, and system
  // atomics are all valid causal publication/import events; the Inline keys
  // already include dispatch and workgroup identity, so scope 1 needs no
  // broader alias domain than the existing transaction.
  if (*site.raw_scope < 1u || *site.raw_scope > 3u)
    return ConSanMoiInlineAtomicSupport::UnsupportedScope;
  if (is_compare_exchange && !*site.returns_old_value)
    return ConSanMoiInlineAtomicSupport::CompareExchangeOutcomeUnavailable;
  return ConSanMoiInlineAtomicSupport::Supported;
}

[[nodiscard]] constexpr std::string_view
consan_moi_inline_atomic_support_name(ConSanMoiInlineAtomicSupport support) {
  switch (support) {
  case ConSanMoiInlineAtomicSupport::Supported:
    return "supported";
  case ConSanMoiInlineAtomicSupport::NonFlatAddress:
    return "non-flat-address";
  case ConSanMoiInlineAtomicSupport::NonzeroOffset:
    return "nonzero-offset";
  case ConSanMoiInlineAtomicSupport::UnsupportedScope:
    return "unsupported-scope";
  case ConSanMoiInlineAtomicSupport::UnsupportedWidth:
    return "unsupported-width";
  case ConSanMoiInlineAtomicSupport::CompareExchangeOutcomeUnavailable:
    return "compare-exchange-outcome-unavailable";
  case ConSanMoiInlineAtomicSupport::UnsupportedEncoding:
    return "unsupported-encoding";
  case ConSanMoiInlineAtomicSupport::MissingOperands:
    return "missing-operands";
  case ConSanMoiInlineAtomicSupport::MissingOrderingMetadata:
    return "missing-ordering-metadata";
  }
  return "unknown";
}

inline constexpr uint64_t kConSanMoiRecordReplayDefaultReportBufferBytes = 64u * 1024u;
inline constexpr uint64_t kConSanMoiSampledDefaultReportBufferBytes = 64u * 1024u;
inline constexpr uint64_t kConSanMoiInlineShadowDefaultReportBufferBytes = 512u * 1024u;
/// Hard ordinary/expert safety bounds. G6B3 owns aggregate reservation against
/// the process ceiling; this pure planner enforces the per-buffer ceiling.
inline constexpr uint64_t kConSanMoiAutoReportBufferCeilingBytes = 16u * 1024u * 1024u;
inline constexpr uint64_t kConSanMoiAutoReportProcessCeilingBytes = 256u * 1024u * 1024u;
// Record/replay synchronization probes can execute many times per static site.
// Reserve bounded dynamic headroom by default so normal loop trip counts do not
// immediately turn a complete static transform into report overflow.
inline constexpr uint64_t kConSanMoiRecordReplayDynamicEventHeadroom = 256u;

struct ConSanMoiAutoReportProcessBudget {
  uint64_t current_live_bytes = 0;
  uint64_t peak_live_bytes = 0;
};

/// Atomically composed by the hook under its registry lock. Kept pure here so
/// boundary, overflow, multi-object, and cleanup behavior are independently
/// testable without HSA allocation APIs.
[[nodiscard]] constexpr bool
reserve_consan_moi_auto_report_bytes(ConSanMoiAutoReportProcessBudget &budget, uint64_t bytes) {
  if (budget.current_live_bytes > kConSanMoiAutoReportProcessCeilingBytes ||
      bytes > kConSanMoiAutoReportProcessCeilingBytes - budget.current_live_bytes)
    return false;
  budget.current_live_bytes += bytes;
  budget.peak_live_bytes = std::max(budget.peak_live_bytes, budget.current_live_bytes);
  return true;
}

[[nodiscard]] constexpr bool
release_consan_moi_auto_report_bytes(ConSanMoiAutoReportProcessBudget &budget, uint64_t bytes) {
  if (bytes > budget.current_live_bytes)
    return false;
  budget.current_live_bytes -= bytes;
  return true;
}

enum class ConSanMoiReportLayoutFlag : uint32_t {
  None = 0,
  VersionedInlineExactShadow = 1u << 0u,
  DispatchQualifiedInlineMetadata = 1u << 1u,
  InlineCausalSnapshots = 1u << 2u,
};

inline constexpr uint32_t kConSanMoiReportKnownLayoutFlags =
    static_cast<uint32_t>(ConSanMoiReportLayoutFlag::VersionedInlineExactShadow) |
    static_cast<uint32_t>(ConSanMoiReportLayoutFlag::DispatchQualifiedInlineMetadata) |
    static_cast<uint32_t>(ConSanMoiReportLayoutFlag::InlineCausalSnapshots);

[[nodiscard]] constexpr uint64_t
consan_moi_default_auto_report_buffer_size(ConSanMoiEngine engine) {
  switch (engine) {
  case ConSanMoiEngine::RecordReplay:
    return kConSanMoiRecordReplayDefaultReportBufferBytes;
  case ConSanMoiEngine::InlineShadow:
    return kConSanMoiInlineShadowDefaultReportBufferBytes;
  case ConSanMoiEngine::Sampled:
    return kConSanMoiSampledDefaultReportBufferBytes;
  }
  return 0;
}

[[nodiscard]] constexpr bool
consan_moi_report_header_is_current(const ConSanMoiReportHeader &header) {
  return header.magic == kConSanMoiReportMagic &&
         header.abi_version == kConSanMoiReportAbiVersion &&
         header.header_size == sizeof(ConSanMoiReportHeader) &&
         header.engine <= static_cast<uint32_t>(ConSanMoiEngine::Sampled) &&
         (header.layout_flags & ~kConSanMoiReportKnownLayoutFlags) == 0;
}

/// Static event classification stored once per instrumented PC in a compact
/// record/replay trace. Dynamic records refer to this table by index.
