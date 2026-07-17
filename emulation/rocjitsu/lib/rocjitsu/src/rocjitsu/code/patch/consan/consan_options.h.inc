namespace rocjitsu {

enum class ConSanFlavor : uint8_t {
  None,
  SuperCollider,
  Moi,
};

enum class ConSanMoiEngine : uint8_t {
  RecordReplay,
  InlineShadow,
  Sampled,
};

enum class ConSanDelayMode : uint8_t {
  Nop,
  Sleep,
  SleepVar,
};

enum class ConSanBarrierMoveDirection : uint8_t {
  LegacyMarker,
  Earlier,
  Later,
};

/// CFG/convergence proof attached to a cross-block whole-barrier move.
enum class ConSanBarrierMoveCfgContract : uint8_t {
  SameBlock,
  /// Both successors of an acyclic conditional guard reach the reconverged
  /// source exactly once; moving the pair into the guard remains completing.
  CompletingStructuredDiamond,
  /// The destination is one optional EXEC-narrowed arm. This intentionally
  /// creates divergent participation and requires destructive X0 containment.
  DestructiveStructuredExecDiamond,
};

enum class ConSanPerturbationKind : uint8_t {
  None,
  Barrier,
  Atomic,
};

enum class ConSanPerturbationEdge : uint8_t {
  Release,
  Acquire,
};

/// Which cache boundary an atomic weaken-order fault removes. `Any` preserves
/// the historical first-qualified-boundary behavior; explicit edges are used
/// by fence-pair qualification and fail closed when that edge is absent.
enum class ConSanAtomicOrderEdge : uint8_t {
  Any,
  Release,
  Acquire,
};

/// Barrier-mutation forms whose evidence must not be conflated across host
/// code-object proof, live hardware, or multi-device/cluster topology.
enum class ConSanBarrierMutationForm : uint8_t {
  PairScopeCrossing,
  PairCompletingId,
  StaticNamedLifecycle,
};

enum class ConSanQualificationEvidence : uint8_t {
  None,
  Proven,
  DeferredA1,
};

struct ConSanBarrierMutationQualification {
  ConSanQualificationEvidence host = ConSanQualificationEvidence::None;
  ConSanQualificationEvidence live_gpu = ConSanQualificationEvidence::None;
  ConSanQualificationEvidence cluster_or_multi_device = ConSanQualificationEvidence::None;
};

/// Durable qualification boundary for the currently implemented barrier
/// mutation slices. This is deliberately narrower than encoder availability:
/// successful synthetic patching cannot silently become a live-GPU or
/// topology claim.
[[nodiscard]] constexpr ConSanBarrierMutationQualification
consan_barrier_mutation_qualification(std::string_view target, ConSanBarrierMutationForm form) {
  if (target == "gfx1201" && form == ConSanBarrierMutationForm::PairScopeCrossing)
    return {.host = ConSanQualificationEvidence::Proven};
  if (target == "gfx1250" && form == ConSanBarrierMutationForm::StaticNamedLifecycle) {
    return {.host = ConSanQualificationEvidence::Proven,
            .live_gpu = ConSanQualificationEvidence::DeferredA1};
  }
  return {};
}
enum class ConSanMoiOwnerSource : uint8_t {
  WorkitemId,
  HwId,
};

/// Controls whether flat/VFLAT sites whose group provenance is heuristic are
/// eligible for instrumentation.
enum class ConSanFlatProvenanceMode : uint8_t {
  Likely,
  Strict,
};

enum class ConSanRegisterAllocationSource : uint8_t {
  Unsupported,
  Explicit,
  LivenessDead,
  DescriptorGrowth,
  SpillRequired,
};

enum class ConSanRegisterPlanReason : uint8_t {
  None,
  InvalidRequest,
  ExplicitMisaligned,
  ExplicitOutOfRange,
  ExplicitLive,
  ForbiddenOverlap,
  MissingInstruction,
  MissingOwner,
  AmbiguousOwners,
  InvalidDescriptor,
  NoLegalWindow,
  DynamicStack,
};

/// Internal ownership retained while composing instrumentation over a staged
/// fault trampoline in a local or appended cave outside its source kernel's
/// FUNC symbol.
struct ConSanPreappliedCodeRange {
  uint64_t text_offset = 0;
  uint32_t size = 0;
  std::string kernel_name;
  std::optional<uint64_t> continuation_text_offset;
};

/// Mutation bytes already committed before a staged instrumentation pass.
/// Placement must never consume these ranges as newly available padding or a
/// cave, because doing so would erase the injected fault being analyzed.
struct ConSanPreappliedReservedRange {
  uint64_t text_offset = 0;
  uint32_t size = 0;
};

/// Caller-supplied exact MOI report layout. Auto allocation inventories the
/// final code, plans one heterogeneous ABI layout, and passes this same value
/// to the transformer, allocator, and reader. Explicit user buffers leave it
/// unset and retain the legacy size-derived layout contract.
struct ConSanMoiReportLayoutOverride {
  ConSanMoiEngine engine = ConSanMoiEngine::RecordReplay;
  uint32_t access_record_capacity = 0;
  uint32_t barrier_record_capacity = 0;
  uint32_t atomic_record_capacity = 0;
  uint32_t fence_record_capacity = 0;
  uint32_t diagnostic_capacity = 0;
  uint32_t exact_shadow_entry_capacity = 0;
  uint32_t inline_atomic_release_capacity = 0;
  uint32_t inline_acquired_epoch_token_capacity = 0;
  uint32_t inline_causal_snapshot_capacity = 0;
  uint32_t sampled_watchpoint_capacity = 0;
  uint32_t sampled_causal_window_capacity = 0;
  uint32_t sampled_sync_metadata_capacity = 0;
  uint32_t sampled_pending_acquire_capacity = 0;
  uint64_t access_records_offset = 0;
  uint64_t barrier_records_offset = 0;
  uint64_t atomic_records_offset = 0;
  uint64_t fence_records_offset = 0;
  uint64_t diagnostic_records_offset = 0;
  uint64_t exact_shadow_entries_offset = 0;
  uint64_t inline_atomic_release_slots_offset = 0;
  uint64_t inline_acquired_epoch_token_slots_offset = 0;
  uint64_t inline_causal_snapshots_offset = 0;
  uint64_t sampled_watchpoints_offset = 0;
  uint64_t sampled_causal_windows_offset = 0;
  uint64_t sampled_sync_metadata_offset = 0;
  uint64_t sampled_pending_acquires_offset = 0;
  uint64_t required_bytes = 0;
};

struct ConSanOptions {
  ConSanFlavor flavor = ConSanFlavor::None;
  ConSanMoiEngine moi_engine = ConSanMoiEngine::RecordReplay;
  ConSanMoiOwnerSource moi_owner_source = ConSanMoiOwnerSource::WorkitemId;
  ConSanFlatProvenanceMode flat_provenance_mode = ConSanFlatProvenanceMode::Likely;
  bool fail_closed = false;
  bool probe_nop = false;
  bool probe_trampoline_nop = false;
  bool probe_endpgm = false;
  bool probe_lds_endpgm = false;
  bool probe_lds_check_trap = false;
  bool probe_flat_check_trap = false;
  bool probe_flat_trap = false;
  /// Replace a statically unique immediate barrier wait that belongs to no
  /// decoded synchronization sequence with a terminal instruction. This is a
  /// bounded fail-fast diagnostic for definitely ill-formed programs, not a
  /// guess about dynamic or ambiguous barrier state.
  bool abort_unmatched_barrier_wait = false;
  bool fault_drop_barrier = false;
  /// Explicit destructive-test opt-in for dropping only one instruction from
  /// an otherwise qualified logical barrier pair.
  bool fault_allow_destructive_incomplete_barrier_drop = false;
  bool fault_move_barrier = false;
  /// Explicit non-destructive-test opt-in for moving a reconverged barrier
  /// earlier into the guard of a narrowly proven completing CFG diamond.
  bool fault_allow_completing_conditional_barrier_move = false;
  /// Explicit destructive-test opt-in for moving a uniform barrier pair into
  /// one arm of a narrowly proven structured EXEC diamond.
  bool fault_allow_destructive_divergent_barrier_move = false;
  bool fault_mutate_barrier_id_scope = false;
  bool fault_mutate_barrier_participants = false;
  bool fault_atomic_wrong_address = false;
  bool fault_atomic_weaken_order = false;
  ConSanAtomicOrderEdge fault_atomic_order_edge = ConSanAtomicOrderEdge::Any;
  bool fault_atomic_weaken_scope = false;
  bool fault_ordinary_wrong_address = false;
  bool fault_ordinary_weaken_order = false;
  bool fault_ordinary_weaken_scope = false;
  /// Caller-validated, aligned byte delta to a distinct valid atomic target.
  uint32_t fault_atomic_address_delta = 4;
  /// Caller-validated, aligned byte delta to a distinct valid ordinary-load target.
  uint32_t fault_ordinary_address_delta = 4;
  bool fault_dry_run = false;
  bool fault_require_exactly_one = false;
  /// SuperCollider-only bounded synchronization-edge perturbation request.
  /// SC2A1 inventories and plans the request but does not emit code.
  ConSanPerturbationKind sc_perturb_kind = ConSanPerturbationKind::None;
  ConSanPerturbationEdge sc_perturb_edge = ConSanPerturbationEdge::Release;
  std::string sc_perturb_identity;
  uint32_t sc_perturb_index = 0;
  uint32_t sc_perturb_max = 1;
  uint32_t sc_perturb_sleep = 1;
  uint32_t sc_perturb_required_count = 0;
  bool moi_init_owner_epoch = false;
  bool moi_track_barriers = false;
  bool moi_track_atomics = false;
  bool moi_dynamic_access_records = false;
  /// Internal rollout marker for the collision-free per-workgroup Inline
  /// shadow. The hook enables this for InlineShadow; direct patcher tests may
  /// leave it disabled while exercising the legacy global layout.
  bool moi_inline_workgroup_shadow = false;
  /// Check the immediately preceding direct-sampled slot in-kernel and
  /// increment the shared diagnostic counter on a sampled conflict.
  bool moi_sampled_check = false;
  /// Debug-only retained capture of the first address-partition pending/group
  /// masks. Never enable this for qualification or acceptance runs.
  bool moi_partition_mask_debug = false;
  /// Test-only control that exercises the spill tier even when a dead or
  /// descriptor-growth VGPR window is available.
  bool force_vgpr_spill = false;
  /// Test-only control that exercises derived-owner/private-epoch state even
  /// when a dedicated persistent VGPR pair is available.
  bool force_private_epoch = false;
  /// Test-only substring filter for selecting MOI candidates from one kernel.
  std::string test_kernel_name_filter;
  /// Internal marker set after ConSan assigns persistent owner/epoch VGPRs.
  bool automatic_moi_persistent_vgprs = false;
  /// Internal marker set when inline shadow derives owner at each probe and
  /// keeps epoch in per-lane private memory.
  bool automatic_moi_private_epoch = false;
  /// Internal marker set after ConSan assigns the scalar EXEC-save window.
  bool automatic_moi_exec_save_sgprs = false;
  /// Internal marker set when an Inline access probe needs the final scalar
  /// save slot for a site-local dynamic-stack spill frame.
  bool moi_dynamic_stack_spill = false;
  /// Internal marker set after ConSan assigns the hw_id owner temporary SGPR.
  bool automatic_moi_owner_sgpr = false;
  /// Internal marker set after ConSan assigns the persistent hardware
  /// dispatch-ID SGPR pair used by dispatch-qualified Inline metadata.
  bool automatic_moi_dispatch_id_sgprs = false;
  /// Internal metadata used only by the mutation-then-instrument composition
  /// path. It prevents appended fault code from being attributed to an
  /// unrelated tail function merely because that FUNC covers the cave bytes.
  std::vector<ConSanPreappliedCodeRange> preapplied_code_ranges;
  std::vector<ConSanPreappliedReservedRange> preapplied_reserved_ranges;
  uint32_t fault_barrier_index = 0;
  uint32_t fault_atomic_index = 0;
  uint32_t fault_ordinary_index = 0;
  /// Exact stable identity from ConSanResult::fault_sites. When set, this
  /// supersedes the legacy per-kind global index.
  std::string fault_site_identity;
  /// Exact stable real-instruction destination for markerless barrier movement.
  std::string fault_barrier_destination_identity;
  /// Exact logical barrier sequence selected for a whole-pair ID/scope dry run.
  std::string fault_barrier_sequence_identity;
  /// Optional independently resolved second complete barrier pair. When both
  /// fields are set, a drop-barrier request is one bounded all-or-nothing
  /// logical mutation over the two exact physical pairs.
  std::string fault_barrier_companion_site_identity;
  std::string fault_barrier_companion_sequence_identity;
  std::optional<int32_t> fault_barrier_target_id;
  std::optional<uint32_t> fault_barrier_target_participant_count;
  std::optional<uint64_t> fault_barrier_target_participant_mask;
  ConSanBarrierMoveDirection fault_barrier_move_direction =
      ConSanBarrierMoveDirection::LegacyMarker;
  /// Test-only s_nop immediate used as the destination of a moved barrier.
  uint16_t fault_barrier_marker_imm = 42;
  /// Internal recursion guard: faults have already been applied to this ELF.
  bool faults_preapplied = false;
  ConSanDelayMode delay_mode = ConSanDelayMode::Nop;
  uint16_t delay_var_ssrc = 106;
  std::optional<uint16_t> scratch_vgpr;
  std::optional<uint16_t> moi_exec_save_sgpr;
  std::optional<uint16_t> moi_owner_sgpr;
  std::optional<uint16_t> moi_dispatch_id_sgpr;
  std::optional<uint16_t> moi_owner_vgpr;
  std::optional<uint16_t> moi_epoch_vgpr;
  /// InlineShadow-only persistent exact workgroup identity. Launch SGPRs are
  /// copied into this VGPR at entry because they become ordinary guest state
  /// after the ABI prologue.
  std::optional<uint16_t> moi_workgroup_key_vgpr;
  /// Internal persistent per-wave sequence used by runtime Sampled selection.
  std::optional<uint16_t> moi_sample_sequence_vgpr;
  std::optional<uint64_t> report_buffer_address;
  std::optional<uint64_t> moi_report_buffer_address;
  uint64_t moi_report_buffer_size = 0;
  std::optional<ConSanMoiReportLayoutOverride> moi_report_layout;
  /// Generation stored in direct sampled entries. Auto report buffers set
  /// this from their versioned header; explicit buffers default to zero.
  uint64_t moi_report_generation = 0;
  /// Dispatch identity stored in direct sampled causal-window descriptors.
  /// Auto report buffers use the stable code-object reader identity.
  uint64_t moi_report_dispatch_id = 0;
  uint32_t delay_nops = 0;
  uint32_t max_patches = 1;
  uint32_t moi_sample_stride = 1;
  uint32_t moi_sample_offset = 0;
  /// Runtime deterministic wave selector for direct sampled probes. The
  /// Stride is a power of two and selects a dispatch/workgroup/wave/access hash.
  uint32_t moi_runtime_sample_stride = 1;
  uint32_t moi_runtime_sample_offset = 0;
  uint32_t report_marker = 1;
};

/// Process-local exact-one budget for repeated dynamic loads of one stable
/// fault-site identity. The caller serializes observe() with code-object load
/// selection. Occurrences are one-based and overflow is permanently fatal.
struct ConSanFaultLoadSelection {
  uint64_t occurrence = 0;
  bool selected = false;
  bool overflow = false;
};

class ConSanFaultLoadSelector {
public:
  explicit ConSanFaultLoadSelector(uint64_t requested_occurrence = 0)
      : requested_occurrence_(requested_occurrence) {}

  [[nodiscard]] ConSanFaultLoadSelection observe() {
    if (observed_ == std::numeric_limits<uint64_t>::max()) {
      overflow_ = true;
      return {observed_, false, true};
    }
    ++observed_;
    const bool selected = requested_occurrence_ != 0 && observed_ == requested_occurrence_;
    selected_ += selected ? 1u : 0u;
    return {observed_, selected, false};
  }

  [[nodiscard]] uint64_t requested_occurrence() const { return requested_occurrence_; }
  [[nodiscard]] uint64_t observed() const { return observed_; }
  [[nodiscard]] uint64_t selected() const { return selected_; }
  [[nodiscard]] bool overflow() const { return overflow_; }
  [[nodiscard]] bool accepted() const {
    return requested_occurrence_ != 0 && !overflow_ && observed_ >= requested_occurrence_ &&
           selected_ == 1;
  }

private:
  uint64_t requested_occurrence_ = 0;
  uint64_t observed_ = 0;
  uint64_t selected_ = 0;
  bool overflow_ = false;
};
