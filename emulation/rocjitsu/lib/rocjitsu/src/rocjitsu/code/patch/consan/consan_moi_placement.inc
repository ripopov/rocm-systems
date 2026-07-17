struct MoiLocalNopIsland {
  uint64_t text_offset = 0;
  bool claimed = false;
};

constexpr uint32_t kMoiLocalIndirectIslandWords = 8u;

class MoiLocalNopIslandAllocator {
public:
  MoiLocalNopIslandAllocator(std::span<const uint8_t> text, const ConSanResult &result,
                             rj_code_arch_t arch, uint32_t island_words) {
    if (island_words == 0)
      return;
    struct OwnerRange {
      uint64_t entry = 0;
      uint64_t code_end = 0;
    };
    std::vector<OwnerRange> owners;
    owners.reserve(result.kernels.size() + result.functions.size());
    const auto append_owner = [&](uint64_t entry, uint64_t code_size) {
      if (code_size == 0 || entry >= text.size())
        return;
      owners.push_back({entry, entry + std::min<uint64_t>(code_size, text.size() - entry)});
    };
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (kernel.has_text_range)
        append_owner(kernel.entry_text_offset, kernel.code_size);
    }
    for (const ConSanFunctionInfo &function : result.functions)
      append_owner(function.entry_text_offset, function.code_size);
    std::ranges::sort(owners, {}, &OwnerRange::entry);
    std::vector<OwnerRange> unique;
    for (const OwnerRange &owner : owners) {
      if (!unique.empty() && unique.back().entry == owner.entry) {
        unique.back().code_end = std::max(unique.back().code_end, owner.code_end);
      } else {
        unique.push_back(owner);
      }
    }

    const uint32_t nop = build_s_nop(0, arch);
    const uint64_t island_bytes = static_cast<uint64_t>(island_words) * sizeof(uint32_t);
    const auto overlaps_patch = [&](uint64_t begin, uint64_t end) {
      return std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
        const uint64_t anchor_end =
            patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1u);
        const uint64_t trampoline_end = patch.trampoline_offset + patch.trampoline_size;
        return (begin < anchor_end && patch.anchor_offset < end) ||
               (patch.trampoline_size != 0 && begin < trampoline_end &&
                patch.trampoline_offset < end);
      });
    };
    for (size_t owner_index = 0; owner_index < unique.size(); ++owner_index) {
      uint64_t cursor = util::align_up(unique[owner_index].code_end, uint64_t{4});
      const uint64_t gap_end =
          owner_index + 1u < unique.size() ? unique[owner_index + 1u].entry : text.size();
      while (cursor + island_bytes <= gap_end) {
        bool all_nops = !overlaps_patch(cursor, cursor + island_bytes);
        for (uint64_t offset = 0; all_nops && offset < island_bytes; offset += sizeof(uint32_t)) {
          uint32_t word = 0;
          std::memcpy(&word, text.data() + cursor + offset, sizeof(word));
          all_nops = word == nop;
        }
        if (all_nops) {
          islands_.push_back({cursor, false});
          cursor += island_bytes;
        } else {
          cursor += sizeof(uint32_t);
        }
      }
    }
  }

  [[nodiscard]] std::optional<uint64_t> claim_reachable(uint64_t anchor_offset) {
    MoiLocalNopIsland *best = nullptr;
    uint64_t best_distance = std::numeric_limits<uint64_t>::max();
    for (MoiLocalNopIsland &island : islands_) {
      if (island.claimed || !compute_sopp_branch_simm16(anchor_offset, island.text_offset))
        continue;
      const uint64_t distance = anchor_offset < island.text_offset
                                    ? island.text_offset - anchor_offset
                                    : anchor_offset - island.text_offset;
      if (distance < best_distance) {
        best = &island;
        best_distance = distance;
      }
    }
    if (best == nullptr)
      return std::nullopt;
    best->claimed = true;
    return best->text_offset;
  }

private:
  std::vector<MoiLocalNopIsland> islands_;
};

void append_nop_padding_to_alignment(std::vector<uint8_t> &bytes, uint64_t alignment,
                                     rj_code_arch_t arch) {
  if (alignment == 0)
    return;
  while (bytes.size() % alignment != 0)
    append_word_bytes(bytes, build_s_nop(0, arch));
}

[[nodiscard]] bool write_word_bytes(std::vector<uint8_t> &bytes, uint64_t offset, uint32_t word) {
  if (offset > bytes.size() || sizeof(word) > bytes.size() - offset)
    return false;
  std::memcpy(bytes.data() + offset, &word, sizeof(word));
  return true;
}

[[nodiscard]] uint32_t moi_descriptor_wavefront_size(const KD &desc) {
  const bool wave32 = AMDHSA_BITS_GET(desc.kernel_code_properties,
                                      kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
  return wave32 ? 32 : 64;
}

[[nodiscard]] uint32_t moi_descriptor_user_sgpr_count(const KD &desc) {
  return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
}

[[nodiscard]] uint16_t moi_descriptor_system_sgpr_count(const KD &desc) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  return static_cast<uint16_t>(
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) != 0 ? 1u : 0u));
}

[[nodiscard]] uint16_t moi_descriptor_dispatch_id_prefix_sgpr_count(const KD &desc) {
  const uint16_t properties = desc.kernel_code_properties;
  return consan_moi_amdhsa_dispatch_id_prefix_sgpr_count(
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) != 0);
}

[[nodiscard]] bool moi_descriptor_has_kernarg_preload(const KD &desc) {
  return AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH) != 0;
}

[[nodiscard]] ConSanMoiDispatchIdPreloadPlan
moi_descriptor_dispatch_id_preload_plan(const KD &desc) {
  const uint16_t properties = desc.kernel_code_properties;
  const bool dispatch_enabled =
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) != 0;
  const uint16_t original_user_count = static_cast<uint16_t>(moi_descriptor_user_sgpr_count(desc));
  const uint16_t prefix = moi_descriptor_dispatch_id_prefix_sgpr_count(desc);
  const uint16_t named_suffix = static_cast<uint16_t>(
      (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT) != 0
           ? 2u
           : 0u) +
      (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE) != 0
           ? 1u
           : 0u) +
      AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH));
  const uint32_t minimum_declared_user_count =
      static_cast<uint32_t>(prefix) + (dispatch_enabled ? 2u : 0u) + named_suffix;
  if (minimum_declared_user_count > original_user_count) {
    ConSanMoiDispatchIdPreloadPlan invalid;
    invalid.support = ConSanMoiDispatchIdPreloadSupport::InvalidDispatchPosition;
    invalid.dispatch_id_sgpr = prefix;
    invalid.original_user_sgpr_count = original_user_count;
    invalid.expanded_user_sgpr_count = original_user_count;
    invalid.system_sgpr_count = moi_descriptor_system_sgpr_count(desc);
    return invalid;
  }
  return consan_moi_plan_dispatch_id_preload(original_user_count,
                                             moi_descriptor_system_sgpr_count(desc), prefix,
                                             dispatch_enabled, kMaxSgprs);
}

[[nodiscard]] std::optional<uint16_t> moi_descriptor_workgroup_id_sgpr(const KD &desc,
                                                                       uint32_t dimension) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  const bool enabled[3] = {
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0,
  };

  uint32_t sgpr = moi_descriptor_user_sgpr_count(desc);
  for (uint32_t i = 0; i < 3; ++i) {
    if (!enabled[i])
      continue;
    if (i == dimension)
      return static_cast<uint16_t>(sgpr);
    ++sgpr;
  }
  return std::nullopt;
}

[[nodiscard]] bool moi_descriptor_enables_workgroup_id(const KD &desc, uint32_t dimension) {
  switch (dimension) {
  case 0:
    return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0;
  case 1:
    return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0;
  case 2:
    return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2,
                           kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0;
  default:
    return false;
  }
}

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources>
moi_descriptor_workgroup_sources(std::span<const uint8_t> image, uint64_t descriptor_file_offset,
                                 rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset) {
    errors.emplace_back("ConSan MOI workgroup-id descriptor exceeds ELF bytes");
    return std::nullopt;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250) {
    // These are the translated RDNA command-processor ABI TTMPs, not the
    // descriptor-selected entry SGPRs copied from them by the kernel prologue.
    // They remain dispatch inputs for the lifetime of the wave and are outside
    // both guest SGPR allocation and ConSan's ordinary SGPR scratch planner.
    return ConSanMoiWorkgroupSources{
        moi_descriptor_enables_workgroup_id(desc, 0)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridX), false}
            : ConSanMoiWorkgroupSource{},
        moi_descriptor_enables_workgroup_id(desc, 1)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridYz),
                                       /*shift_right_16=*/false, /*mask_low_16=*/true}
            : ConSanMoiWorkgroupSource{},
        moi_descriptor_enables_workgroup_id(desc, 2)
            ? ConSanMoiWorkgroupSource{ttmp_scalar_operand(kTtmpRdna4GridYz),
                                       /*shift_right_16=*/true}
            : ConSanMoiWorkgroupSource{},
    };
  }
  const auto sgpr_source = [](std::optional<uint16_t> sgpr) {
    return sgpr ? ConSanMoiWorkgroupSource{*sgpr, false} : ConSanMoiWorkgroupSource{};
  };
  return ConSanMoiWorkgroupSources{
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 0)),
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 1)),
      sgpr_source(moi_descriptor_workgroup_id_sgpr(desc, 2)),
  };
}

[[nodiscard]] uint32_t moi_descriptor_vgpr_granularity(const KD &desc) {
  return moi_descriptor_wavefront_size(desc) == 32 ? 8 : 4;
}

[[nodiscard]] uint32_t moi_descriptor_vgpr_allocation_count(const KD &desc) {
  const uint32_t granularity = moi_descriptor_vgpr_granularity(desc);
  const uint32_t granulated =
      AMDHSA_BITS_GET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t allocated = (granulated + 1u) * granularity;
  return std::min<uint32_t>(allocated, kMaxVgprs);
}

[[nodiscard]] uint32_t moi_descriptor_sgpr_allocation_count(const KD &desc) {
  const uint32_t granulated = AMDHSA_BITS_GET(
      desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  const uint32_t allocated = (granulated + 1u) * 8u;
  return std::min<uint32_t>(allocated, kMaxSgprs);
}

[[nodiscard]] std::optional<uint16_t> moi_descriptor_owner_shift(std::span<const uint8_t> image,
                                                                 uint64_t descriptor_file_offset,
                                                                 std::vector<std::string> &errors) {
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset) {
    errors.emplace_back("ConSan MOI owner derivation descriptor exceeds ELF bytes");
    return std::nullopt;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + descriptor_file_offset, sizeof(desc));
  return moi_descriptor_wavefront_size(desc) == 32 ? 5 : 6;
}

[[nodiscard]] std::optional<uint16_t> moi_kernel_owner_shift(std::span<const uint8_t> image,
                                                             const ConSanKernelInfo &kernel,
                                                             std::vector<std::string> &errors) {
  return moi_descriptor_owner_shift(image, kernel.descriptor_file_offset, errors);
}

[[nodiscard]] bool grow_moi_descriptor_vgpr_allocation(KD &desc, uint32_t required_count) {
  if (required_count == 0 || required_count > kMaxVgprs)
    return false;
  if (required_count <= moi_descriptor_vgpr_allocation_count(desc))
    return true;

  const uint32_t granularity = std::max<uint32_t>(moi_descriptor_vgpr_granularity(desc), 1u);
  const uint32_t rounded = ((required_count + granularity - 1u) / granularity) * granularity;
  if (rounded > kMaxVgprs)
    return false;
  const uint32_t granulated = rounded / granularity - 1u;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  granulated);
  return true;
}

[[nodiscard]] bool grow_moi_descriptor_sgpr_allocation(KD &desc, uint32_t required_count) {
  if (required_count == 0 || required_count > kMaxSgprs)
    return false;
  if (required_count <= moi_descriptor_sgpr_allocation_count(desc))
    return true;

  constexpr uint32_t kSgprGranularity = 8;
  const uint32_t rounded =
      ((required_count + kSgprGranularity - 1u) / kSgprGranularity) * kSgprGranularity;
  // s0:s105 are the ordinary scalar operand file; the descriptor expresses
  // allocation in eight-register granules. The final legal request therefore
  // encodes the 112-register granule even though instruction operands stop at
  // s105. Decoding clamps that granule back to kMaxSgprs above.
  constexpr uint32_t kMaxEncodedSgprs =
      ((kMaxSgprs + kSgprGranularity - 1u) / kSgprGranularity) * kSgprGranularity;
  if (rounded > kMaxEncodedSgprs)
    return false;
  const uint32_t granulated = rounded / kSgprGranularity - 1u;
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  granulated);
  return true;
}

[[nodiscard]] bool grow_moi_kernel_descriptor_vgprs(CodeObjectPatcher &patcher,
                                                    std::span<const uint8_t> image,
                                                    const ConSanKernelInfo &kernel,
                                                    uint32_t required_count,
                                                    std::vector<std::string> &errors) {
  if (kernel.descriptor_file_offset > image.size() ||
      sizeof(KD) > image.size() - kernel.descriptor_file_offset) {
    errors.emplace_back("ConSan MOI descriptor VGPR growth exceeds ELF bytes");
    return false;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + kernel.descriptor_file_offset, sizeof(desc));
  if (!grow_moi_descriptor_vgpr_allocation(desc, required_count)) {
    errors.emplace_back("ConSan MOI could not grow descriptor VGPR allocation");
    return false;
  }
  if (!patcher.patch_kernel_descriptor(kernel.descriptor_file_offset,
                                       {reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)})) {
    errors.emplace_back("ConSan MOI could not patch descriptor VGPR allocation");
    return false;
  }
  return true;
}

[[nodiscard]] bool grow_moi_kernel_descriptor_vgprs(CodeObjectPatcher &patcher,
                                                    std::span<const uint8_t> image,
                                                    ConSanResult &result, uint32_t required_count) {
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!grow_moi_kernel_descriptor_vgprs(patcher, image, kernel, required_count, result.errors)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool append_moi_delay_words(std::vector<uint32_t> &words, rj_code_arch_t arch,
                                          const ConSanOptions &options,
                                          std::vector<std::string> &errors,
                                          std::string_view context) {
  if (options.delay_nops == 0)
    return true;

  switch (options.delay_mode) {
  case ConSanDelayMode::Nop:
    for (uint32_t i = 0; i < options.delay_nops; ++i)
      words.push_back(build_s_nop(0, arch));
    return true;
  case ConSanDelayMode::Sleep:
    if (options.delay_nops > std::numeric_limits<uint16_t>::max()) {
      errors.emplace_back(std::string(context) +
                          " sleep delay immediate exceeds the 16-bit s_sleep field");
      return false;
    }
    words.push_back(build_s_sleep(static_cast<uint16_t>(options.delay_nops), arch));
    return true;
  case ConSanDelayMode::SleepVar:
    if (options.delay_var_ssrc > std::numeric_limits<uint8_t>::max()) {
      errors.emplace_back(std::string(context) +
                          " sleep_var source exceeds the 8-bit scalar source field");
      return false;
    }
    words.push_back(build_s_sleep_var(options.delay_var_ssrc, arch));
    return true;
  }

  errors.emplace_back(std::string(context) + " has unknown delay mode '" +
                      consan_delay_mode_name(options.delay_mode) + "'");
  return false;
}

[[nodiscard]] uint16_t inline_shadow_scratch_count(const ConSanOptions &options) {
  // Versioned exact publication retains the slot address, new packed access,
  // prior dispatch, ready version, and CAS operands through commit. The token
  // hash aliases the now-larger transaction window.
  return options.moi_exec_save_sgpr ? (options.moi_track_atomics ? 24u : 16u) : 11u;
}

[[nodiscard]] uint16_t inline_shadow_loop_scratch_count(const ConSanOptions &options,
                                                        const ConSanMoiCandidate &candidate) {
  return options.moi_inline_workgroup_shadow &&
                 candidate.width_bits > consan_moi_exact_shadow::granule_bytes * 8u
             ? 2u
             : 0u;
}

[[nodiscard]] bool validate_inline_shadow_exec_save_sgpr(const ConSanOptions &options,
                                                         std::vector<std::string> &errors) {
  if (!options.moi_exec_save_sgpr)
    return true;
  // Diagnostics use s[base:base+10]. Address-partitioned publication retains
  // incoming, pending, and current-group masks plus two scalar keys through
  // s[base+19]. Workgroup validation preserves the pre-filter EXEC mask in
  // s[base+20:base+21]; local publication retains the validated subset in
  // s[base+22:base+23]. Dynamic-stack spilling retains the caller's frame
  // base in s[base+24] only when a dynamic-stack spill is planned.
  const ConSanMoiReportBufferLayout layout = resolve_inline_report_layout(options);
  const uint16_t max_exec_save_sgpr =
      options.moi_dynamic_stack_spill       ? (layout.diagnostic_capacity != 0 ? 79u : 81u)
      : options.moi_inline_workgroup_shadow ? (layout.diagnostic_capacity != 0 ? 80u : 82u)
                                            : (layout.diagnostic_capacity != 0 ? 82u : 84u);
  if (*options.moi_exec_save_sgpr > max_exec_save_sgpr || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back("ConSan MOI inline-shadow diagnostics require an even "
                        "RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0.." +
                        std::to_string(max_exec_save_sgpr));
    return false;
  }
  return true;
}

[[nodiscard]] bool reject_optional_scratch_overlap(std::optional<uint16_t> value,
                                                   uint16_t scratch_vgpr,
                                                   std::string_view register_name,
                                                   std::vector<std::string> &errors) {
  if (!value || !overlaps_scratch_triplet(*value, scratch_vgpr))
    return false;
  errors.emplace_back(std::string("ConSan MOI probe scratch VGPRs overlap the ") +
                      std::string(register_name) + " VGPR");
  return true;
}

[[nodiscard]] std::optional<uint16_t>
candidate_lds_byte_offset_vgpr(const ConSanMoiCandidate &candidate,
                               std::vector<std::string> &errors) {
  if (!candidate.addr_vgpr) {
    errors.emplace_back("ConSan MOI probe requires an LDS address VGPR");
    return std::nullopt;
  }
  return *candidate.addr_vgpr;
}

[[nodiscard]] uint16_t candidate_payload_vgpr_count(const ConSanMoiCandidate &candidate) {
  if (candidate.width_bits == 0)
    return 1;
  const bool two_address = equals_any(
      candidate.mnemonic, {"ds_load_2addr_b32", "ds_load_2addr_b64", "ds_load_2addr_stride64_b32",
                           "ds_load_2addr_stride64_b64", "ds_store_2addr_b32", "ds_store_2addr_b64",
                           "ds_store_2addr_stride64_b32", "ds_store_2addr_stride64_b64"});
  const uint32_t per_address = std::max<uint32_t>((candidate.width_bits + 31u) / 32u, 1u);
  return static_cast<uint16_t>(per_address * (two_address ? 2u : 1u));
}

[[nodiscard]] bool moi_load_clobbers_address(const ConSanMoiCandidate &candidate) {
  if (!candidate.addr_vgpr || !candidate.dst_vgpr)
    return false;
  return range_overlaps(*candidate.addr_vgpr, 1u, *candidate.dst_vgpr,
                        candidate_payload_vgpr_count(candidate));
}

[[nodiscard]] uint16_t inline_shadow_scratch_count(const ConSanOptions &options,
                                                   const ConSanMoiCandidate &candidate) {
  return static_cast<uint16_t>(inline_shadow_scratch_count(options) +
                               inline_shadow_loop_scratch_count(options, candidate) +
                               (moi_load_clobbers_address(candidate) ? 1u : 0u));
}

[[nodiscard]] uint16_t direct_sampled_scratch_count(const ConSanOptions &options,
                                                    const ConSanMoiCandidate &candidate) {
  const uint16_t base_scratch_count = options.moi_sampled_check ? 7u : 5u;
  return static_cast<uint16_t>(base_scratch_count +
                               (options.moi_runtime_sample_stride > 1 ? 1u : 0u) +
                               (moi_load_clobbers_address(candidate) ? 1u : 0u) +
                               (options.automatic_moi_private_epoch ? 1u : 0u));
}

[[nodiscard]] bool reject_candidate_scratch_range_overlap(const ConSanMoiCandidate &candidate,
                                                          uint16_t scratch_vgpr,
                                                          uint16_t scratch_count,
                                                          std::vector<std::string> &errors) {
  auto address_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!address_vgpr)
    return true;
  const uint16_t address_vgpr_count =
      candidate.source == ConSanMoiCandidateSource::NativeLds ? 1u : 2u;
  if (range_overlaps(*address_vgpr, address_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the LDS address VGPRs");
    return true;
  }
  const uint16_t payload_vgpr_count = candidate_payload_vgpr_count(candidate);
  if (candidate.dst_vgpr &&
      range_overlaps(*candidate.dst_vgpr, payload_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the destination VGPRs");
    return true;
  }
  if (candidate.data_vgpr &&
      range_overlaps(*candidate.data_vgpr, payload_vgpr_count, scratch_vgpr, scratch_count)) {
    errors.emplace_back("ConSan MOI probe scratch VGPRs overlap the data VGPRs");
    return true;
  }
  return false;
}

struct MoiKernelResourceContext {
  const ConSanKernelInfo *kernel = nullptr;
  KernelCfgScope scope;
  std::unique_ptr<LivenessAnalysis> liveness;
  uint16_t current_vgpr_count = 0;
  uint16_t max_referenced_vgpr_count = 0;
  uint16_t current_sgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  bool has_indirect_vgpr_access = false;
  bool descriptor_valid = false;
};

[[nodiscard]] bool has_indirect_vgpr_access(const KernelCfgScope &scope) {
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    if (std::ranges::any_of(block->instructions(), [](const Instruction &inst) {
          return inst.mnemonic().starts_with("v_movrel");
        })) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] uint16_t max_referenced_vgpr_count(const KernelCfgScope &scope) {
  uint32_t max_count = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      const InstDefUse du(inst);
      const RegisterSet referenced = du.defs | du.uses;
      referenced.for_each([&](RegisterRef ref) {
        if (ref.cls == RegClass::VGPR)
          max_count = std::max<uint32_t>(max_count, static_cast<uint32_t>(ref.index) + 1u);
      });
    }
  }
  return static_cast<uint16_t>(std::min<uint32_t>(max_count, kMaxVgprs));
}

[[nodiscard]] uint16_t max_referenced_sgpr_count(const KernelCfgScope &scope) {
  uint32_t max_count = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      const InstDefUse du(inst);
      const RegisterSet referenced = du.defs | du.uses;
      referenced.for_each([&](RegisterRef ref) {
        if (ref.cls == RegClass::SGPR)
          max_count = std::max<uint32_t>(max_count, static_cast<uint32_t>(ref.index) + 1u);
      });
    }
  }
  return static_cast<uint16_t>(std::min<uint32_t>(max_count, kMaxSgprs));
}

[[nodiscard]] const Instruction *instruction_at(BasicBlock *block, uint64_t text_offset) {
  if (block == nullptr)
    return nullptr;
  const auto it =
      std::ranges::find_if(block->instructions(), [text_offset](const Instruction &inst) {
        return inst.src_loc() == text_offset;
      });
  return it == block->instructions().end() ? nullptr : &*it;
}

[[nodiscard]] uint16_t moi_access_scratch_vgpr_count(const ConSanOptions &options,
                                                     const ConSanMoiCandidate &candidate) {
  switch (options.moi_engine) {
  case ConSanMoiEngine::RecordReplay:
    return static_cast<uint16_t>((options.moi_dynamic_access_records ? 6u : 3u) +
                                 (moi_load_clobbers_address(candidate) ? 1u : 0u));
  case ConSanMoiEngine::InlineShadow:
    return inline_shadow_scratch_count(options, candidate);
  case ConSanMoiEngine::Sampled:
    return direct_sampled_scratch_count(options, candidate);
  }
  return 0;
}

struct MoiResourcePlanningState {
  std::span<const uint8_t> bytes;
  AmdGpuCodeObject code_object;
  std::unique_ptr<Decoder> decoder;
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  BlockOffsetIndex block_index;
  std::vector<MoiKernelResourceContext> contexts;
  bool valid = false;

  MoiResourcePlanningState(std::span<const uint8_t> input, rj_code_arch_t arch,
                           const ConSanOptions &options, const ConSanResult &result)
      : bytes(input), code_object(input.data(), input.size()), decoder(Decoder::create(arch)) {
    if (!code_object.is_valid() || !decoder)
      return;

    std::vector<uint64_t> leaders;
    leaders.reserve(result.kernels.size() + result.functions.size());
    std::vector<uint64_t> kernel_entries;
    kernel_entries.reserve(result.kernels.size());
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (!kernel.has_text_range)
        continue;
      leaders.push_back(kernel.entry_text_offset);
      kernel_entries.push_back(kernel.entry_text_offset);
    }
    for (const ConSanFunctionInfo &function : result.functions)
      leaders.push_back(function.entry_text_offset);
    for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
      leaders.push_back(range.text_offset);
      if (range.continuation_text_offset)
        leaders.push_back(*range.continuation_text_offset);
    }
    std::ranges::sort(leaders);
    leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());
    std::ranges::sort(kernel_entries);
    kernel_entries.erase(std::ranges::unique(kernel_entries).begin(), kernel_entries.end());

    std::vector<BasicBlock::CodeRange> code_ranges;
    code_ranges.reserve(code_object.functions().size());
    for (const AmdGpuFunctionInfo &function : code_object.functions()) {
      if (function.code_size != 0)
        code_ranges.push_back(
            {.start_offset = function.entry_text_offset, .size = function.code_size});
    }
    for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
      if (range.size != 0)
        code_ranges.push_back({.start_offset = range.text_offset, .size = range.size});
    }

    blocks = BasicBlock::build(code_object, *decoder, arch, leaders, code_ranges);
    block_index = build_block_offset_index(blocks);
    CodeObjectPatcher patcher(code_object);
    const std::span<const uint8_t> text = patcher.text_bytes();

    contexts.reserve(result.kernels.size());
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (!kernel.has_text_range)
        continue;
      std::vector<uint64_t> additional_entry_offsets;
      for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
        if (range.kernel_name != kernel.name)
          continue;
        if (block_for_offset(block_index, range.text_offset) != nullptr)
          additional_entry_offsets.push_back(range.text_offset);
        if (range.continuation_text_offset &&
            block_for_offset(block_index, *range.continuation_text_offset) != nullptr)
          additional_entry_offsets.push_back(*range.continuation_text_offset);
      }
      auto scope = build_kernel_cfg_scope(
          blocks, block_index,
          KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                             .additional_entry_offsets = additional_entry_offsets},
          kernel_entries, text);
      if (!scope)
        continue;

      MoiKernelResourceContext context;
      context.kernel = &kernel;
      context.scope = std::move(*scope);
      contexts.push_back(std::move(context));
      MoiKernelResourceContext &stored = contexts.back();
      stored.liveness = std::make_unique<LivenessAnalysis>(KernelBlockScope(stored.scope.blocks),
                                                           LivenessAnalysisOptions{},
                                                           stored.scope.liveness_edges);
      stored.max_referenced_vgpr_count = max_referenced_vgpr_count(stored.scope);
      stored.max_referenced_sgpr_count = std::max<uint16_t>(max_referenced_sgpr_count(stored.scope),
                                                            kernel.sgpr_count.value_or(0u));
      stored.has_indirect_vgpr_access = has_indirect_vgpr_access(stored.scope);

      if (kernel.descriptor_file_offset > bytes.size() ||
          sizeof(KD) > bytes.size() - kernel.descriptor_file_offset) {
        continue;
      }
      KD descriptor{};
      std::memcpy(&descriptor, bytes.data() + kernel.descriptor_file_offset, sizeof(descriptor));
      stored.current_vgpr_count =
          static_cast<uint16_t>(moi_descriptor_vgpr_allocation_count(descriptor));
      stored.current_sgpr_count =
          static_cast<uint16_t>(moi_descriptor_sgpr_allocation_count(descriptor));
      stored.original_private_segment_size = descriptor.private_segment_fixed_size;
      stored.descriptor_valid = true;
    }
    valid = true;
  }
};

[[nodiscard]] ConSanCandidateResourcePlan
plan_moi_resource_site(MoiResourcePlanningState &state, const ConSanOptions &options,
                       ConSanResourceSiteKind site_kind, size_t candidate_index,
                       uint64_t text_offset, std::optional<uint64_t> kernel_descriptor_file_offset,
                       uint16_t scratch_count) {
  ConSanCandidateResourcePlan plan;
  plan.site_kind = site_kind;
  plan.candidate_index = candidate_index;
  plan.text_offset = text_offset;
  plan.scratch_vgpr_count = scratch_count;
  if (!state.valid) {
    plan.reason = ConSanRegisterPlanReason::MissingInstruction;
    return plan;
  }

  BasicBlock *anchor_block = block_for_offset(state.block_index, text_offset);
  const Instruction *anchor = instruction_at(anchor_block, text_offset);
  if (anchor == nullptr) {
    plan.reason = ConSanRegisterPlanReason::MissingInstruction;
    return plan;
  }

  std::vector<MoiKernelResourceContext *> owners;
  for (MoiKernelResourceContext &context : state.contexts) {
    if (kernel_descriptor_file_offset &&
        context.kernel->descriptor_file_offset != *kernel_descriptor_file_offset) {
      continue;
    }
    if (std::ranges::find(context.scope.blocks, anchor_block) == context.scope.blocks.end())
      continue;
    owners.push_back(&context);
    plan.owner_descriptor_file_offsets.push_back(context.kernel->descriptor_file_offset);
  }
  if (owners.empty()) {
    plan.reason = ConSanRegisterPlanReason::MissingOwner;
    return plan;
  }

  plan.current_vgpr_count = kMaxVgprs;
  plan.current_sgpr_count = kMaxSgprs;
  RegisterSet live_before_all_owners;
  bool descriptors_valid = true;
  for (MoiKernelResourceContext *owner : owners) {
    plan.current_vgpr_count = std::min(plan.current_vgpr_count, owner->current_vgpr_count);
    plan.max_referenced_vgpr_count =
        std::max(plan.max_referenced_vgpr_count, owner->max_referenced_vgpr_count);
    plan.current_sgpr_count = std::min(plan.current_sgpr_count, owner->current_sgpr_count);
    plan.max_referenced_sgpr_count =
        std::max(plan.max_referenced_sgpr_count, owner->max_referenced_sgpr_count);
    plan.original_private_segment_size =
        std::max(plan.original_private_segment_size, owner->original_private_segment_size);
    live_before_all_owners |= owner->liveness->live_before(*anchor);
    // Relative VGPR instructions address a register array through M0. Their
    // dynamically selected elements are intentionally absent from ordinary
    // explicit-operand def/use sets, so a normal liveness proof can otherwise
    // misclassify a live array element as instrumentation scratch. Conservatively
    // reserve the whole guest allocation; the planner can grow the descriptor or
    // use the existing spill path instead of clobbering indirect guest state.
    if (owner->has_indirect_vgpr_access) {
      for (uint16_t index = 0; index < owner->current_vgpr_count; ++index)
        live_before_all_owners.expand({RegClass::VGPR, index, 1});
    }
    descriptors_valid &= owner->descriptor_valid;
  }
  plan.required_vgpr_count = plan.current_vgpr_count;
  if (!descriptors_valid) {
    plan.reason = ConSanRegisterPlanReason::InvalidDescriptor;
    return plan;
  }

  const InstDefUse anchor_def_use(*anchor);
  ConSanRegisterRequest request;
  request.reg_class = RegClass::VGPR;
  request.count = scratch_count;
  request.current_allocation_count = plan.current_vgpr_count;
  request.max_referenced_count = plan.max_referenced_vgpr_count;
  request.architecture_limit = kMaxVgprs;
  request.explicit_base = options.scratch_vgpr;
  request.force_spill = options.force_vgpr_spill;
  request.forbidden = anchor_def_use.defs | anchor_def_use.uses;
  if (!options.moi_owner_vgpr && options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId)
    request.forbidden.expand({RegClass::VGPR, kRdna4WorkitemIdX, 1});
  if (options.moi_owner_vgpr)
    request.forbidden.expand({RegClass::VGPR, *options.moi_owner_vgpr, 1});
  if (options.moi_epoch_vgpr)
    request.forbidden.expand({RegClass::VGPR, *options.moi_epoch_vgpr, 1});
  if (options.moi_workgroup_key_vgpr)
    request.forbidden.expand({RegClass::VGPR, *options.moi_workgroup_key_vgpr, 1});

  const ConSanRegisterPlan register_plan = plan_consan_registers(request, live_before_all_owners);
  plan.source = register_plan.source;
  plan.reason = register_plan.reason;
  plan.scratch_vgpr = register_plan.base;
  plan.required_vgpr_count = register_plan.required_descriptor_count;
  if (plan.source == ConSanRegisterAllocationSource::SpillRequired &&
      options.moi_engine != ConSanMoiEngine::InlineShadow &&
      std::ranges::any_of(owners, [](const MoiKernelResourceContext *owner) {
        return owner->kernel->uses_dynamic_stack.value_or(false);
      })) {
    plan.source = ConSanRegisterAllocationSource::Unsupported;
    plan.reason = ConSanRegisterPlanReason::DynamicStack;
    plan.scratch_vgpr.reset();
    plan.required_vgpr_count = plan.current_vgpr_count;
  }
  return plan;
}

void append_moi_resource_plans(std::span<const uint8_t> bytes, const ConSanOptions &options,
                               rj_code_arch_t arch, ConSanResult &result) {
  result.resource_plans.clear();
  result.resource_plans.reserve(result.moi_candidates.size());
  MoiResourcePlanningState state(bytes, arch, options, result);
  for (size_t candidate_index = 0; candidate_index < result.moi_candidates.size();
       ++candidate_index) {
    const ConSanMoiCandidate &candidate = result.moi_candidates[candidate_index];
    const uint16_t scratch_count = moi_access_scratch_vgpr_count(options, candidate);
    result.resource_plans.push_back(plan_moi_resource_site(
        state, options, ConSanResourceSiteKind::Access, candidate_index, candidate.text_offset,
        candidate.kernel_descriptor_file_offset, scratch_count));
  }
}

struct ResolvedMoiScratchPlan {
  uint16_t base = 0;
  uint16_t count = 0;
  uint16_t required_vgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  std::vector<uint64_t> owner_descriptor_file_offsets;
  ConSanRegisterAllocationSource source = ConSanRegisterAllocationSource::Unsupported;
};

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_candidate(const ConSanResult &result, const ConSanMoiCandidate &candidate) {
  if (result.moi_candidates.empty())
    return nullptr;
  const auto it = std::ranges::find_if(result.moi_candidates,
                                       [&](const auto &item) { return &item == &candidate; });
  if (it == result.moi_candidates.end())
    return nullptr;
  const size_t index = static_cast<size_t>(std::distance(result.moi_candidates.begin(), it));
  if (index >= result.resource_plans.size() ||
      result.resource_plans[index].candidate_index != index) {
    return nullptr;
  }
  return &result.resource_plans[index];
}

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_site(const ConSanResult &result, ConSanResourceSiteKind site_kind,
                       uint64_t text_offset) {
  const auto it = std::ranges::find_if(result.resource_plans, [&](const auto &plan) {
    return plan.site_kind == site_kind && plan.text_offset == text_offset;
  });
  return it == result.resource_plans.end() ? nullptr : &*it;
}

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch_plan(const ConSanCandidateResourcePlan &plan, const ConSanOptions &options,
                         uint16_t expected_count) {
  if (plan.scratch_vgpr && plan.scratch_vgpr_count == expected_count &&
      (plan.source == ConSanRegisterAllocationSource::Explicit ||
       plan.source == ConSanRegisterAllocationSource::LivenessDead ||
       plan.source == ConSanRegisterAllocationSource::DescriptorGrowth ||
       plan.source == ConSanRegisterAllocationSource::SpillRequired)) {
    ResolvedMoiScratchPlan resolved;
    resolved.base = *plan.scratch_vgpr;
    resolved.count = expected_count;
    resolved.required_vgpr_count = plan.required_vgpr_count;
    resolved.original_private_segment_size = plan.original_private_segment_size;
    resolved.owner_descriptor_file_offsets = plan.owner_descriptor_file_offsets;
    resolved.source = plan.source;
    if (options.moi_owner_vgpr) {
      resolved.required_vgpr_count = std::max<uint16_t>(
          resolved.required_vgpr_count, static_cast<uint16_t>(*options.moi_owner_vgpr + 1u));
    }
    if (options.moi_epoch_vgpr) {
      resolved.required_vgpr_count = std::max<uint16_t>(
          resolved.required_vgpr_count, static_cast<uint16_t>(*options.moi_epoch_vgpr + 1u));
    }
    if (options.moi_workgroup_key_vgpr) {
      resolved.required_vgpr_count =
          std::max<uint16_t>(resolved.required_vgpr_count,
                             static_cast<uint16_t>(*options.moi_workgroup_key_vgpr + 1u));
    }
    return resolved;
  }

  return std::nullopt;
}

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch(const ConSanResult &result, const ConSanMoiCandidate &candidate,
                    const ConSanOptions &options, uint16_t expected_count) {
  const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(result, candidate);
  return plan == nullptr ? std::nullopt : resolve_moi_scratch_plan(*plan, options, expected_count);
}

void apply_test_kernel_filter(std::vector<const ConSanMoiCandidate *> &candidates,
                              const ConSanOptions &options) {
  if (options.test_kernel_name_filter.empty())
    return;
  std::erase_if(candidates, [&](const ConSanMoiCandidate *candidate) {
    return candidate == nullptr ||
           candidate->container_name.find(options.test_kernel_name_filter) == std::string::npos;
  });
}

template <typename Candidate>
void apply_test_container_filter(std::vector<Candidate> &candidates, const ConSanOptions &options) {
  if (options.test_kernel_name_filter.empty())
    return;
  std::erase_if(candidates, [&](const Candidate &candidate) {
    return candidate.container_name.find(options.test_kernel_name_filter) == std::string::npos;
  });
}

[[nodiscard]] uint16_t moi_exec_save_sgpr_count(const ConSanOptions &options) {
  if (options.moi_engine == ConSanMoiEngine::Sampled)
    // Claim selection and optional wave sampling predicate on VCC. A winning
    // lane narrows EXEC while publishing, and every scalar predicate may
    // change SCC. Keep those snapshots separate from the immediate checker's
    // nested VCC pair.
    return options.moi_report_buffer_address ? (options.moi_track_atomics ? 10u : 7u) : 0u;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    if (!options.moi_report_buffer_address)
      return 0;
    const ConSanMoiReportBufferLayout layout = resolve_inline_report_layout(options);
    (void)layout;
    // Diagnostic state through base+10, padding at base+11, three EXEC-mask
    // pairs, two scalar partition keys, and the pre-workgroup-filter EXEC
    // pair. A dynamic-stack spill additionally retains its caller frame base.
    if (options.moi_dynamic_stack_spill)
      return 25u;
    return options.moi_inline_workgroup_shadow ? 24u : 20u;
  }
  // One EXEC-save pair, one VCC-save pair, and one SCC snapshot.
  return options.moi_report_buffer_address ? 5u : 0u;
}

[[nodiscard]] uint16_t moi_dispatch_state_sgpr_count(const ConSanOptions &options) {
  return options.moi_engine == ConSanMoiEngine::Sampled && options.moi_runtime_sample_stride > 1u
             ? 3u
             : 2u;
}

[[nodiscard]] std::optional<uint16_t>
moi_runtime_sample_counter_sgpr(const ConSanOptions &options) {
  if (moi_dispatch_state_sgpr_count(options) != 3u || !options.moi_dispatch_id_sgpr)
    return std::nullopt;
  return static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 2u);
}

struct MoiSpecialStateSgprs {
  uint16_t vcc_save_sgpr = 0;
  uint16_t scc_save_sgpr = 0;
};

struct MoiIndirectJumpSgprs {
  uint16_t pc_sgpr = 0;
  uint16_t scc_save_sgpr = 0;
};

[[nodiscard]] std::optional<MoiSpecialStateSgprs>
moi_special_state_sgprs(const ConSanOptions &options) {
  if (!options.moi_exec_save_sgpr)
    return std::nullopt;
  const uint16_t vcc_offset = options.moi_engine == ConSanMoiEngine::InlineShadow ? 8u : 2u;
  return MoiSpecialStateSgprs{
      .vcc_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + vcc_offset),
      .scc_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + vcc_offset + 2u),
  };
}

[[nodiscard]] std::optional<MoiIndirectJumpSgprs>
moi_indirect_jump_sgprs(const ConSanOptions &options) {
  const auto special_state = moi_special_state_sgprs(options);
  if (!options.moi_exec_save_sgpr || !special_state)
    return std::nullopt;
  if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
    return MoiIndirectJumpSgprs{
        .pc_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 12u),
        .scc_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 10u),
    };
  }
  // Record/Replay reserves five SGPRs. Its indirect island runs before the
  // body snapshots EXEC/VCC/SCC, and its return runs after those snapshots
  // have been restored, so the EXEC-save pair and SCC slot are free scratch
  // at both boundaries. Do not address Inline's wider +10/+12 slots here.
  return MoiIndirectJumpSgprs{
      .pc_sgpr = *options.moi_exec_save_sgpr,
      .scc_save_sgpr = special_state->scc_save_sgpr,
  };
}

[[nodiscard]] bool append_save_moi_special_state(std::vector<uint32_t> &words,
                                                 const ConSanOptions &options,
                                                 rj_code_arch_t arch) {
  const auto registers = moi_special_state_sgprs(options);
  if (!registers)
    return false;
  const auto save_scc = build_rdna4_s_cselect_b32(
      registers->scc_save_sgpr, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
  const auto save_vcc = build_s_mov_b64(registers->vcc_save_sgpr, kRdna4VccLo, arch);
  if (!save_scc || !save_vcc)
    return false;
  // Snapshot SCC before any instrumentation instruction can modify it.
  words.push_back(*save_scc);
  words.push_back(*save_vcc);
  return true;
}

[[nodiscard]] bool append_restore_moi_special_state(std::vector<uint32_t> &words,
                                                    const ConSanOptions &options,
                                                    rj_code_arch_t arch) {
  const auto registers = moi_special_state_sgprs(options);
  if (!registers)
    return false;
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, registers->vcc_save_sgpr, arch);
  const auto restore_scc =
      build_rdna4_s_cmp_lg_u32(registers->scc_save_sgpr, scalar_positive_inline_u32(0), arch);
  if (!restore_vcc || !restore_scc)
    return false;
  words.push_back(*restore_vcc);
  // Keep this last: every scalar comparison or saveexec before it may write SCC.
  words.push_back(*restore_scc);
  return true;
}

[[nodiscard]] bool configure_automatic_moi_exec_save_sgprs(ConSanOptions &options,
                                                           ConSanResult &result,
                                                           std::span<const uint8_t> bytes,
                                                           rj_code_arch_t arch) {
  if (options.moi_exec_save_sgpr)
    return false;
  const uint16_t count = moi_exec_save_sgpr_count(options);
  if (count == 0)
    return false;

  uint32_t first = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    first = std::max<uint32_t>(first, plan.max_referenced_sgpr_count);
  }
  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic EXEC-save SGPRs without an owned candidate "
        "scope");
    return false;
  }

  first = util::align_up(first, 2u);
  for (uint32_t base = first; base + count <= kMaxSgprs; base += 2u) {
    // Dynamic-stack kernels use s32 as the stack top and s33 as the current
    // frame base even when those architectural roles are absent from ordinary
    // decoded operand liveness. Never assign either register to persistent
    // Inline scalar state.
    if (options.moi_dynamic_stack_spill &&
        range_overlaps(static_cast<uint16_t>(base), count, /*stack top=*/32u,
                       /*stack registers=*/2u)) {
      continue;
    }
    if (options.moi_owner_sgpr &&
        range_overlaps(static_cast<uint16_t>(base), count, *options.moi_owner_sgpr, 1u)) {
      continue;
    }
    if (options.moi_dispatch_id_sgpr &&
        range_overlaps(static_cast<uint16_t>(base), count, *options.moi_dispatch_id_sgpr,
                       moi_dispatch_state_sgpr_count(options))) {
      continue;
    }
    options.moi_exec_save_sgpr = static_cast<uint16_t>(base);
    options.automatic_moi_exec_save_sgprs = true;
    result.resolved_moi_exec_save_sgpr = options.moi_exec_save_sgpr;
    result.moi_exec_save_sgprs_automatic = true;
    result.warnings.emplace_back("ConSan MOI automatically assigned EXEC-save SGPRs s" +
                                 std::to_string(base) + ":s" + std::to_string(base + count - 1u));
    return true;
  }

  // A high-numbered scalar operand anywhere in a kernel does not imply that
  // every lower SGPR is live at every instrumentation site. Large generated
  // kernels commonly use s104 for a short-lived address calculation while a
  // five-register window is dead throughout all LDS/synchronization sites.
  // Prove one common window with the existing CFG liveness before falling
  // back to an unsupported resource result.
  MoiResourcePlanningState state(bytes, arch, options, result);
  RegisterSet unavailable;
  bool saw_site = false;
  const auto include_site = [&](uint64_t text_offset) {
    BasicBlock *block = block_for_offset(state.block_index, text_offset);
    const Instruction *instruction = instruction_at(block, text_offset);
    if (block == nullptr || instruction == nullptr)
      return;
    for (const MoiKernelResourceContext &context : state.contexts) {
      if (std::ranges::find(context.scope.blocks, block) == context.scope.blocks.end())
        continue;
      unavailable |= context.liveness->live_before(*instruction);
      const InstDefUse def_use(*instruction);
      unavailable |= def_use.defs | def_use.uses;
      saw_site = true;
    }
  };
  for (const ConSanMoiCandidate &candidate : result.moi_candidates)
    include_site(candidate.text_offset);
  for (const ConSanSiteDispositionRecord &site : result.site_dispositions) {
    if (site.disposition == ConSanSiteDisposition::Supported)
      include_site(site.text_offset);
  }
  if (options.moi_init_owner_epoch) {
    for (const ConSanKernelInfo &kernel : result.kernels)
      include_site(kernel.entry_text_offset);
  }
  if (state.valid && saw_site) {
    const auto unavailable_intersects = [&](uint16_t base) {
      for (uint16_t i = 0; i < count; ++i) {
        if (unavailable.contains({RegClass::SGPR, static_cast<uint16_t>(base + i), 1u})) {
          return true;
        }
      }
      return false;
    };
    for (uint32_t base = 0; base + count <= kMaxSgprs; base += 2u) {
      if (unavailable_intersects(static_cast<uint16_t>(base)) ||
          (options.moi_dynamic_stack_spill &&
           range_overlaps(static_cast<uint16_t>(base), count, /*stack top=*/32u,
                          /*stack registers=*/2u)) ||
          (options.moi_owner_sgpr &&
           range_overlaps(static_cast<uint16_t>(base), count, *options.moi_owner_sgpr, 1u)) ||
          (options.moi_dispatch_id_sgpr &&
           range_overlaps(static_cast<uint16_t>(base), count, *options.moi_dispatch_id_sgpr,
                          moi_dispatch_state_sgpr_count(options)))) {
        continue;
      }
      options.moi_exec_save_sgpr = static_cast<uint16_t>(base);
      options.automatic_moi_exec_save_sgprs = true;
      result.resolved_moi_exec_save_sgpr = options.moi_exec_save_sgpr;
      result.moi_exec_save_sgprs_automatic = true;
      result.warnings.emplace_back(
          "ConSan MOI automatically assigned liveness-dead EXEC-save SGPRs s" +
          std::to_string(base) + ":s" + std::to_string(base + count - 1u));
      return true;
    }
  }

  result.warnings.emplace_back(
      "ConSan MOI could not place a fresh automatic EXEC-save SGPR window");
  return false;
}

[[nodiscard]] bool configure_automatic_moi_owner_sgpr(ConSanOptions &options,
                                                      ConSanResult &result) {
  if (options.moi_owner_source != ConSanMoiOwnerSource::HwId || options.moi_owner_sgpr)
    return false;

  uint32_t first = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    first = std::max<uint32_t>(first, plan.max_referenced_sgpr_count);
  }
  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive an automatic hw_id owner SGPR without an owned candidate "
        "scope");
    return false;
  }

  for (uint32_t sgpr = first; sgpr < kMaxSgprs; ++sgpr) {
    if (options.moi_exec_save_sgpr &&
        range_overlaps(static_cast<uint16_t>(sgpr), 1u, *options.moi_exec_save_sgpr,
                       moi_exec_save_sgpr_count(options))) {
      continue;
    }
    options.moi_owner_sgpr = static_cast<uint16_t>(sgpr);
    options.automatic_moi_owner_sgpr = true;
    result.resolved_moi_owner_sgpr = options.moi_owner_sgpr;
    result.moi_owner_sgpr_automatic = true;
    result.warnings.emplace_back("ConSan MOI automatically assigned hw_id owner temporary s" +
                                 std::to_string(sgpr));
    return true;
  }

  result.warnings.emplace_back("ConSan MOI could not place a fresh automatic hw_id owner SGPR");
  return false;
}

[[nodiscard]] bool configure_automatic_moi_dispatch_id_sgprs(ConSanOptions &options,
                                                             ConSanResult &result) {
  const bool has_inline_dispatch_consumer =
      options.moi_engine == ConSanMoiEngine::InlineShadow &&
      std::ranges::any_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        return !plan.owner_descriptor_file_offsets.empty() &&
               (plan.site_kind == ConSanResourceSiteKind::Access ||
                plan.site_kind == ConSanResourceSiteKind::Atomic);
      });
  const bool needs_dispatch_id =
      has_inline_dispatch_consumer ||
      (options.moi_engine == ConSanMoiEngine::Sampled && options.moi_runtime_sample_stride > 1u);
  if (!needs_dispatch_id || !options.moi_report_buffer_address || options.moi_dispatch_id_sgpr) {
    return false;
  }

  // Sixteen initialized user SGPRs plus X/Y/Z and WORKGROUP_INFO is the
  // largest legal entry-input window after insertion. Keeping the persistent
  // pair above that window makes it safe for every per-kernel preload map.
  uint32_t first = 20u;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    first = std::max<uint32_t>(first, plan.max_referenced_sgpr_count);
  }
  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic dispatch-ID SGPRs without an owned candidate "
        "scope");
    return false;
  }

  first = util::align_up(first, 2u);
  const uint16_t dispatch_state_count = moi_dispatch_state_sgpr_count(options);
  for (uint32_t base = first; base + dispatch_state_count <= kMaxSgprs; base += 2u) {
    if (options.moi_owner_sgpr && range_overlaps(static_cast<uint16_t>(base), dispatch_state_count,
                                                 *options.moi_owner_sgpr, 1u)) {
      continue;
    }
    if (options.moi_exec_save_sgpr &&
        range_overlaps(static_cast<uint16_t>(base), dispatch_state_count,
                       *options.moi_exec_save_sgpr, moi_exec_save_sgpr_count(options))) {
      continue;
    }
    options.moi_dispatch_id_sgpr = static_cast<uint16_t>(base);
    options.automatic_moi_dispatch_id_sgprs = true;
    result.resolved_moi_dispatch_id_sgpr = options.moi_dispatch_id_sgpr;
    result.moi_dispatch_id_sgprs_automatic = true;
    result.warnings.emplace_back("ConSan MOI automatically assigned persistent dispatch-ID "
                                 "state SGPRs s" +
                                 std::to_string(base) + ":s" +
                                 std::to_string(base + dispatch_state_count - 1u));
    return true;
  }

  result.outcome = ConSanTransformOutcome::Unsupported;
  result.warnings.emplace_back(
      "ConSan MOI could not place a persistent hardware dispatch-ID SGPR pair");
  return false;
}

[[nodiscard]] bool validate_moi_dispatch_id_sgprs(const ConSanOptions &options,
                                                  ConSanResult &result) {
  if (!options.moi_dispatch_id_sgpr)
    return true;
  const uint16_t base = *options.moi_dispatch_id_sgpr;
  const uint16_t dispatch_state_count = moi_dispatch_state_sgpr_count(options);
  const bool overlaps_owner = options.moi_owner_sgpr && range_overlaps(base, dispatch_state_count,
                                                                       *options.moi_owner_sgpr, 1u);
  const bool overlaps_exec = options.moi_exec_save_sgpr &&
                             range_overlaps(base, dispatch_state_count, *options.moi_exec_save_sgpr,
                                            moi_exec_save_sgpr_count(options));
  if (base < 20u || base % 2u != 0u ||
      static_cast<uint32_t>(base) + dispatch_state_count > kMaxSgprs || overlaps_owner ||
      overlaps_exec) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan MOI persistent dispatch-ID SGPR pair is unaligned, overlaps entry/scratch "
        "state, or exceeds the architectural SGPR file");
    return false;
  }
  return true;
}

[[nodiscard]] bool configure_automatic_moi_persistent_vgprs(ConSanOptions &options,
                                                            ConSanResult &result) {
  if (options.moi_owner_vgpr || options.moi_epoch_vgpr || options.moi_workgroup_key_vgpr ||
      options.moi_sample_sequence_vgpr)
    return false;
  const bool has_supported_sampled_sync =
      options.moi_engine == ConSanMoiEngine::Sampled &&
      std::ranges::any_of(result.resource_plans,
                          [](const ConSanCandidateResourcePlan &plan) {
                            return plan.site_kind == ConSanResourceSiteKind::Barrier ||
                                   plan.site_kind == ConSanResourceSiteKind::Atomic;
                          }) &&
      std::ranges::any_of(result.site_dispositions, [&](const ConSanSiteDispositionRecord &site) {
        if ((site.site_kind != ConSanResourceSiteKind::Barrier &&
             site.site_kind != ConSanResourceSiteKind::Atomic) ||
            site.disposition != ConSanSiteDisposition::Supported)
          return false;
        return std::ranges::any_of(
            result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
              return plan.site_kind == site.site_kind && plan.text_offset == site.text_offset &&
                     plan.source != ConSanRegisterAllocationSource::Unsupported &&
                     !plan.owner_descriptor_file_offsets.empty();
            });
      });
  // Access-only Sampled objects select waves with the persistent dispatch
  // counter and derive the access owner locally. A per-workitem sequence is
  // only needed when causal synchronization state is actually represented.
  const bool needs_sample_sequence = options.moi_engine == ConSanMoiEngine::Sampled &&
                                     options.moi_runtime_sample_stride > 1u &&
                                     has_supported_sampled_sync;
  const bool needs_workgroup_key =
      options.moi_engine == ConSanMoiEngine::InlineShadow &&
      std::ranges::any_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        // Inline sync dispositions are appended only after persistent-state
        // placement. Its operational resource plans have already filtered
        // out semantically unsupported atomics, so they are the authoritative
        // pre-placement signal here.
        return plan.site_kind == ConSanResourceSiteKind::Atomic &&
               plan.source != ConSanRegisterAllocationSource::Unsupported &&
               !plan.owner_descriptor_file_offsets.empty();
      });
  if (options.moi_engine == ConSanMoiEngine::Sampled && !needs_sample_sequence &&
      !has_supported_sampled_sync) {
    // Access-only Sampled probes derive their owner locally and encode epoch
    // zero. If every requested synchronization site failed semantic
    // qualification, persistent owner/epoch state has no consumer. Avoid a
    // needless descriptor/prologue mutation while retaining the unsupported
    // site dispositions as explicit coverage evidence.
    if (options.moi_init_owner_epoch || options.moi_track_barriers || options.moi_track_atomics) {
      result.warnings.emplace_back(
          "ConSan MOI sampled skipped persistent owner/epoch state because no sync site was "
          "supported");
    }
    options.moi_init_owner_epoch = false;
    return false;
  }
  const bool needs_persistent_state =
      options.moi_init_owner_epoch ||
      (options.moi_track_atomics && options.moi_engine != ConSanMoiEngine::RecordReplay) ||
      (options.moi_engine == ConSanMoiEngine::Sampled &&
       (options.moi_track_barriers || options.moi_runtime_sample_stride > 1u)) ||
      options.moi_engine == ConSanMoiEngine::InlineShadow;
  if (!needs_persistent_state)
    return false;
  if (!options.moi_report_buffer_address)
    return false;

  uint32_t persistent_base = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    persistent_base = std::max<uint32_t>(persistent_base, plan.max_referenced_vgpr_count);
  }

  if (!found_owned_scope) {
    result.warnings.emplace_back(
        "ConSan MOI could not derive automatic persistent VGPRs without an owned candidate "
        "scope");
    return false;
  }
  const uint32_t persistent_count = 2u + static_cast<uint32_t>(needs_sample_sequence) +
                                    static_cast<uint32_t>(needs_workgroup_key);
  const uint32_t persistent_end = persistent_base + persistent_count;
  const bool would_grow_descriptor =
      std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        return !plan.owner_descriptor_file_offsets.empty() &&
               plan.source != ConSanRegisterAllocationSource::Unsupported &&
               persistent_end > plan.current_vgpr_count;
      });
  const bool has_dynamic_stack_owner =
      std::ranges::any_of(result.kernels, [&](const ConSanKernelInfo &kernel) {
        if (!kernel.uses_dynamic_stack.value_or(false))
          return false;
        return std::ranges::any_of(
            result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
              return plan.source != ConSanRegisterAllocationSource::Unsupported &&
                     std::ranges::find(plan.owner_descriptor_file_offsets,
                                       kernel.descriptor_file_offset) !=
                         plan.owner_descriptor_file_offsets.end();
            });
      });
  const size_t record_replay_barrier_sites =
      options.moi_engine == ConSanMoiEngine::RecordReplay && options.moi_track_barriers
          ? std::ranges::count_if(result.resource_plans,
                                  [](const ConSanCandidateResourcePlan &plan) {
                                    return plan.site_kind == ConSanResourceSiteKind::Barrier &&
                                           plan.source !=
                                               ConSanRegisterAllocationSource::Unsupported &&
                                           !plan.owner_descriptor_file_offsets.empty();
                                  })
          : 0u;
  const size_t record_replay_access_sites =
      options.moi_engine == ConSanMoiEngine::RecordReplay
          ? std::ranges::count_if(result.resource_plans,
                                  [](const ConSanCandidateResourcePlan &plan) {
                                    return plan.site_kind == ConSanResourceSiteKind::Access &&
                                           plan.source !=
                                               ConSanRegisterAllocationSource::Unsupported &&
                                           !plan.owner_descriptor_file_offsets.empty();
                                  })
          : 0u;
  // This is an automatic operating-point choice, not a user-facing limit.
  // Small barrier inventories benefit from compact persistent-epoch barriers;
  // larger barrier-dense Record/Replay objects benefit more from private-epoch
  // access coalescing. Qwen's 28 barrier sites and the access-heavy TP
  // workloads avoid millions of dynamic barrier records.
  constexpr size_t kCompactRecordReplayBarrierSiteLimit = 32u;
  const bool prefer_compact_record_replay_barriers =
      record_replay_barrier_sites > 0u &&
      (record_replay_barrier_sites <= kCompactRecordReplayBarrierSiteLimit ||
       record_replay_access_sites >= 2u * record_replay_barrier_sites);
  // Private epoch storage is address-free scratch and therefore cannot be
  // composed with a guest dynamic stack. If persistent state fits in the
  // architectural VGPR file, grow those owner descriptors instead. The
  // rebuilt resource plans reserve the selected tail VGPRs before allocating
  // per-site scratch, so this growth cannot overlap instrumentation scratch.
  // A bounded Record/Replay barrier inventory is another exception: without
  // persistent epoch state it requires one dynamic record per participating
  // workitem and barrier execution. That can turn a small static inventory
  // into millions of records. Prefer bounded descriptor growth so those
  // barriers use the compact inline-epoch path instead.
  // A synchronization-aware Sampled probe must preserve one owner identity
  // from kernel entry through both access and sync sites. Its private-epoch
  // fallback otherwise derives the access owner from guest-reusable v0 and
  // can publish the reserved owner zero on real compiler output. Access-only
  // Sampled objects retain the cheaper private-state choice.
  const bool sampled_requires_persistent_owner =
      options.moi_engine == ConSanMoiEngine::Sampled && has_supported_sampled_sync;
  // InlineShadow reads owner state on every instrumented LDS access. A
  // scratch-backed owner made a complete multi-kernel CLIP invocation spend
  // more than a minute repeatedly loading that state, whereas two persistent
  // tail VGPRs complete the same clean row in about 1.5 seconds. Prefer the
  // descriptor-backed representation whenever it fits; private state remains
  // the capacity fallback for near-full VGPR files.
  const bool prefer_private_epoch = would_grow_descriptor && !has_dynamic_stack_owner &&
                                    !prefer_compact_record_replay_barriers &&
                                    !needs_workgroup_key && !sampled_requires_persistent_owner &&
                                    options.moi_engine != ConSanMoiEngine::InlineShadow;
  if (options.force_private_epoch || persistent_end > kMaxVgprs || prefer_private_epoch) {
    options.moi_init_owner_epoch = true;
    options.automatic_moi_private_epoch = true;
    result.moi_private_epoch_automatic = true;
    result.warnings.emplace_back(
        std::string("ConSan MOI automatically selected derived-owner/private-"
                    "epoch persistent state") +
        (would_grow_descriptor ? " to preserve the guest VGPR allocation" : ""));
    return true;
  }

  options.moi_owner_vgpr = static_cast<uint16_t>(persistent_base);
  options.moi_epoch_vgpr = static_cast<uint16_t>(persistent_base + 1u);
  uint16_t next_persistent_vgpr = static_cast<uint16_t>(persistent_base + 2u);
  if (needs_sample_sequence)
    options.moi_sample_sequence_vgpr = next_persistent_vgpr++;
  if (needs_workgroup_key)
    options.moi_workgroup_key_vgpr = next_persistent_vgpr++;
  options.moi_init_owner_epoch = true;
  options.automatic_moi_persistent_vgprs = true;
  result.resolved_moi_owner_vgpr = options.moi_owner_vgpr;
  result.resolved_moi_epoch_vgpr = options.moi_epoch_vgpr;
  result.resolved_moi_workgroup_key_vgpr = options.moi_workgroup_key_vgpr;
  result.resolved_moi_sample_sequence_vgpr = options.moi_sample_sequence_vgpr;
  result.moi_persistent_vgprs_automatic = true;
  result.warnings.emplace_back(
      "ConSan MOI automatically assigned persistent owner/epoch" +
      std::string(needs_sample_sequence ? "/sample-sequence" : "") +
      std::string(needs_workgroup_key ? "/workgroup-key" : "") + " VGPRs v" +
      std::to_string(*options.moi_owner_vgpr) + ":v" + std::to_string(next_persistent_vgpr - 1u) +
      (would_grow_descriptor && has_dynamic_stack_owner
           ? " with descriptor growth because private epoch storage is incompatible with a "
             "dynamic-stack owner"
           : ""));
  return true;
}

[[nodiscard]] bool automatic_moi_persistent_state(const ConSanOptions &options) {
  return options.automatic_moi_persistent_vgprs || options.automatic_moi_private_epoch;
}

using MoiDescriptorVgprRequirements = std::unordered_map<uint64_t, uint16_t>;
using MoiDescriptorSgprRequirements = std::unordered_map<uint64_t, uint16_t>;
using MoiDescriptorPrivateRequirements = std::unordered_map<uint64_t, uint32_t>;
using MoiDescriptorLdsRequirements = std::unordered_map<uint64_t, uint32_t>;
using MoiSpillManagers = std::unordered_map<uint64_t, SpillManager>;

void note_descriptor_requirements(MoiDescriptorVgprRequirements &requirements,
                                  const ResolvedMoiScratchPlan &plan) {
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, plan.required_vgpr_count);
    if (!inserted)
      it->second = std::max(it->second, plan.required_vgpr_count);
  }
}

void note_moi_sgpr_requirements(MoiDescriptorSgprRequirements &requirements,
                                const ResolvedMoiScratchPlan &plan, const ConSanOptions &options) {
  uint16_t required_count = 0;
  if (options.moi_exec_save_sgpr) {
    const uint16_t count = moi_exec_save_sgpr_count(options);
    required_count = static_cast<uint16_t>(*options.moi_exec_save_sgpr + count);
  }
  if (options.moi_owner_source == ConSanMoiOwnerSource::HwId && options.moi_owner_sgpr) {
    required_count =
        std::max<uint16_t>(required_count, static_cast<uint16_t>(*options.moi_owner_sgpr + 1u));
  }
  if (options.moi_dispatch_id_sgpr) {
    required_count = std::max<uint16_t>(
        required_count, static_cast<uint16_t>(*options.moi_dispatch_id_sgpr +
                                              moi_dispatch_state_sgpr_count(options)));
  }
  if (required_count == 0)
    return;
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, required_count);
    if (!inserted)
      it->second = std::max(it->second, required_count);
  }
}

void note_spill_descriptor_requirements(MoiDescriptorPrivateRequirements &requirements,
                                        const ResolvedMoiScratchPlan &plan,
                                        const VgprSpillSequence &spill) {
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] = requirements.emplace(descriptor_offset, spill.total_private_bytes);
    if (!inserted)
      it->second = std::max(it->second, spill.total_private_bytes);
  }
}

void note_moi_lds_requirements(MoiDescriptorLdsRequirements &requirements,
                               const ResolvedMoiScratchPlan &plan,
                               const ConSanMoiWorkgroupShadowLayout &layout) {
  for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
    auto [it, inserted] =
        requirements.emplace(descriptor_offset, layout.required_group_segment_size);
    if (!inserted)
      it->second = std::max(it->second, layout.required_group_segment_size);
  }
}

[[nodiscard]] const ConSanKernelInfo *kernel_for_descriptor(const ConSanResult &result,
                                                            uint64_t descriptor_offset) {
  const auto it = std::ranges::find_if(result.kernels, [descriptor_offset](const auto &kernel) {
    return kernel.descriptor_file_offset == descriptor_offset;
  });
  return it == result.kernels.end() ? nullptr : &*it;
}
