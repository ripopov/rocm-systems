struct BarrierRecordCandidate {
  std::string container_name;
  ConSanBarrierSite site;
  std::optional<uint64_t> kernel_descriptor_file_offset;
};

[[nodiscard]] bool is_rocclr_runtime_kernel_name(std::string_view name) {
  return name.starts_with("__amd_rocclr_");
}

void append_barrier_epoch_candidates(const ConSanKernelInfo &kernel,
                                     std::vector<BarrierRecordCandidate> &candidates) {
  if (is_rocclr_runtime_kernel_name(kernel.name))
    return;
  for (const ConSanBarrierSite &site : kernel.barrier_sites)
    candidates.push_back({"kernel:" + kernel.name, site, kernel.descriptor_file_offset});
}

void append_barrier_epoch_candidates(const ConSanFunctionInfo &function,
                                     std::vector<BarrierRecordCandidate> &candidates) {
  for (const ConSanBarrierSite &site : function.barrier_sites)
    candidates.push_back({"function:" + function.name, site, std::nullopt});
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_barrier_record_cave_words(
    std::span<const uint8_t> bytes, const BarrierRecordCandidate &candidate,
    const ConSanOptions &options, const VgprSpillSequence *spill, rj_code_arch_t arch,
    uint32_t barrier_record_capacity, size_t barrier_records_offset, uint32_t original_barrier_word,
    uint64_t cave_text_offset, uint64_t return_text_offset, std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI barrier record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + 6u > kMaxVgprs) {
    errors.emplace_back("ConSan MOI barrier record patch needs six scratch VGPRs");
    return std::nullopt;
  }
  if (!options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI barrier record patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return std::nullopt;
  }
  if (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back(
        "ConSan MOI barrier record patch requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..100");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr, 6,
                                            "MOI owner", errors))
    return std::nullopt;

  std::optional<uint16_t> derived_owner_vgpr;
  std::optional<uint32_t> derived_owner_word;
  if (!options.moi_owner_vgpr) {
    if (!candidate.kernel_descriptor_file_offset) {
      errors.emplace_back(
          "ConSan MOI barrier record patch requires RJ_CONSAN_MOI_OWNER_VGPR for function "
          "barriers");
      return std::nullopt;
    }
    const auto owner_shift =
        moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, errors);
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    const auto owner_init = build_v_lshrrev_b32_e32(
        value_vgpr, scalar_positive_inline_u32(*owner_shift), kRdna4WorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI barrier record patch could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_word = *owner_init;
  }
  ConSanMoiWorkgroupSources workgroup_sources;
  if (candidate.kernel_descriptor_file_offset) {
    const auto descriptor_workgroup_sources = moi_descriptor_workgroup_sources(
        bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!descriptor_workgroup_sources)
      return std::nullopt;
    workgroup_sources = *descriptor_workgroup_sources;
  }

  std::vector<uint32_t> words;
  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t barrier_record_base = base + barrier_records_offset;
  const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t lane_rank_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  constexpr uint16_t kScalarInlineMinusOne = 0xC1;

  words.reserve(160 + (spill ? spill->save_words.size() + spill->restore_words.size() : 0u));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (!append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not save VCC/SCC");
    return std::nullopt;
  }
  const auto mbcnt_lo = build_v_mbcnt_lo_u32_b32(lane_rank_vgpr, kScalarInlineMinusOne,
                                                 scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = build_v_mbcnt_hi_u32_b32(lane_rank_vgpr, kScalarInlineMinusOne,
                                                 vector_source_vgpr(lane_rank_vgpr), arch);
  const auto first_active_lane =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
  const auto save_exec = build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  if (!mbcnt_lo || !mbcnt_hi || !first_active_lane || !save_exec) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode EXEC narrowing");
    return std::nullopt;
  }
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*save_exec);

  if (!append_atomic_fetch_add_one_u32(words,
                                       base + offsetof(ConSanMoiReportHeader, barrier_record_count),
                                       slot_vgpr, *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode dynamic slot reserve");
    return std::nullopt;
  }

  const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  const auto mov_capacity = build_v_mov_b32_e64_literal(value_vgpr, barrier_record_capacity, arch);
  const auto slot_in_capacity =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(value_vgpr), slot_vgpr, arch);
  if (!mov_capacity || !slot_in_capacity) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode capacity guard");
    return std::nullopt;
  }
  words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
  words.push_back(*slot_in_capacity);

  std::vector<uint32_t> record_words;
  record_words.reserve(128);
  if (derived_owner_word)
    record_words.push_back(*derived_owner_word);
  if ((derived_owner_vgpr &&
       !append_dynamic_barrier_store_u32_vgpr(
           record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id),
           *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_barrier_event_index_store(
          record_words, base + offsetof(ConSanMoiReportHeader, event_counter),
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, event_index), slot_vgpr,
          *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_workgroup_source(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_x),
          workgroup_sources.x, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_workgroup_source(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_y),
          workgroup_sources.y, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_workgroup_source(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_z),
          workgroup_sources.z, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_u32_scalar_src(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, lane_mask),
          *options.moi_exec_save_sgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_barrier_store_u32_scalar_src(
          record_words,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, lane_mask) + sizeof(uint32_t),
          static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u), slot_vgpr, *options.scratch_vgpr,
          arch) ||
      (options.moi_owner_vgpr &&
       !append_dynamic_barrier_store_u32_vgpr(
           record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id),
           *options.moi_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_barrier_store_u32_literal(
          record_words, barrier_record_base + offsetof(ConSanMoiBarrierRecord, instruction_offset),
          static_cast<uint32_t>(candidate.site.text_offset), slot_vgpr, *options.scratch_vgpr,
          arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode record stores");
    return std::nullopt;
  }

  if (record_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan MOI barrier record overflow branch is out of range");
    return std::nullopt;
  }
  const auto skip_record = build_s_cbranch_vccz(static_cast<int16_t>(record_words.size()), arch);
  const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!skip_record || !restore_exec) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode EXEC restore");
    return std::nullopt;
  }
  words.push_back(*skip_record);
  words.insert(words.end(), record_words.begin(), record_words.end());
  words.push_back(*restore_exec);
  if (!append_restore_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not restore VCC/SCC");
    return std::nullopt;
  }

  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  words.push_back(original_barrier_word);

  if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                            arch)) {
    errors.emplace_back("ConSan MOI barrier record could not encode its return");
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_shadow_barrier_epoch_cave_words(
    const ConSanBarrierSite &site, const ConSanOptions &options, rj_code_arch_t arch,
    uint32_t original_barrier_word, uint64_t cave_text_offset, uint64_t return_text_offset,
    std::vector<std::string> &errors) {
  if (!options.moi_epoch_vgpr) {
    errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch requires RJ_CONSAN_MOI_EPOCH_VGPR");
    return std::nullopt;
  }

  const auto increment_epoch = build_v_add_nc_u32_e32(
      *options.moi_epoch_vgpr, scalar_positive_inline_u32(1), *options.moi_epoch_vgpr, arch);
  const auto saturate_epoch = build_v_min_u32_e32_literal(
      *options.moi_epoch_vgpr, consan_moi_exact_shadow::max_epoch, *options.moi_epoch_vgpr, arch);
  if (!increment_epoch || !saturate_epoch) {
    errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch could not encode saturating epoch add");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(5);
  words.push_back(original_barrier_word);
  words.push_back(*increment_epoch);
  words.insert(words.end(), saturate_epoch->begin(), saturate_epoch->end());

  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset);
  if (ret) {
    words.push_back(build_s_branch(*ret, arch));
  } else {
    const auto jump_sgprs = moi_indirect_jump_sgprs(options);
    if (!jump_sgprs || !append_moi_scc_preserving_indirect_jump(
                           words, cave_text_offset, return_text_offset, jump_sgprs->pc_sgpr,
                           jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch could not encode long return at " +
          std::to_string(site.text_offset));
      return std::nullopt;
    }
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_inline_shadow_private_epoch_barrier_cave_words(
    const ConSanBarrierSite &site, const ConSanOptions &options, uint16_t scratch_vgpr,
    uint32_t epoch_offset, const VgprSpillSequence &spill, rj_code_arch_t arch,
    uint32_t original_barrier_word, uint64_t cave_text_offset, uint64_t return_text_offset,
    std::vector<std::string> &errors) {
  const auto load_epoch = build_address_free_scratch_load_b32(scratch_vgpr, epoch_offset, arch);
  const auto wait_load = build_s_wait_loadcnt0(arch);
  const auto increment_epoch =
      build_v_add_nc_u32_e32(scratch_vgpr, scalar_positive_inline_u32(1), scratch_vgpr, arch);
  const auto saturate_epoch = build_v_min_u32_e32_literal(
      scratch_vgpr, consan_moi_exact_shadow::max_epoch, scratch_vgpr, arch);
  const auto store_epoch = build_address_free_scratch_store_b32(scratch_vgpr, epoch_offset, arch);
  const auto wait_store = build_s_wait_storecnt0(arch);
  if (!load_epoch || !wait_load || !increment_epoch || !saturate_epoch || !store_epoch ||
      !wait_store) {
    errors.emplace_back("ConSan MOI inline-shadow barrier could not encode private epoch update");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(2u + spill.save_words.size() + load_epoch->size() + store_epoch->size() +
                saturate_epoch->size() + spill.restore_words.size() + 3u);
  words.push_back(original_barrier_word);
  words.insert(words.end(), spill.save_words.begin(), spill.save_words.end());
  words.insert(words.end(), load_epoch->begin(), load_epoch->end());
  words.push_back(*wait_load);
  words.push_back(*increment_epoch);
  words.insert(words.end(), saturate_epoch->begin(), saturate_epoch->end());
  words.insert(words.end(), store_epoch->begin(), store_epoch->end());
  words.push_back(*wait_store);
  words.insert(words.end(), spill.restore_words.begin(), spill.restore_words.end());

  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(branch_pc, return_text_offset);
  if (ret) {
    words.push_back(build_s_branch(*ret, arch));
  } else {
    const auto jump_sgprs = moi_indirect_jump_sgprs(options);
    if (!jump_sgprs || !append_moi_scc_preserving_indirect_jump(
                           words, cave_text_offset, return_text_offset, jump_sgprs->pc_sgpr,
                           jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow private epoch could not encode long return at " +
          std::to_string(site.text_offset));
      return std::nullopt;
    }
  }
  return words;
}

void try_apply_inline_shadow_barrier_epoch_patch(std::span<const uint8_t> bytes,
                                                 const ConSanOptions &options, rj_code_arch_t arch,
                                                 ConSanResult &result) {
  if (!options.moi_track_barriers)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch currently supports only RDNA4");
    return;
  }
  if (!options.moi_epoch_vgpr && !options.automatic_moi_private_epoch) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch has no persistent epoch state");
    return;
  }
  if (options.moi_epoch_vgpr && static_cast<uint32_t>(*options.moi_epoch_vgpr) + 1u > kMaxVgprs) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch needs one epoch VGPR");
    return;
  }

  std::vector<BarrierRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_barrier_epoch_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_barrier_epoch_candidates(function, candidates);
  apply_test_container_filter(candidates, options);
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline-shadow barrier epoch patch found no barriers");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no .text section");
    return;
  }
  std::vector<BarrierRecordCandidate> selected_candidates;
  selected_candidates.reserve(options.max_patches);
  for (const BarrierRecordCandidate &candidate : candidates) {
    if (selected_candidates.size() == options.max_patches)
      break;
    const ConSanBarrierSite &site = candidate.site;
    if (site.size != sizeof(uint32_t)) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch skipped non-32-bit barrier " +
          site.mnemonic + " in " + candidate.container_name);
      continue;
    }
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch site is outside .text");
      return;
    }
    selected_candidates.push_back(candidate);
  }

  if (selected_candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no patchable barriers");
    return;
  }

  struct PlannedInlineEpochBarrier {
    BarrierRecordCandidate candidate;
    std::vector<uint64_t> owner_descriptor_file_offsets;
    std::optional<uint16_t> scratch_vgpr;
    std::optional<uint32_t> private_epoch_offset;
    std::optional<VgprSpillSequence> spill;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedInlineEpochBarrier> planned_candidates;
  MoiDescriptorPrivateRequirements private_requirements;
  if (options.automatic_moi_private_epoch) {
    for (const BarrierRecordCandidate &candidate : selected_candidates) {
      if (!candidate.kernel_descriptor_file_offset)
        continue;
      const ConSanKernelInfo *kernel =
          kernel_for_descriptor(result, *candidate.kernel_descriptor_file_offset);
      if (kernel == nullptr)
        continue;
      const auto access_patch =
          std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
            return (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
                    patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore) &&
                   patch.scratch_vgpr && patch.persistent_epoch_private_offset &&
                   kernel_owns_patch(*kernel, patch);
          });
      if (access_patch == result.patches.end()) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier has no access-probe state for kernel " +
            kernel->name);
        continue;
      }
      const uint32_t ephemeral_base = access_patch->persistent_private_state_end.value_or(
          *access_patch->persistent_epoch_private_offset + SpillManager::kSlotBytes);
      SpillManager manager(ephemeral_base, kMaxAddressFreeScratchPrivateBytes);
      auto spill = build_vgpr_spill_sequence(manager, *access_patch->scratch_vgpr,
                                             /*vgpr_count=*/1, arch);
      if (!spill) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier could not preserve its temporary VGPR");
        continue;
      }
      const uint32_t required_private_bytes =
          std::max(access_patch->required_private_segment_size, spill->total_private_bytes);
      auto [it, inserted] = private_requirements.emplace(*candidate.kernel_descriptor_file_offset,
                                                         required_private_bytes);
      if (!inserted)
        it->second = std::max(it->second, required_private_bytes);
      planned_candidates.push_back(PlannedInlineEpochBarrier{
          .candidate = candidate,
          .owner_descriptor_file_offsets = {*candidate.kernel_descriptor_file_offset},
          .scratch_vgpr = access_patch->scratch_vgpr,
          .private_epoch_offset = access_patch->persistent_epoch_private_offset,
          .spill = std::move(spill),
          .required_private_bytes = required_private_bytes,
      });
    }
    if (!apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                             private_requirements, result.errors))
      return;
  } else if (options.automatic_moi_persistent_vgprs) {
    MoiDescriptorVgprRequirements requirements;
    for (const BarrierRecordCandidate &candidate : selected_candidates) {
      const ConSanCandidateResourcePlan *plan = resource_plan_for_site(
          result, ConSanResourceSiteKind::Barrier, candidate.site.text_offset);
      if (plan == nullptr || plan->owner_descriptor_file_offsets.empty()) {
        result.warnings.emplace_back(
            "ConSan MOI inline barrier has no owned automatic persistent-state scope");
        continue;
      }
      for (uint64_t descriptor_offset : plan->owner_descriptor_file_offsets) {
        requirements[descriptor_offset] = static_cast<uint16_t>(*options.moi_epoch_vgpr + 1u);
      }
    }
    if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result, requirements,
                                       result.errors)) {
      return;
    }
  } else if (!grow_moi_kernel_descriptor_vgprs(patcher, active_bytes, result,
                                               static_cast<uint32_t>(*options.moi_epoch_vgpr) +
                                                   1u)) {
    return;
  }
  if (!options.automatic_moi_private_epoch) {
    for (const BarrierRecordCandidate &candidate : selected_candidates) {
      std::vector<uint64_t> owners;
      if (const ConSanCandidateResourcePlan *plan = resource_plan_for_site(
              result, ConSanResourceSiteKind::Barrier, candidate.site.text_offset)) {
        owners = plan->owner_descriptor_file_offsets;
      }
      if (owners.empty() && candidate.kernel_descriptor_file_offset)
        owners.push_back(*candidate.kernel_descriptor_file_offset);
      if (automatic_moi_persistent_state(options) && owners.empty())
        continue;
      planned_candidates.push_back(PlannedInlineEpochBarrier{
          .candidate = candidate,
          .owner_descriptor_file_offsets = std::move(owners),
          .scratch_vgpr = std::nullopt,
          .private_epoch_offset = std::nullopt,
          .spill = std::nullopt,
          .required_private_bytes = 0,
      });
    }
  }
  if (planned_candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no compatible persistent state");
    return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator local_indirect_allocator(old_text, result, arch,
                                                      kMoiLocalIndirectIslandWords);
  std::vector<ConSanPatchInfo> patches;
  patches.reserve(planned_candidates.size() * 2u);
  constexpr uint64_t kIndirectIslandWords = 7u;
  const uint64_t indirect_island_bytes = kIndirectIslandWords * sizeof(uint32_t);
  std::optional<uint64_t> barrier_island_begin;
  if (options.moi_exec_save_sgpr) {
    uint64_t first_access_body = std::numeric_limits<uint64_t>::max();
    uint64_t access_island_begin = std::numeric_limits<uint64_t>::max();
    uint64_t access_island_end = 0u;
    for (const ConSanPatchInfo &patch : result.patches) {
      if (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
          patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore ||
          patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) {
        first_access_body = std::min(first_access_body, patch.trampoline_offset);
      } else if (patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland) {
        access_island_begin = std::min(access_island_begin, patch.trampoline_offset);
        access_island_end =
            std::max(access_island_end,
                     patch.trampoline_offset + static_cast<uint64_t>(patch.trampoline_size));
      }
    }
    const uint64_t required_island_bytes = planned_candidates.size() * indirect_island_bytes;
    const auto is_reserved_nop_range = [&](uint64_t candidate_begin) {
      bool all_reserved_nops = candidate_begin + required_island_bytes <= old_text.size();
      for (uint64_t offset = candidate_begin;
           all_reserved_nops && offset < candidate_begin + required_island_bytes;
           offset += sizeof(uint32_t)) {
        uint32_t word = 0;
        std::memcpy(&word, old_text.data() + offset, sizeof(word));
        all_reserved_nops = word == build_s_nop(0, arch);
      }
      return all_reserved_nops;
    };
    // Keep barrier islands first: early synchronization sites are more likely
    // to be near the start of a large generated object, while sorted access
    // islands retain the same monotonic anchor-to-island mapping after them.
    if (access_island_begin != std::numeric_limits<uint64_t>::max() &&
        access_island_begin >= required_island_bytes) {
      const uint64_t candidate_begin = access_island_begin - required_island_bytes;
      if (is_reserved_nop_range(candidate_begin))
        barrier_island_begin = candidate_begin;
    }
    // Accept the older trailing reservation layout as well so independently
    // composed exact-shadow producers remain compatible.
    if (!barrier_island_begin && first_access_body != std::numeric_limits<uint64_t>::max() &&
        first_access_body >= required_island_bytes &&
        first_access_body - required_island_bytes >= access_island_end) {
      const uint64_t candidate_begin = first_access_body - required_island_bytes;
      if (is_reserved_nop_range(candidate_begin))
        barrier_island_begin = candidate_begin;
    }
  }

  for (size_t planned_index = 0; planned_index < planned_candidates.size(); ++planned_index) {
    const PlannedInlineEpochBarrier &planned = planned_candidates[planned_index];
    const BarrierRecordCandidate &candidate = planned.candidate;
    const ConSanBarrierSite &site = candidate.site;
    uint32_t original_barrier_word = 0;
    std::memcpy(&original_barrier_word, new_text.data() + site.text_offset,
                sizeof(original_barrier_word));

    // The persistent-VGPR epoch update is small enough to live entirely in a
    // local NOP island. Prefer that layout over an indirect island followed by
    // an appended body: besides being smaller, it avoids perturbing SCC and
    // consuming temporary SGPR state on both the entry and return paths. The
    // private-epoch form is larger and continues to use the general appended
    // placement below.
    if (!planned.private_epoch_offset) {
      if (const auto local_body_offset =
              local_indirect_allocator.claim_reachable(site.text_offset)) {
        const uint64_t return_text_offset = site.text_offset + site.size;
        auto local_words = build_inline_shadow_barrier_epoch_cave_words(
            site, options, arch, original_barrier_word, *local_body_offset, return_text_offset,
            result.errors);
        if (!local_words)
          return;
        if (local_words->size() > kMoiLocalIndirectIslandWords ||
            *local_body_offset + local_words->size() * sizeof(uint32_t) > new_text.size()) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow barrier epoch local body does not fit its NOP island");
          return;
        }
        const auto fwd = compute_sopp_branch_simm16(site.text_offset, *local_body_offset);
        if (!fwd || !write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow barrier epoch could not enter its local body");
          return;
        }
        std::memcpy(new_text.data() + *local_body_offset, local_words->data(),
                    local_words->size() * sizeof(uint32_t));

        ConSanPatchInfo info;
        info.kind = ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
        info.anchor_offset = site.text_offset;
        info.trampoline_offset = *local_body_offset;
        info.original_size = site.size;
        info.trampoline_size = static_cast<uint32_t>(local_words->size() * sizeof(uint32_t));
        info.owner_descriptor_file_offsets = planned.owner_descriptor_file_offsets;
        patches.push_back(std::move(info));
        continue;
      }
    }

    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + site.size;
    std::optional<std::vector<uint32_t>> cave_words;
    if (planned.private_epoch_offset && planned.scratch_vgpr && planned.spill) {
      cave_words = build_inline_shadow_private_epoch_barrier_cave_words(
          site, options, *planned.scratch_vgpr, *planned.private_epoch_offset, *planned.spill, arch,
          original_barrier_word, cave_text_offset, return_text_offset, result.errors);
    } else {
      cave_words = build_inline_shadow_barrier_epoch_cave_words(
          site, options, arch, original_barrier_word, cave_text_offset, return_text_offset,
          result.errors);
    }
    if (!cave_words)
      return;

    std::optional<uint64_t> island_offset =
        barrier_island_begin
            ? std::optional<uint64_t>(*barrier_island_begin + planned_index * indirect_island_bytes)
            : std::nullopt;
    if (!island_offset &&
        !compute_sopp_branch_simm16(site.text_offset, cave_text_offset).has_value()) {
      island_offset = local_indirect_allocator.claim_reachable(site.text_offset);
      if (!island_offset) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow barrier epoch found no reachable indirect entry island in " +
            candidate.container_name);
        continue;
      }
    }
    std::optional<DbiPatchPlacement> placement;
    if (island_offset) {
      std::string placement_error;
      placement = placement_planner.plan_indirect_appended(
          site.text_offset, site.size, cave_words->size() * sizeof(uint32_t), &placement_error);
      if (!placement)
        result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch placement failed: " +
                                   placement_error);
    } else {
      placement = plan_prebuilt_appended_cave(placement_planner, site.text_offset, site.size,
                                              *cave_words, result.errors,
                                              "ConSan MOI inline-shadow barrier epoch patch");
    }
    if (!placement || placement->body_offset != cave_text_offset ||
        new_text.size() != placement->body_offset) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch has a stale cave mapping");
      return;
    }

    const uint64_t entry_target = island_offset.value_or(placement->body_offset);
    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, entry_target);
    if (!fwd) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch forward branch is out of range");
      return;
    }
    if (!write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch could not rewrite barrier slot");
      return;
    }
    if (island_offset) {
      const auto jump_sgprs = moi_indirect_jump_sgprs(options);
      std::vector<uint32_t> island_words;
      if (!jump_sgprs ||
          !append_moi_scc_preserving_indirect_jump(
              island_words, *island_offset, placement->body_offset, jump_sgprs->pc_sgpr,
              jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch) ||
          island_words.size() != kIndirectIslandWords ||
          *island_offset + indirect_island_bytes > new_text.size()) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow barrier epoch could not encode its entry island");
        return;
      }
      std::memcpy(new_text.data() + *island_offset, island_words.data(), indirect_island_bytes);
      ConSanPatchInfo island_info;
      island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      island_info.anchor_offset = site.text_offset;
      island_info.trampoline_offset = *island_offset;
      island_info.trampoline_size = static_cast<uint32_t>(indirect_island_bytes);
      island_info.owner_descriptor_file_offsets = planned.owner_descriptor_file_offsets;
      patches.push_back(std::move(island_info));
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.scratch_vgpr = planned.scratch_vgpr;
    info.persistent_epoch_private_offset = planned.private_epoch_offset;
    info.required_private_segment_size = planned.required_private_bytes;
    info.owner_descriptor_file_offsets = planned.owner_descriptor_file_offsets;
    if (planned.spill)
      info.spilled_vgpr_count = planned.spill->vgpr_count;
    patches.push_back(info);
  }

  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch patch could not grow .text");
    return;
  }

  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void try_apply_barrier_epoch_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   rj_code_arch_t arch, ConSanResult &result) {
  if (!options.moi_track_barriers)
    return;
  // Sampled consumes only associated complete barrier sequences and publishes
  // into its selected causal windows in a later, engine-specific pass. Never
  // let it inherit Record/Replay's intentionally per-instruction recorder.
  if (options.moi_engine == ConSanMoiEngine::Sampled)
    return;
  // Record/Replay access records already carry the persistent per-wave epoch.
  // When that state is initialized, advance it at the barrier instead of
  // emitting an event for every dynamic barrier execution. This keeps the
  // trace bounded by access-record capacity while preserving barrier epochs.
  const bool has_record_replay_access_patch =
      options.moi_engine == ConSanMoiEngine::RecordReplay &&
      std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
               patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
      });
  if (options.moi_engine == ConSanMoiEngine::InlineShadow ||
      (has_record_replay_access_patch && options.moi_init_owner_epoch && options.moi_epoch_vgpr &&
       !options.automatic_moi_private_epoch)) {
    try_apply_inline_shadow_barrier_epoch_patch(bytes, options, arch, result);
    return;
  }
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_GFX1250) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch currently supports only the RDNA4 family");
    return;
  }
  if (!options.moi_report_buffer_address) {
    result.warnings.emplace_back("ConSan MOI barrier record patch requires a MOI report buffer");
    return;
  }
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1)) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires room for access and barrier records");
    return;
  }
  const ConSanMoiReportBufferLayout layout = resolve_record_replay_report_layout(
      options, /*include_barriers=*/true, options.moi_track_atomics, options.moi_track_atomics);
  const uint32_t max_candidates = std::min(options.max_patches, layout.barrier_record_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back("ConSan MOI barrier record patch has no barrier record capacity");
    return;
  }
  if (!options.moi_exec_save_sgpr) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return;
  }
  if (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u) {
    result.warnings.emplace_back(
        "ConSan MOI barrier record patch requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..100");
    return;
  }

  std::vector<BarrierRecordCandidate> candidates;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_barrier_epoch_candidates(kernel, candidates);
  for (const ConSanFunctionInfo &function : result.functions)
    append_barrier_epoch_candidates(function, candidates);
  apply_test_container_filter(candidates, options);
  std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.site.text_offset < rhs.site.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no barriers");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI barrier record patch found no .text section");
    return;
  }
  struct PlannedBarrierRecord {
    BarrierRecordCandidate candidate;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
  };
  std::vector<PlannedBarrierRecord> selected_candidates;
  selected_candidates.reserve(max_candidates);
  MoiResourcePlanningState resource_state(bytes, arch, options, result);
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const BarrierRecordCandidate &candidate = candidates[candidate_index];
    if (selected_candidates.size() == max_candidates)
      break;
    const ConSanBarrierSite &site = candidate.site;
    if (site.size != sizeof(uint32_t)) {
      result.warnings.emplace_back("ConSan MOI barrier record patch skipped non-32-bit barrier " +
                                   site.mnemonic + " in " + candidate.container_name);
      continue;
    }
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI barrier record patch site is outside .text");
      return;
    }
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Barrier, site.text_offset);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Barrier,
                                     candidate_index, site.text_offset,
                                     candidate.kernel_descriptor_file_offset,
                                     /*scratch_count=*/6u);
    if (!existing_plan)
      result.resource_plans.push_back(plan);
    const auto resources = resolve_moi_scratch_plan(plan, options, /*expected_count=*/6u);
    if (!resources) {
      result.warnings.emplace_back(
          "ConSan MOI barrier record patch has no usable resource plan in " +
          candidate.container_name);
      continue;
    }
    BarrierRecordCandidate planned_candidate = candidate;
    if (!options.moi_owner_vgpr && !planned_candidate.kernel_descriptor_file_offset) {
      planned_candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      if (!planned_candidate.kernel_descriptor_file_offset)
        continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(result, *resources, options, spill_managers, arch,
                                       result.warnings);
      if (!spill)
        continue;
      note_spill_descriptor_requirements(private_requirements, *resources, *spill);
    }
    note_descriptor_requirements(descriptor_requirements, *resources);
    note_moi_sgpr_requirements(scalar_requirements, *resources, options);
    selected_candidates.push_back({std::move(planned_candidate), *resources, std::move(spill)});
  }

  if (selected_candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no patchable barriers");
    return;
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     descriptor_requirements, result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors) ||
      !apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements,
                                          result.errors)) {
    return;
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result, arch,
                                                kMoiLocalIndirectIslandWords);
  std::vector<ConSanPatchInfo> patches;
  for (const PlannedBarrierRecord &planned : selected_candidates) {
    const BarrierRecordCandidate &candidate = planned.candidate;
    const ConSanBarrierSite &site = candidate.site;
    uint32_t original_barrier_word = 0;
    std::memcpy(&original_barrier_word, new_text.data() + site.text_offset,
                sizeof(original_barrier_word));

    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + site.size;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    auto cave_words = build_barrier_record_cave_words(
        active_bytes, candidate, candidate_options, planned.spill ? &*planned.spill : nullptr, arch,
        layout.barrier_record_capacity, layout.barrier_records_offset, original_barrier_word,
        cave_text_offset, return_text_offset, result.errors);
    if (!cave_words)
      return;

    const bool direct_entry =
        compute_sopp_branch_simm16(site.text_offset, cave_text_offset).has_value();
    const std::optional<uint64_t> indirect_entry_island =
        direct_entry ? std::nullopt : indirect_allocator.claim_reachable(site.text_offset);
    if (!direct_entry && !indirect_entry_island) {
      result.warnings.emplace_back(
          "ConSan MOI barrier record patch found no reachable indirect entry island in " +
          candidate.container_name);
      continue;
    }
    std::string placement_error;
    const auto placement = placement_planner.plan_indirect_appended(
        site.text_offset, site.size, cave_words->size() * sizeof(uint32_t), &placement_error);
    if (!placement) {
      result.errors.emplace_back("ConSan MOI barrier record placement failed: " + placement_error);
      return;
    }
    if (placement->body_offset != cave_text_offset || new_text.size() != placement->body_offset) {
      result.errors.emplace_back("ConSan MOI barrier record patch has a stale cave mapping");
      return;
    }

    const uint64_t entry_target = indirect_entry_island.value_or(placement->body_offset);
    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, entry_target);
    if (!fwd) {
      result.errors.emplace_back("ConSan MOI barrier record forward branch is out of range");
      return;
    }
    if (!write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
      result.errors.emplace_back("ConSan MOI barrier record patch could not rewrite barrier slot");
      return;
    }
    if (indirect_entry_island &&
        !emit_moi_local_indirect_entry_island(
            new_text, *indirect_entry_island, placement->body_offset, site.text_offset, options,
            planned.resources.owner_descriptor_file_offsets, arch, patches, result.errors,
            "ConSan MOI barrier record patch")) {
      return;
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiBarrierRecord;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.scratch_vgpr = planned.resources.base;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      info.required_private_segment_size = planned.spill->total_private_bytes;
    }
    patches.push_back(info);
  }

  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI barrier record patch found no patchable barriers");
    return;
  }
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI barrier record patch could not grow .text");
    return;
  }

  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}
