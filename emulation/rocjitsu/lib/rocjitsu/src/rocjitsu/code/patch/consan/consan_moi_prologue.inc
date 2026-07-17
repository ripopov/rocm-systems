void append_moi_entry_salu_write(std::vector<uint32_t> &words, uint32_t word, rj_code_arch_t arch) {
  words.push_back(word);
  words.push_back(build_s_delay_alu(kDelayAluSaluDep1, arch));
}

[[nodiscard]] bool
append_dispatch_id_capture_and_restore(std::vector<uint32_t> &words,
                                       const ConSanMoiDispatchIdPreloadPlan &plan,
                                       std::optional<uint16_t> persistent_dispatch_sgpr,
                                       rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (!plan.supported()) {
    errors.emplace_back("ConSan MOI dispatch-ID prologue received an unsupported preload plan");
    return false;
  }
  if (!persistent_dispatch_sgpr || *persistent_dispatch_sgpr < 20u ||
      *persistent_dispatch_sgpr % 2u != 0u ||
      static_cast<uint32_t>(*persistent_dispatch_sgpr) + 2u > kMaxSgprs ||
      static_cast<uint32_t>(plan.dispatch_id_sgpr) + 2u > kMaxSgprs) {
    errors.emplace_back("ConSan MOI dispatch-ID prologue has an invalid persistent or source pair");
    return false;
  }

  append_moi_entry_salu_write(
      words, build_s_mov_b32(*persistent_dispatch_sgpr, plan.dispatch_id_sgpr, arch), arch);
  append_moi_entry_salu_write(words,
                              build_s_mov_b32(static_cast<uint16_t>(*persistent_dispatch_sgpr + 1u),
                                              static_cast<uint16_t>(plan.dispatch_id_sgpr + 1u),
                                              arch),
                              arch);
  for (uint16_t destination = plan.first_shifted_guest_sgpr;
       destination < plan.first_shifted_guest_sgpr + plan.shifted_guest_sgpr_count; ++destination) {
    const auto source = consan_moi_dispatch_id_restore_source(plan, destination);
    if (!source || *source >= kMaxSgprs) {
      errors.emplace_back("ConSan MOI dispatch-ID prologue has an invalid guest restore source");
      return false;
    }
    append_moi_entry_salu_write(words, build_s_mov_b32(destination, *source, arch), arch);
  }
  return true;
}

[[nodiscard]] bool append_moi_prologue_return(std::vector<uint32_t> &words,
                                              uint64_t prologue_text_offset,
                                              uint64_t original_entry_text_offset,
                                              std::optional<uint16_t> pc_sgpr, rj_code_arch_t arch,
                                              std::string_view prologue_name,
                                              std::vector<std::string> &errors) {
  const uint64_t branch_pc =
      prologue_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  if (const auto branch = compute_sopp_branch_simm16(branch_pc, original_entry_text_offset)) {
    words.push_back(build_s_branch(*branch, arch));
    return true;
  }

  // Large all-site transformations can place the entry prologue beyond a
  // SOPP branch's signed 16-bit dword range. The descriptor-backed ConSan
  // scalar state is unobservable by the original kernel at entry, so its first
  // pair is safe temporary state for an absolute indirect return.
  if (!pc_sgpr || *pc_sgpr > 126u || *pc_sgpr % 2u != 0u) {
    errors.emplace_back("ConSan MOI " + std::string(prologue_name) +
                        " branch target is out of range and has no valid long-return SGPR pair");
    return false;
  }
  words.push_back(pack_sop1(/*s_getpc_b64=*/0x47, *pc_sgpr, 0));
  const uint64_t pc_after_getpc = branch_pc + sizeof(uint32_t);
  if (original_entry_text_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      pc_after_getpc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    errors.emplace_back("ConSan MOI " + std::string(prologue_name) +
                        " long-return address exceeds the signed builder range");
    return false;
  }
  const int64_t delta =
      static_cast<int64_t>(original_entry_text_offset) - static_cast<int64_t>(pc_after_getpc);
  if (!append_pc_delta_builder(words, arch, *pc_sgpr, delta)) {
    errors.emplace_back("ConSan MOI " + std::string(prologue_name) +
                        " could not encode its long return");
    return false;
  }
  const auto wait_pc = build_s_wait_alu_sa_sdst0(arch);
  if (!wait_pc) {
    errors.emplace_back("ConSan MOI " + std::string(prologue_name) +
                        " could not encode its long-return scalar dependency wait");
    return false;
  }
  words.push_back(*wait_pc);
  words.push_back(pack_sop1(/*s_setpc_b64=*/0x48, 0, *pc_sgpr));
  return true;
}

[[nodiscard]] bool append_moi_scc_preserving_indirect_jump(
    std::vector<uint32_t> &words, uint64_t words_text_offset, uint64_t target_text_offset,
    uint16_t pc_sgpr, uint16_t saved_scc_sgpr, bool capture_scc, rj_code_arch_t arch) {
  if (capture_scc) {
    const auto save_scc = build_rdna4_s_cselect_b32(saved_scc_sgpr, scalar_positive_inline_u32(1),
                                                    scalar_positive_inline_u32(0), arch);
    if (!save_scc)
      return false;
    words.push_back(*save_scc);
  }
  const uint64_t getpc_text_offset =
      words_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  words.push_back(pack_sop1(/*s_getpc_b64=*/0x47, pc_sgpr, 0));
  const uint64_t pc_after_getpc = getpc_text_offset + sizeof(uint32_t);
  if (target_text_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      pc_after_getpc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  const int64_t delta =
      static_cast<int64_t>(target_text_offset) - static_cast<int64_t>(pc_after_getpc);
  if (!append_pc_delta_builder(words, arch, pc_sgpr, delta))
    return false;
  const auto restore_scc =
      build_rdna4_s_cmp_lg_u32(saved_scc_sgpr, scalar_positive_inline_u32(0), arch);
  if (!restore_scc)
    return false;
  words.push_back(*restore_scc);
  words.push_back(pack_sop1(/*s_setpc_b64=*/0x48, 0, pc_sgpr));
  return true;
}

[[nodiscard]] bool append_moi_direct_or_indirect_return(std::vector<uint32_t> &words,
                                                        uint64_t cave_text_offset,
                                                        uint64_t return_text_offset,
                                                        const ConSanOptions &options,
                                                        rj_code_arch_t arch) {
  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  if (const auto direct = compute_sopp_branch_simm16(branch_pc, return_text_offset)) {
    words.push_back(build_s_branch(*direct, arch));
    return true;
  }
  const auto jump_sgprs = moi_indirect_jump_sgprs(options);
  if (!jump_sgprs || !append_moi_scc_preserving_indirect_jump(
                         words, cave_text_offset, return_text_offset, jump_sgprs->pc_sgpr,
                         jump_sgprs->scc_save_sgpr, /*capture_scc=*/false, arch)) {
    return false;
  }
  const auto wait_pc = build_s_wait_alu_sa_sdst0(arch);
  if (!wait_pc)
    return false;
  words.insert(words.end() - 1, *wait_pc);
  return true;
}

[[nodiscard]] bool
emit_moi_local_indirect_entry_island(std::vector<uint8_t> &text, uint64_t island_text_offset,
                                     uint64_t cave_text_offset, uint64_t anchor_text_offset,
                                     const ConSanOptions &options,
                                     std::span<const uint64_t> owner_descriptor_file_offsets,
                                     rj_code_arch_t arch, std::vector<ConSanPatchInfo> &patches,
                                     std::vector<std::string> &errors, std::string_view context) {
  const auto jump_sgprs = moi_indirect_jump_sgprs(options);
  std::vector<uint32_t> words;
  if (!jump_sgprs || !append_moi_scc_preserving_indirect_jump(
                         words, island_text_offset, cave_text_offset, jump_sgprs->pc_sgpr,
                         jump_sgprs->scc_save_sgpr, /*capture_scc=*/true, arch)) {
    errors.emplace_back(std::string(context) + " could not encode its indirect entry island");
    return false;
  }
  words.insert(words.end() - 1, build_s_delay_alu(kDelayAluSaluDep1, arch));
  if (words.size() != kMoiLocalIndirectIslandWords || island_text_offset > text.size() ||
      words.size() * sizeof(uint32_t) > text.size() - island_text_offset) {
    errors.emplace_back(std::string(context) + " has an invalid indirect entry island");
    return false;
  }
  std::memcpy(text.data() + island_text_offset, words.data(), words.size() * sizeof(uint32_t));

  ConSanPatchInfo island_info;
  island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
  island_info.anchor_offset = anchor_text_offset;
  island_info.trampoline_offset = island_text_offset;
  island_info.trampoline_size = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  island_info.owner_descriptor_file_offsets.assign(owner_descriptor_file_offsets.begin(),
                                                   owner_descriptor_file_offsets.end());
  patches.push_back(std::move(island_info));
  return true;
}

[[nodiscard]] bool emit_moi_local_indirect_kernel_entry_island(
    std::vector<uint8_t> &text, uint64_t island_text_offset, uint64_t cave_text_offset,
    uint64_t anchor_text_offset, const ConSanOptions &options,
    std::span<const uint64_t> owner_descriptor_file_offsets, rj_code_arch_t arch,
    std::vector<ConSanPatchInfo> &patches, std::vector<std::string> &errors,
    std::string_view context) {
  const auto jump_sgprs = moi_indirect_jump_sgprs(options);
  std::vector<uint32_t> words;
  if (!jump_sgprs ||
      !append_moi_scc_preserving_indirect_jump(words, island_text_offset, cave_text_offset,
                                               jump_sgprs->pc_sgpr, jump_sgprs->scc_save_sgpr,
                                               /*capture_scc=*/true, arch) ||
      words.size() != 7u) {
    errors.emplace_back(std::string(context) + " could not encode its indirect entry island");
    return false;
  }
  // SCC has no incoming value at a hardware kernel entry. Reuse its restore
  // slot for the gfx12 scalar dependency delay and fit the relay in seven
  // words, while ordinary mid-kernel relays retain SCC preservation.
  words[words.size() - 2u] = build_s_delay_alu(kDelayAluSaluDep1, arch);
  if (island_text_offset > text.size() ||
      words.size() * sizeof(uint32_t) > text.size() - island_text_offset) {
    errors.emplace_back(std::string(context) + " has an invalid indirect entry island");
    return false;
  }
  std::memcpy(text.data() + island_text_offset, words.data(), words.size() * sizeof(uint32_t));

  ConSanPatchInfo island_info;
  island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
  island_info.anchor_offset = anchor_text_offset;
  island_info.trampoline_offset = island_text_offset;
  island_info.trampoline_size = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  island_info.owner_descriptor_file_offsets.assign(owner_descriptor_file_offsets.begin(),
                                                   owner_descriptor_file_offsets.end());
  patches.push_back(std::move(island_info));
  return true;
}

[[nodiscard]] bool append_moi_workgroup_shadow_initialization(
    std::vector<uint32_t> &words, const ConSanMoiWorkgroupShadowLayout &layout,
    uint16_t address_vgpr, uint16_t zero_vgpr, std::optional<uint16_t> state_sgpr,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (!state_sgpr || *state_sgpr > 120u || layout.size == 0u || layout.size % 8u != 0u) {
    errors.emplace_back(
        "ConSan MOI workgroup-shadow initializer requires aligned LDS and scalar state");
    return false;
  }
  const uint16_t exec_save = *state_sgpr;
  const uint16_t vcc_save = static_cast<uint16_t>(*state_sgpr + 2u);
  const uint16_t counter = static_cast<uint16_t>(*state_sgpr + 4u);
  const uint16_t saved_scc = static_cast<uint16_t>(*state_sgpr + 5u);
  const uint32_t slot_count = layout.size / 8u;

  const auto save_scc = build_rdna4_s_cselect_b32(saved_scc, scalar_positive_inline_u32(1),
                                                  scalar_positive_inline_u32(0), arch);
  const auto save_vcc = build_s_mov_b64(vcc_save, kRdna4VccLo, arch);
  const auto lane_zero =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), kRdna4WorkitemIdX, arch);
  const auto narrow_exec = build_s_and_saveexec_b64(exec_save, kRdna4VccLo, arch);
  const auto skip_empty_wave = build_s_cbranch_execz(0, arch);
  const auto address = build_v_mov_b32_e64_literal(address_vgpr, layout.base, arch);
  const auto store_low = build_ds_store_b32(address_vgpr, zero_vgpr, 0u, arch);
  const auto store_high = build_ds_store_b32(address_vgpr, zero_vgpr, 4u, arch);
  const auto advance =
      build_v_add_nc_u32_e32(address_vgpr, scalar_positive_inline_u32(8), address_vgpr, arch);
  const auto decrement = build_s_sub_u32(counter, counter, scalar_positive_inline_u32(1), arch);
  const auto more = build_rdna4_s_cmp_lg_u32(counter, scalar_positive_inline_u32(0), arch);
  const auto loop = build_s_cbranch_scc1(0, arch);
  const auto wait_ds = build_s_wait_loadcnt_dscnt0(arch);
  const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, exec_save, arch);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, vcc_save, arch);
  const auto barrier_signal = build_s_barrier_signal_all(arch);
  const auto barrier_wait = build_s_barrier_wait_all(arch);
  const auto restore_scc = build_rdna4_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch);
  if (!save_scc || !save_vcc || !lane_zero || !narrow_exec || !skip_empty_wave || !address ||
      !store_low || !store_high || !advance || !decrement || !more || !loop || !wait_ds ||
      !restore_exec || !restore_vcc || !barrier_signal || !barrier_wait || !restore_scc) {
    errors.emplace_back("ConSan MOI workgroup-shadow initializer could not encode RDNA4 state");
    return false;
  }

  words.push_back(*save_scc);
  words.push_back(*save_vcc);
  words.push_back(*lane_zero);
  words.push_back(*narrow_exec);
  const size_t skip_empty_wave_index = words.size();
  words.push_back(*skip_empty_wave);
  words.insert(words.end(), address->begin(), address->end());
  words.push_back(build_v_mov_b32_e32(zero_vgpr, scalar_positive_inline_u32(0), arch));
  words.push_back(build_s_mov_b32(counter, kVopLiteralSource, arch));
  words.push_back(slot_count);
  const size_t loop_begin = words.size();
  words.insert(words.end(), store_low->begin(), store_low->end());
  words.insert(words.end(), store_high->begin(), store_high->end());
  words.push_back(*advance);
  words.push_back(*decrement);
  words.push_back(*more);
  const size_t loop_branch_index = words.size();
  words.push_back(*loop);
  words.push_back(*wait_ds);
  const size_t restore_state_index = words.size();
  words.push_back(*restore_exec);
  words.push_back(*restore_vcc);
  words.push_back(*barrier_signal);
  words.push_back(*barrier_wait);
  words.push_back(*restore_scc);

  const int64_t skip_delta =
      static_cast<int64_t>(restore_state_index) - static_cast<int64_t>(skip_empty_wave_index + 1u);
  const int64_t loop_delta =
      static_cast<int64_t>(loop_begin) - static_cast<int64_t>(loop_branch_index + 1u);
  if (skip_delta < std::numeric_limits<int16_t>::min() ||
      skip_delta > std::numeric_limits<int16_t>::max() ||
      loop_delta < std::numeric_limits<int16_t>::min() ||
      loop_delta > std::numeric_limits<int16_t>::max()) {
    errors.emplace_back("ConSan MOI workgroup-shadow initializer branch exceeds SOPP range");
    return false;
  }
  words[skip_empty_wave_index] = *build_s_cbranch_execz(static_cast<int16_t>(skip_delta), arch);
  words[loop_branch_index] = *build_s_cbranch_scc1(static_cast<int16_t>(loop_delta), arch);
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_owner_epoch_prologue_words(
    uint64_t prologue_text_offset, uint64_t original_entry_text_offset, uint16_t owner_vgpr,
    uint16_t epoch_vgpr, std::optional<uint16_t> workgroup_key_vgpr,
    std::optional<uint16_t> sample_sequence_vgpr, uint16_t owner_shift_bits,
    ConSanMoiOwnerSource owner_source, std::optional<uint16_t> owner_sgpr, bool one_based_owner_ids,
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan,
    std::optional<uint16_t> persistent_dispatch_sgpr,
    std::optional<uint16_t> runtime_sample_counter_sgpr, std::optional<uint16_t> return_pc_sgpr,
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow,
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources,
    std::span<const uint32_t> displaced_entry_words, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  std::vector<uint32_t> words;
  words.reserve(48);
  if (dispatch_plan && !append_dispatch_id_capture_and_restore(
                           words, *dispatch_plan, persistent_dispatch_sgpr, arch, errors)) {
    return std::nullopt;
  }
  if (runtime_sample_counter_sgpr) {
    append_moi_entry_salu_write(
        words, build_s_mov_b32(*runtime_sample_counter_sgpr, scalar_positive_inline_u32(0), arch),
        arch);
  }
  if (workgroup_shadow &&
      !append_moi_workgroup_shadow_initialization(words, *workgroup_shadow, owner_vgpr, epoch_vgpr,
                                                  return_pc_sgpr, arch, errors)) {
    return std::nullopt;
  }
  if (workgroup_key_vgpr) {
    if (!workgroup_sources || !return_pc_sgpr) {
      errors.emplace_back(
          "ConSan MOI workgroup-key prologue requires launch sources and EXEC-save state");
      return std::nullopt;
    }
    words.push_back(build_v_mov_b32_e32(*workgroup_key_vgpr, scalar_positive_inline_u32(0), arch));
    ConSanOptions key_options;
    key_options.moi_exec_save_sgpr = return_pc_sgpr;
    if (!append_inline_workgroup_key(words, *workgroup_sources, key_options, *workgroup_key_vgpr,
                                     owner_vgpr, epoch_vgpr,
                                     /*original_exec_save_offset=*/20u, arch)) {
      errors.emplace_back("ConSan MOI workgroup-key prologue could not encode launch identity");
      return std::nullopt;
    }
    const auto restore_exec =
        build_s_mov_b64(kRdna4ExecLo, static_cast<uint16_t>(*return_pc_sgpr + 20u), arch);
    if (!restore_exec)
      return std::nullopt;
    words.push_back(*restore_exec);
  }
  switch (owner_source) {
  case ConSanMoiOwnerSource::WorkitemId: {
    auto owner_init = build_v_lshrrev_b32_e32(
        owner_vgpr, scalar_positive_inline_u32(owner_shift_bits), kRdna4WorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI owner/epoch prologue could not encode owner VGPR init");
      return std::nullopt;
    }
    words.push_back(*owner_init);
    if (one_based_owner_ids) {
      const auto make_nonzero =
          build_v_add_nc_u32_e32(owner_vgpr, scalar_positive_inline_u32(1), owner_vgpr, arch);
      if (!make_nonzero) {
        errors.emplace_back("ConSan MOI owner/epoch prologue could not bias owner IDs");
        return std::nullopt;
      }
      // Inline metadata reserves zero for absent ownership. Keep its persistent
      // owner IDs one-based; Record/Replay and Sampled retain zero-based IDs
      // because they use owner zero to elect the canonical wave.
      words.push_back(*make_nonzero);
    }
    break;
  }
  case ConSanMoiOwnerSource::HwId: {
    if (!owner_sgpr) {
      errors.emplace_back("ConSan MOI hw_id owner source requires RJ_CONSAN_MOI_OWNER_SGPR");
      return std::nullopt;
    }
    const auto hwreg = build_hwreg_imm(kGfx12HwRegHwId1, /*offset=*/0, kGfx12HwIdOwnerBits);
    if (!hwreg) {
      errors.emplace_back("ConSan MOI owner/epoch prologue could not encode HW_ID1 hwreg");
      return std::nullopt;
    }
    const auto get_hw_id = build_s_getreg_b32(*owner_sgpr, *hwreg, arch);
    if (!get_hw_id) {
      errors.emplace_back("ConSan MOI owner/epoch prologue could not encode s_getreg_b32");
      return std::nullopt;
    }
    append_moi_entry_salu_write(words, *get_hw_id, arch);
    words.push_back(build_v_mov_b32_e32(owner_vgpr, *owner_sgpr, arch));
    break;
  }
  }
  words.push_back(build_v_mov_b32_e32(epoch_vgpr, scalar_positive_inline_u32(0), arch));
  if (sample_sequence_vgpr)
    words.push_back(
        build_v_mov_b32_e32(*sample_sequence_vgpr, scalar_positive_inline_u32(0), arch));
  words.insert(words.end(), displaced_entry_words.begin(), displaced_entry_words.end());

  if (!append_moi_prologue_return(words, prologue_text_offset, original_entry_text_offset,
                                  return_pc_sgpr, arch, "owner/epoch prologue", errors)) {
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_private_epoch_prologue_words(
    uint64_t prologue_text_offset, uint64_t original_entry_text_offset, uint16_t scratch_vgpr,
    uint32_t epoch_offset, std::optional<uint32_t> owner_offset,
    std::optional<uint32_t> sample_sequence_offset, const VgprSpillSequence &spill,
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow,
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan,
    std::optional<uint16_t> persistent_dispatch_sgpr,
    std::optional<uint16_t> runtime_sample_counter_sgpr, std::optional<uint16_t> return_pc_sgpr,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  const auto epoch_store = build_address_free_scratch_store_b32(scratch_vgpr, epoch_offset, arch);
  std::optional<std::array<uint32_t, 3>> owner_store;
  if (owner_offset)
    owner_store = build_address_free_scratch_store_b32(scratch_vgpr, *owner_offset, arch);
  std::optional<std::array<uint32_t, 3>> sequence_store;
  if (sample_sequence_offset) {
    sequence_store =
        build_address_free_scratch_store_b32(scratch_vgpr, *sample_sequence_offset, arch);
  }
  const auto wait_store = build_s_wait_storecnt0(arch);
  if (!epoch_store || (owner_offset && !owner_store) ||
      (sample_sequence_offset && !sequence_store) || !wait_store) {
    errors.emplace_back(
        "ConSan MOI private-epoch prologue could not encode persistent-state initialization");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(spill.save_words.size() + spill.restore_words.size() + 48u);
  if (dispatch_plan && !append_dispatch_id_capture_and_restore(
                           words, *dispatch_plan, persistent_dispatch_sgpr, arch, errors)) {
    return std::nullopt;
  }
  if (runtime_sample_counter_sgpr) {
    append_moi_entry_salu_write(
        words, build_s_mov_b32(*runtime_sample_counter_sgpr, scalar_positive_inline_u32(0), arch),
        arch);
  }
  words.insert(words.end(), spill.save_words.begin(), spill.save_words.end());
  if (workgroup_shadow &&
      !append_moi_workgroup_shadow_initialization(words, *workgroup_shadow, scratch_vgpr,
                                                  static_cast<uint16_t>(scratch_vgpr + 1u),
                                                  return_pc_sgpr, arch, errors)) {
    return std::nullopt;
  }
  if (owner_store) {
    words.push_back(build_v_mov_b32_e32(scratch_vgpr, kRdna4WorkitemIdX, arch));
    words.insert(words.end(), owner_store->begin(), owner_store->end());
  }
  words.push_back(build_v_mov_b32_e32(scratch_vgpr, scalar_positive_inline_u32(0), arch));
  words.insert(words.end(), epoch_store->begin(), epoch_store->end());
  if (sequence_store)
    words.insert(words.end(), sequence_store->begin(), sequence_store->end());
  words.push_back(*wait_store);
  words.insert(words.end(), spill.restore_words.begin(), spill.restore_words.end());

  if (!append_moi_prologue_return(words, prologue_text_offset, original_entry_text_offset,
                                  return_pc_sgpr, arch, "private-epoch prologue", errors)) {
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] bool kernel_contains_patch_anchor(const ConSanKernelInfo &kernel,
                                                const ConSanPatchInfo &patch) {
  return kernel.has_text_range && patch.anchor_offset >= kernel.entry_text_offset &&
         patch.anchor_offset - kernel.entry_text_offset < kernel.code_size;
}

[[nodiscard]] bool kernel_owns_patch(const ConSanKernelInfo &kernel, const ConSanPatchInfo &patch) {
  if (!patch.owner_descriptor_file_offsets.empty()) {
    return std::ranges::find(patch.owner_descriptor_file_offsets, kernel.descriptor_file_offset) !=
           patch.owner_descriptor_file_offsets.end();
  }
  return kernel_contains_patch_anchor(kernel, patch);
}

[[nodiscard]] bool grow_moi_kernel_descriptor_registers(
    CodeObjectPatcher &patcher, std::span<const uint8_t> image, const ConSanKernelInfo &kernel,
    uint32_t required_vgpr_count, uint32_t required_sgpr_count, std::vector<std::string> &errors) {
  if (kernel.descriptor_file_offset > image.size() ||
      sizeof(KD) > image.size() - kernel.descriptor_file_offset) {
    errors.emplace_back("ConSan MOI descriptor register growth exceeds ELF bytes");
    return false;
  }

  KD desc{};
  std::memcpy(&desc, image.data() + kernel.descriptor_file_offset, sizeof(desc));
  if (!grow_moi_descriptor_vgpr_allocation(desc, required_vgpr_count)) {
    errors.emplace_back("ConSan MOI could not grow descriptor VGPR allocation");
    return false;
  }
  if (required_sgpr_count != 0 && !grow_moi_descriptor_sgpr_allocation(desc, required_sgpr_count)) {
    errors.emplace_back("ConSan MOI could not grow descriptor SGPR allocation");
    return false;
  }
  if (!patcher.patch_kernel_descriptor(kernel.descriptor_file_offset,
                                       {reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)})) {
    errors.emplace_back("ConSan MOI could not patch descriptor register allocation");
    return false;
  }
  return true;
}

[[nodiscard]] bool apply_moi_dispatch_id_descriptor(CodeObjectPatcher &patcher,
                                                    uint64_t descriptor_file_offset,
                                                    const ConSanMoiDispatchIdPreloadPlan &plan,
                                                    uint16_t persistent_dispatch_sgpr,
                                                    std::vector<std::string> &errors) {
  const std::span<const uint8_t> current_image = patcher.image_bytes();
  if (descriptor_file_offset > current_image.size() ||
      sizeof(KD) > current_image.size() - descriptor_file_offset) {
    errors.emplace_back("ConSan MOI dispatch-ID descriptor exceeds ELF bytes");
    return false;
  }
  if (!plan.supported() || static_cast<uint32_t>(persistent_dispatch_sgpr) + 2u > kMaxSgprs) {
    errors.emplace_back("ConSan MOI dispatch-ID descriptor has an unsupported preload plan");
    return false;
  }

  KD desc{};
  std::memcpy(&desc, current_image.data() + descriptor_file_offset, sizeof(desc));
  const ConSanMoiDispatchIdPreloadPlan current_plan = moi_descriptor_dispatch_id_preload_plan(desc);
  if (current_plan.support != plan.support ||
      current_plan.dispatch_id_sgpr != plan.dispatch_id_sgpr ||
      current_plan.original_user_sgpr_count != plan.original_user_sgpr_count ||
      current_plan.system_sgpr_count != plan.system_sgpr_count) {
    errors.emplace_back("ConSan MOI dispatch-ID descriptor changed after preload planning");
    return false;
  }

  if (plan.descriptor_change_required()) {
    AMDHSA_BITS_SET(desc.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID,
                    1u);
    AMDHSA_BITS_SET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                    plan.expanded_user_sgpr_count);
  }
  const uint32_t required_sgpr_count = std::max<uint32_t>(
      plan.required_sgpr_count, static_cast<uint32_t>(persistent_dispatch_sgpr) + 2u);
  if (!grow_moi_descriptor_sgpr_allocation(desc, required_sgpr_count)) {
    errors.emplace_back("ConSan MOI could not grow dispatch-ID descriptor SGPR allocation");
    return false;
  }
  if (!patcher.patch_kernel_descriptor(
          descriptor_file_offset,
          std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&desc), sizeof(desc)))) {
    errors.emplace_back("ConSan MOI could not patch dispatch-ID descriptor state");
    return false;
  }
  return true;
}

void rollback_moi_dispatch_id_as_unsupported(ConSanResult &result, std::string reason) {
  result.outcome = ConSanTransformOutcome::Unsupported;
  result.modified = false;
  result.final_validation_passed = false;
  result.elf_bytes.clear();
  result.patches.clear();
  result.warnings.emplace_back("ConSan MOI dispatch-ID capture is unsupported: " +
                               std::move(reason));
}

void note_dispatch_id_patch_info(ConSanPatchInfo &info, const ConSanMoiDispatchIdPreloadPlan &plan,
                                 uint16_t persistent_dispatch_sgpr) {
  info.dispatch_id_capture_sgpr = persistent_dispatch_sgpr;
  info.dispatch_id_source_sgpr = plan.dispatch_id_sgpr;
  info.dispatch_id_original_user_sgpr_count = plan.original_user_sgpr_count;
  info.dispatch_id_expanded_user_sgpr_count = plan.expanded_user_sgpr_count;
  info.dispatch_id_system_sgpr_count = plan.system_sgpr_count;
  info.dispatch_id_required_sgpr_count = static_cast<uint16_t>(std::max<uint32_t>(
      plan.required_sgpr_count, static_cast<uint32_t>(persistent_dispatch_sgpr) + 2u));
  info.dispatch_id_preload_inserted = plan.descriptor_change_required();
}

void try_apply_private_epoch_prologue_patch(const ConSanOptions &options, rj_code_arch_t arch,
                                            ConSanResult &result) {
  if (!result.modified || result.elf_bytes.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI private-epoch prologue skipped because no private-epoch access probe was "
        "emitted");
    return;
  }

  std::span<const uint8_t> active_bytes(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI private-epoch prologue found no .text section");
    return;
  }

  struct PlannedPrivateEpochPrologue {
    const ConSanKernelInfo *kernel = nullptr;
    uint64_t active_descriptor_file_offset = 0;
    uint64_t active_entry_text_offset = 0;
    uint16_t scratch_vgpr = 0;
    MoiPrivateEpochLayout layout;
    VgprSpillSequence spill;
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    uint32_t required_private_bytes = 0;
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;
    bool has_kernarg_preload = false;
  };
  std::vector<PlannedPrivateEpochPrologue> planned;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    const auto access_patch =
        std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
          return (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
                  patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore ||
                  patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
                  patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore) &&
                 patch.scratch_vgpr && kernel_owns_patch(kernel, patch);
        });
    if (access_patch == result.patches.end())
      continue;

    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel.name;
        });
    if (active_kernel == code_object.kernels().end()) {
      result.warnings.emplace_back(
          "ConSan MOI private-epoch prologue could not resolve an active kernel descriptor");
      continue;
    }

    if (!access_patch->persistent_epoch_private_offset)
      continue;
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;
    bool has_kernarg_preload = false;
    if (options.moi_dispatch_id_sgpr) {
      if (active_kernel->descriptor_file_offset > active_bytes.size() ||
          sizeof(KD) > active_bytes.size() - active_kernel->descriptor_file_offset) {
        rollback_moi_dispatch_id_as_unsupported(result,
                                                "private-epoch descriptor exceeds ELF bytes");
        return;
      }
      KD descriptor{};
      std::memcpy(&descriptor, active_bytes.data() + active_kernel->descriptor_file_offset,
                  sizeof(descriptor));
      dispatch_plan = moi_descriptor_dispatch_id_preload_plan(descriptor);
      if (!dispatch_plan->supported() ||
          *options.moi_dispatch_id_sgpr < dispatch_plan->required_sgpr_count) {
        rollback_moi_dispatch_id_as_unsupported(
            result, "private-epoch kernel has an invalid or overlapping preload layout");
        return;
      }
      has_kernarg_preload = moi_descriptor_has_kernarg_preload(descriptor);
      if (has_kernarg_preload && kernel.code_size <= kAmdhsaKernelEntryAlignment) {
        rollback_moi_dispatch_id_as_unsupported(
            result, "kernarg-preload kernel has no valid secondary hardware entry");
        return;
      }
    }
    const bool has_sample_sequence = options.moi_sample_sequence_vgpr.has_value();
    const bool has_private_owner = access_patch->persistent_owner_private_offset.has_value();
    uint32_t persistent_end =
        *access_patch->persistent_epoch_private_offset + SpillManager::kSlotBytes;
    if (has_private_owner)
      persistent_end = std::max(persistent_end, *access_patch->persistent_owner_private_offset +
                                                    SpillManager::kSlotBytes);
    const std::optional<uint32_t> sample_sequence_offset =
        has_sample_sequence ? std::optional<uint32_t>(persistent_end) : std::nullopt;
    if (sample_sequence_offset)
      persistent_end += SpillManager::kSlotBytes;
    MoiPrivateEpochLayout layout{
        .epoch_offset = *access_patch->persistent_epoch_private_offset,
        .owner_offset = access_patch->persistent_owner_private_offset,
        .sample_sequence_offset = sample_sequence_offset,
        .ephemeral_base = access_patch->persistent_private_state_end.value_or(persistent_end)};

    SpillManager manager(layout.ephemeral_base, kMaxAddressFreeScratchPrivateBytes);
    const std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow =
        access_patch->workgroup_shadow_size == 0u
            ? std::nullopt
            : std::optional<ConSanMoiWorkgroupShadowLayout>(ConSanMoiWorkgroupShadowLayout{
                  .base = access_patch->workgroup_shadow_base,
                  .size = access_patch->workgroup_shadow_size,
                  .required_group_segment_size = access_patch->required_group_segment_size,
              });
    auto spill = build_vgpr_spill_sequence(manager, *access_patch->scratch_vgpr,
                                           workgroup_shadow ? 2u : 1u, arch);
    if (!spill) {
      result.warnings.emplace_back(
          "ConSan MOI private-epoch prologue could not preserve its temporary VGPR");
      continue;
    }
    uint32_t required_private_bytes = spill->total_private_bytes;
    for (const ConSanPatchInfo &patch : result.patches) {
      if (kernel_owns_patch(kernel, patch))
        required_private_bytes =
            std::max(required_private_bytes, patch.required_private_segment_size);
    }
    private_requirements[kernel.descriptor_file_offset] = required_private_bytes;
    planned.push_back({&kernel, active_kernel->descriptor_file_offset,
                       active_kernel->entry_text_offset, *access_patch->scratch_vgpr, layout,
                       *spill, workgroup_shadow, required_private_bytes, dispatch_plan,
                       has_kernarg_preload});
  }

  if (planned.empty()) {
    result.warnings.emplace_back("ConSan MOI private-epoch prologue found no patchable kernels");
    return;
  }
  if (!apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors))
    return;
  if (options.moi_dispatch_id_sgpr) {
    for (const PlannedPrivateEpochPrologue &item : planned) {
      if (!item.dispatch_plan ||
          !apply_moi_dispatch_id_descriptor(patcher, item.active_descriptor_file_offset,
                                            *item.dispatch_plan, *options.moi_dispatch_id_sgpr,
                                            result.errors)) {
        return;
      }
    }
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  patches.reserve(planned.size());
  for (const PlannedPrivateEpochPrologue &item : planned) {
    append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
    const uint64_t prologue_text_offset = static_cast<uint64_t>(new_text.size());
    auto words = build_private_epoch_prologue_words(
        prologue_text_offset, item.active_entry_text_offset, item.scratch_vgpr,
        item.layout.epoch_offset, item.layout.owner_offset, item.layout.sample_sequence_offset,
        item.spill, item.workgroup_shadow, item.dispatch_plan, options.moi_dispatch_id_sgpr,
        moi_runtime_sample_counter_sgpr(options), options.moi_exec_save_sgpr, arch, result.errors);
    if (!words)
      return;
    if (!patcher.redirect_kernel_entry(item.active_descriptor_file_offset,
                                       item.active_entry_text_offset, prologue_text_offset)) {
      result.errors.emplace_back(
          "ConSan MOI private-epoch prologue could not redirect kernel entry");
      return;
    }
    append_words_bytes(new_text, *words);
    if (item.has_kernarg_preload) {
      if (words->size() * sizeof(uint32_t) > kAmdhsaKernelEntryAlignment) {
        rollback_moi_dispatch_id_as_unsupported(
            result, "private-epoch kernarg-preload prologue exceeds its 256-byte window");
        return;
      }
      append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
      const uint64_t secondary_prologue_offset = static_cast<uint64_t>(new_text.size());
      auto secondary_words = build_private_epoch_prologue_words(
          secondary_prologue_offset, item.active_entry_text_offset + kAmdhsaKernelEntryAlignment,
          item.scratch_vgpr, item.layout.epoch_offset, item.layout.owner_offset,
          item.layout.sample_sequence_offset, item.spill, item.workgroup_shadow, item.dispatch_plan,
          options.moi_dispatch_id_sgpr, moi_runtime_sample_counter_sgpr(options),
          options.moi_exec_save_sgpr, arch, result.errors);
      if (!secondary_words ||
          secondary_words->size() * sizeof(uint32_t) > kAmdhsaKernelEntryAlignment) {
        rollback_moi_dispatch_id_as_unsupported(
            result, "could not encode the secondary kernarg-preload prologue");
        return;
      }
      append_words_bytes(new_text, *secondary_words);
    }

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
    info.anchor_offset = item.kernel->entry_text_offset;
    info.trampoline_offset = prologue_text_offset;
    info.trampoline_size = static_cast<uint32_t>(words->size() * sizeof(uint32_t));
    info.scratch_vgpr = item.scratch_vgpr;
    info.persistent_epoch_private_offset = item.layout.epoch_offset;
    info.persistent_owner_private_offset = item.layout.owner_offset;
    info.persistent_private_state_end = item.layout.ephemeral_base;
    info.spilled_vgpr_count = item.workgroup_shadow ? 2u : 1u;
    info.required_private_segment_size = item.required_private_bytes;
    if (item.dispatch_plan && options.moi_dispatch_id_sgpr)
      note_dispatch_id_patch_info(info, *item.dispatch_plan, *options.moi_dispatch_id_sgpr);
    info.trampoline_size = static_cast<uint32_t>(new_text.size() - prologue_text_offset);
    info.owner_descriptor_file_offsets.push_back(item.kernel->descriptor_file_offset);
    patches.push_back(info);
  }

  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI private-epoch prologue could not grow .text");
    return;
  }
  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void try_apply_owner_epoch_prologue_patch(std::span<const uint8_t> bytes,
                                          const ConSanOptions &options, rj_code_arch_t arch,
                                          ConSanResult &result) {
  if (!options.moi_init_owner_epoch)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI owner/epoch prologue currently supports only RDNA4");
    return;
  }
  if (options.automatic_moi_private_epoch) {
    try_apply_private_epoch_prologue_patch(options, arch, result);
    return;
  }
  if (!options.moi_owner_vgpr || !options.moi_epoch_vgpr) {
    result.errors.emplace_back(
        "ConSan MOI owner/epoch prologue requires RJ_CONSAN_MOI_OWNER_VGPR and "
        "RJ_CONSAN_MOI_EPOCH_VGPR");
    return;
  }
  if (*options.moi_owner_vgpr == *options.moi_epoch_vgpr) {
    result.errors.emplace_back("ConSan MOI owner and epoch VGPRs must be distinct");
    return;
  }
  if (options.moi_owner_source == ConSanMoiOwnerSource::HwId && !options.moi_owner_sgpr) {
    result.errors.emplace_back("ConSan MOI hw_id owner source requires RJ_CONSAN_MOI_OWNER_SGPR");
    return;
  }
  if (result.kernels.empty()) {
    result.warnings.emplace_back("ConSan MOI owner/epoch prologue found no kernel descriptors");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI owner/epoch prologue found no .text section");
    return;
  }
  uint32_t required_vgpr_count =
      std::max<uint32_t>(*options.moi_owner_vgpr, *options.moi_epoch_vgpr) + 1u;
  if (options.moi_sample_sequence_vgpr)
    required_vgpr_count =
        std::max<uint32_t>(required_vgpr_count, *options.moi_sample_sequence_vgpr + 1u);
  if (options.moi_workgroup_key_vgpr)
    required_vgpr_count =
        std::max<uint32_t>(required_vgpr_count, *options.moi_workgroup_key_vgpr + 1u);
  const uint32_t owner_required_sgpr_count =
      options.moi_owner_source == ConSanMoiOwnerSource::HwId
          ? static_cast<uint32_t>(*options.moi_owner_sgpr) + 1u
          : 0u;
  struct PlannedOwnerEpochPrologue {
    ConSanKernelInfo kernel;
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    bool has_kernarg_preload = false;
  };
  std::vector<PlannedOwnerEpochPrologue> target_kernels;
  target_kernels.reserve(result.kernels.size());
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range)
      continue;
    const bool owns_emitted_patch =
        std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
          return kernel_owns_patch(kernel, patch);
        });
    const bool owns_planned_site =
        std::ranges::any_of(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
          return std::ranges::find(plan.owner_descriptor_file_offsets,
                                   kernel.descriptor_file_offset) !=
                     plan.owner_descriptor_file_offsets.end() &&
                 plan.source != ConSanRegisterAllocationSource::Unsupported;
        });
    // Automatic persistent state exists only to serve emitted instrumentation.
    // Resource planning deliberately inventories more sites than selection may
    // admit (for example under max_patches or a kernel filter). Mutating every
    // descriptor/entry that merely owns one of those speculative plans can
    // perturb otherwise untouched kernels and, on real multi-kernel modules,
    // can make execution hang before the selected probe is reached.
    if (options.automatic_moi_persistent_vgprs && !owns_emitted_patch)
      continue;
    if (options.moi_dispatch_id_sgpr && !owns_emitted_patch && !owns_planned_site) {
      continue;
    }
    const auto active_kernel =
        std::ranges::find_if(code_object.kernels(), [&](const AmdGpuKernelInfo &candidate) {
          return candidate.name == kernel.name;
        });
    if (active_kernel == code_object.kernels().end()) {
      result.warnings.emplace_back(
          "ConSan MOI owner/epoch prologue could not resolve an active kernel descriptor");
      continue;
    }
    ConSanKernelInfo active = kernel;
    active.descriptor_file_offset = active_kernel->descriptor_file_offset;
    active.entry_text_offset = active_kernel->entry_text_offset;
    active.text_file_offset = active_kernel->text_file_offset;
    if (active.descriptor_file_offset > active_bytes.size() ||
        sizeof(KD) > active_bytes.size() - active.descriptor_file_offset) {
      if (options.moi_dispatch_id_sgpr) {
        rollback_moi_dispatch_id_as_unsupported(result, "owner descriptor exceeds ELF bytes");
        return;
      }
      result.errors.emplace_back("ConSan MOI owner/epoch descriptor exceeds ELF bytes");
      return;
    }
    KD descriptor{};
    std::memcpy(&descriptor, active_bytes.data() + active.descriptor_file_offset,
                sizeof(descriptor));
    std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_plan;
    if (options.moi_dispatch_id_sgpr) {
      dispatch_plan = moi_descriptor_dispatch_id_preload_plan(descriptor);
      if (!dispatch_plan->supported() ||
          *options.moi_dispatch_id_sgpr < dispatch_plan->required_sgpr_count) {
        rollback_moi_dispatch_id_as_unsupported(
            result, "kernel '" + kernel.name + "' has an invalid or overlapping preload layout");
        return;
      }
    }
    const bool has_kernarg_preload = moi_descriptor_has_kernarg_preload(descriptor);
    if (has_kernarg_preload && kernel.code_size <= kAmdhsaKernelEntryAlignment) {
      if (options.moi_dispatch_id_sgpr) {
        rollback_moi_dispatch_id_as_unsupported(
            result, "kernel '" + kernel.name + "' has no valid secondary hardware entry");
      } else {
        result.errors.emplace_back(
            "ConSan MOI kernarg-preload kernel has no valid secondary hardware entry");
      }
      return;
    }
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    for (const ConSanPatchInfo &patch : result.patches) {
      if (!kernel_owns_patch(kernel, patch))
        continue;
      if (patch.workgroup_shadow_size == 0u)
        continue;
      const ConSanMoiWorkgroupShadowLayout candidate{
          .base = patch.workgroup_shadow_base,
          .size = patch.workgroup_shadow_size,
          .required_group_segment_size = patch.required_group_segment_size,
      };
      if (workgroup_shadow &&
          (workgroup_shadow->base != candidate.base || workgroup_shadow->size != candidate.size ||
           workgroup_shadow->required_group_segment_size !=
               candidate.required_group_segment_size)) {
        result.errors.emplace_back(
            "ConSan MOI owner/epoch prologue found incompatible workgroup-shadow layouts");
        return;
      }
      workgroup_shadow = candidate;
    }
    const bool owns_inline_barrier =
        std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
          return kernel_owns_patch(kernel, patch) &&
                 patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
        });
    const bool owns_non_barrier_instrumentation =
        std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
          return kernel_owns_patch(kernel, patch) &&
                 patch.kind != ConSanPatchKind::TrampolineMoiInlineEpochBarrier &&
                 patch.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
        });
    if (options.moi_engine == ConSanMoiEngine::InlineShadow &&
        options.automatic_moi_persistent_vgprs && owns_inline_barrier &&
        !owns_non_barrier_instrumentation) {
      result.warnings.emplace_back(
          "ConSan MOI skipped unobserved owner/epoch initialization for a barrier-only kernel");
      continue;
    }
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    if (options.moi_workgroup_key_vgpr) {
      std::vector<std::string> source_errors;
      workgroup_sources = moi_descriptor_workgroup_sources(
          active_bytes, active.descriptor_file_offset, arch, source_errors);
      if (!workgroup_sources) {
        result.errors.insert(result.errors.end(), source_errors.begin(), source_errors.end());
        return;
      }
    }
    target_kernels.push_back({std::move(active), dispatch_plan, workgroup_shadow, workgroup_sources,
                              has_kernarg_preload});
  }
  for (const PlannedOwnerEpochPrologue &item : target_kernels) {
    uint32_t required_sgpr_count = owner_required_sgpr_count;
    if (options.moi_dispatch_id_sgpr) {
      required_sgpr_count = std::max<uint32_t>(
          required_sgpr_count, static_cast<uint32_t>(*options.moi_dispatch_id_sgpr) +
                                   moi_dispatch_state_sgpr_count(options));
    }
    if (!grow_moi_kernel_descriptor_registers(patcher, active_bytes, item.kernel,
                                              required_vgpr_count, required_sgpr_count,
                                              result.errors)) {
      return;
    }
    if (item.dispatch_plan && options.moi_dispatch_id_sgpr &&
        !apply_moi_dispatch_id_descriptor(patcher, item.kernel.descriptor_file_offset,
                                          *item.dispatch_plan, *options.moi_dispatch_id_sgpr,
                                          result.errors)) {
      return;
    }
  }

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  std::vector<ConSanPatchInfo> patches;
  MoiLocalNopIslandAllocator entry_islands(old_text, result, arch, 7u);
  for (const PlannedOwnerEpochPrologue &item : target_kernels) {
    const ConSanKernelInfo &kernel = item.kernel;

    // Keep the hardware entry stable and displace its first instruction
    // through the same local-island mechanism used by ordinary long-range
    // probes. Besides being required by dynamic-stack kernels, this avoids a
    // gfx12 runtime failure observed when a large instrumented module redirects
    // a descriptor entry across particular late .text placement boundaries.
    const bool requires_in_place_entry = kernel.uses_dynamic_stack.value_or(false);
    bool instrument_entry_in_place =
        requires_in_place_entry || (options.moi_engine == ConSanMoiEngine::InlineShadow &&
                                    options.automatic_moi_persistent_vgprs);
    if (instrument_entry_in_place && item.has_kernarg_preload) {
      if (options.automatic_moi_persistent_vgprs) {
        // The in-place relay currently represents one hardware entry. Retain
        // the paired descriptor-redirection path when kernarg preloading
        // exposes a second entry that must run identical initialization.
        instrument_entry_in_place = false;
      } else {
        result.errors.emplace_back(
            "ConSan MOI dynamic-stack owner/epoch entry trampoline does not yet support a "
            "secondary kernarg-preload entry");
        return;
      }
    }
    std::vector<uint32_t> displaced_entry_words;
    std::optional<uint64_t> entry_island_offset;
    bool indirect_at_entry = false;
    uint64_t prologue_return_target = kernel.entry_text_offset;
    if (instrument_entry_in_place) {
      std::vector<std::string> relocation_errors;
      auto decoded = decode_relocatable_entry_instruction(old_text, kernel.entry_text_offset, arch,
                                                          relocation_errors);
      if (!decoded) {
        if (requires_in_place_entry) {
          result.errors.insert(result.errors.end(), relocation_errors.begin(),
                               relocation_errors.end());
          return;
        }
        // A non-dynamic kernel with a PC-relative first instruction cannot
        // safely displace that instruction. Retain descriptor redirection as
        // the compatibility fallback for this uncommon entry shape.
        instrument_entry_in_place = false;
      } else {
        displaced_entry_words = std::move(*decoded);
        prologue_return_target += displaced_entry_words.size() * sizeof(uint32_t);
      }
    }

    append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
    const uint64_t prologue_text_offset = static_cast<uint64_t>(new_text.size());
    if (instrument_entry_in_place &&
        !compute_sopp_branch_simm16(kernel.entry_text_offset, prologue_text_offset)) {
      entry_island_offset = entry_islands.claim_reachable(kernel.entry_text_offset);
      if (!entry_island_offset) {
        constexpr size_t kDirectEntryRelayWords = 7u;
        while (displaced_entry_words.size() < kDirectEntryRelayWords) {
          auto decoded = decode_relocatable_entry_instruction(
              old_text, kernel.entry_text_offset + displaced_entry_words.size() * sizeof(uint32_t),
              arch, result.errors);
          if (!decoded)
            return;
          displaced_entry_words.insert(displaced_entry_words.end(), decoded->begin(),
                                       decoded->end());
        }
        indirect_at_entry = true;
        prologue_return_target =
            kernel.entry_text_offset + displaced_entry_words.size() * sizeof(uint32_t);
      }
    }
    const std::optional<uint16_t> owner_shift =
        moi_kernel_owner_shift(active_bytes, kernel, result.errors);
    if (!owner_shift)
      return;
    auto words = build_owner_epoch_prologue_words(
        prologue_text_offset, prologue_return_target, *options.moi_owner_vgpr,
        *options.moi_epoch_vgpr, options.moi_workgroup_key_vgpr, options.moi_sample_sequence_vgpr,
        *owner_shift, options.moi_owner_source, options.moi_owner_sgpr,
        options.moi_engine == ConSanMoiEngine::InlineShadow, item.dispatch_plan,
        options.moi_dispatch_id_sgpr, moi_runtime_sample_counter_sgpr(options),
        options.moi_exec_save_sgpr, item.workgroup_shadow, item.workgroup_sources,
        displaced_entry_words, arch, result.errors);
    if (!words)
      return;
    if (instrument_entry_in_place) {
      size_t first_padding_word = 1u;
      if (indirect_at_entry) {
        const size_t patch_count_before_entry_relay = patches.size();
        if (!emit_moi_local_indirect_kernel_entry_island(
                new_text, kernel.entry_text_offset, prologue_text_offset, kernel.entry_text_offset,
                options, std::span<const uint64_t>(&kernel.descriptor_file_offset, 1u), arch,
                patches, result.errors, "ConSan MOI dynamic-stack owner/epoch prologue")) {
          return;
        }
        // The prologue record below owns this same entry anchor, so retain
        // only one authoritative range for final overlap validation.
        patches.resize(patch_count_before_entry_relay);
        first_padding_word = 7u;
      } else {
        const uint64_t entry_target = entry_island_offset.value_or(prologue_text_offset);
        const auto branch = compute_sopp_branch_simm16(kernel.entry_text_offset, entry_target);
        if (!branch ||
            !write_word_bytes(new_text, kernel.entry_text_offset, build_s_branch(*branch, arch))) {
          result.errors.emplace_back(
              "ConSan MOI owner/epoch prologue could not instrument the dynamic-stack entry");
          return;
        }
      }
      for (size_t word = first_padding_word; word < displaced_entry_words.size(); ++word) {
        if (!write_word_bytes(new_text, kernel.entry_text_offset + word * sizeof(uint32_t),
                              build_s_nop(0, arch))) {
          result.errors.emplace_back(
              "ConSan MOI owner/epoch prologue could not pad its displaced entry instruction");
          return;
        }
      }
      if (!indirect_at_entry && entry_island_offset &&
          !emit_moi_local_indirect_kernel_entry_island(
              new_text, *entry_island_offset, prologue_text_offset, kernel.entry_text_offset,
              options, std::span<const uint64_t>(&kernel.descriptor_file_offset, 1u), arch, patches,
              result.errors, "ConSan MOI dynamic-stack owner/epoch prologue")) {
        return;
      }
    } else if (!patcher.redirect_kernel_entry(kernel.descriptor_file_offset,
                                              kernel.entry_text_offset, prologue_text_offset)) {
      result.errors.emplace_back("ConSan MOI owner/epoch prologue could not redirect kernel entry");
      return;
    }
    append_words_bytes(new_text, *words);
    if (item.has_kernarg_preload) {
      if (words->size() * sizeof(uint32_t) > kAmdhsaKernelEntryAlignment) {
        if (options.moi_dispatch_id_sgpr) {
          rollback_moi_dispatch_id_as_unsupported(
              result, "kernarg-preload owner prologue exceeds its 256-byte window");
        } else {
          result.errors.emplace_back(
              "ConSan MOI kernarg-preload owner prologue exceeds its 256-byte window");
        }
        return;
      }
      append_nop_padding_to_alignment(new_text, kAmdhsaKernelEntryAlignment, arch);
      const uint64_t secondary_prologue_offset = static_cast<uint64_t>(new_text.size());
      auto secondary_words = build_owner_epoch_prologue_words(
          secondary_prologue_offset, kernel.entry_text_offset + kAmdhsaKernelEntryAlignment,
          *options.moi_owner_vgpr, *options.moi_epoch_vgpr, options.moi_workgroup_key_vgpr,
          options.moi_sample_sequence_vgpr, *owner_shift, options.moi_owner_source,
          options.moi_owner_sgpr, options.moi_engine == ConSanMoiEngine::InlineShadow,
          item.dispatch_plan, options.moi_dispatch_id_sgpr,
          moi_runtime_sample_counter_sgpr(options), options.moi_exec_save_sgpr,
          item.workgroup_shadow, item.workgroup_sources, std::span<const uint32_t>{}, arch,
          result.errors);
      if (!secondary_words ||
          secondary_words->size() * sizeof(uint32_t) > kAmdhsaKernelEntryAlignment) {
        if (options.moi_dispatch_id_sgpr) {
          rollback_moi_dispatch_id_as_unsupported(
              result, "could not encode the secondary kernarg-preload owner prologue");
        } else {
          result.errors.emplace_back(
              "ConSan MOI could not encode the secondary kernarg-preload owner prologue");
        }
        return;
      }
      append_words_bytes(new_text, *secondary_words);
    }

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
    info.anchor_offset = kernel.entry_text_offset;
    info.trampoline_offset = prologue_text_offset;
    info.original_size =
        instrument_entry_in_place ? displaced_entry_words.size() * sizeof(uint32_t) : 0u;
    info.trampoline_size = static_cast<uint32_t>(new_text.size() - prologue_text_offset);
    if (item.workgroup_shadow) {
      info.workgroup_shadow_base = item.workgroup_shadow->base;
      info.workgroup_shadow_size = item.workgroup_shadow->size;
      info.required_group_segment_size = item.workgroup_shadow->required_group_segment_size;
    }
    if (item.dispatch_plan && options.moi_dispatch_id_sgpr)
      note_dispatch_id_patch_info(info, *item.dispatch_plan, *options.moi_dispatch_id_sgpr);
    const auto canonical_kernel =
        std::ranges::find(result.kernels, kernel.name, &ConSanKernelInfo::name);
    if (canonical_kernel == result.kernels.end()) {
      result.errors.emplace_back("ConSan MOI owner/epoch prologue lost its canonical kernel owner");
      return;
    }
    info.owner_descriptor_file_offsets.push_back(canonical_kernel->descriptor_file_offset);
    patches.push_back(info);
  }

  if (patches.empty()) {
    result.warnings.emplace_back("ConSan MOI owner/epoch prologue found no patchable kernels");
    return;
  }
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI owner/epoch prologue could not grow .text");
    return;
  }

  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}
