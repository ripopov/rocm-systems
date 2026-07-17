void try_apply_lds_load_check_trap_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                         const ConSanOptions &options, ConSanResult &result) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan LDS check/trap proof currently supports only RDNA4");
    return;
  }

  const auto delay_words =
      delay_instruction_word_count(options, result.errors, "ConSan LDS check/trap proof");
  if (!delay_words)
    return;

  if (!has_supported_check_trap_site(result)) {
    result.warnings.emplace_back("ConSan LDS check/trap proof found no supported "
                                 "ds_load_b{32,64,128}, ds_load_2addr_b{32,64}, "
                                 "ds_load_u16_d16(_hi), ds_store_b{32,64,128}, or "
                                 "ds_store_2addr_b{32,64} site");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  uint64_t appended_cave_offset = 0;
  std::span<const uint8_t> pristine_text;
  if (code_object.text_sections().size() == 1) {
    appended_cave_offset = code_object.text_sections().front()->size();
    pristine_text = std::span<const uint8_t>(
        reinterpret_cast<const uint8_t *>(code_object.text_sections().front()->data()),
        code_object.text_sections().front()->size());
  }
  std::vector<LocalNopCave> local_nop_caves = find_uncovered_nop_caves(code_object, result, arch);
  std::erase_if(local_nop_caves, [&](const LocalNopCave &cave) {
    return overlaps_preapplied_reserved_range(
        options, {cave.text_offset,
                  cave.text_offset + static_cast<uint64_t>(cave.word_count) * sizeof(uint32_t)});
  });
  uint32_t max_uncovered_nop_cave_words = 0;
  for (const LocalNopCave &cave : local_nop_caves)
    max_uncovered_nop_cave_words = std::max(max_uncovered_nop_cave_words, cave.word_count);

  // A full check body consumes far more padding than its seven-word absolute
  // entry jump. Large generated objects commonly have enough aggregate local
  // padding for every site's entry island, but spread it across a smaller
  // number of long caves. Treat each non-overlapping seven-word slice as an
  // independently reservable island instead of wasting the remainder after
  // the first site claims a cave.
  size_t local_entry_island_capacity = 0;
  for (const LocalNopCave &cave : local_nop_caves)
    local_entry_island_capacity += cave.word_count / kLdsCheckTrapIndirectJumpWords;
  std::vector<LocalNopCave> local_entry_islands;
  local_entry_islands.reserve(local_entry_island_capacity);
  for (const LocalNopCave &cave : local_nop_caves) {
    const uint32_t island_count = cave.word_count / kLdsCheckTrapIndirectJumpWords;
    for (uint32_t island_index = 0; island_index < island_count; ++island_index) {
      const uint64_t word_offset =
          static_cast<uint64_t>(island_index) * kLdsCheckTrapIndirectJumpWords;
      local_entry_islands.push_back(LocalNopCave{
          .text_offset = cave.text_offset + word_offset * sizeof(uint32_t),
          .file_offset = cave.file_offset + word_offset * sizeof(uint32_t),
          .word_count = static_cast<uint32_t>(kLdsCheckTrapIndirectJumpWords),
          .owner_entry_text_offset = cave.owner_entry_text_offset,
      });
    }
  }

  std::vector<std::unique_ptr<BasicBlock>> blocks;
  std::vector<BasicBlock *> block_ptrs;
  struct ScReservoirOwnerContext {
    uint64_t descriptor_file_offset = 0;
    KernelCfgScope scope;
    std::unique_ptr<LivenessAnalysis> liveness;
    uint32_t wavefront_size = 0;
  };
  std::vector<ScReservoirOwnerContext> reservoir_owner_contexts;
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (decoder) {
    std::vector<uint64_t> extra_leaders = kernel_entry_offsets(result);
    extra_leaders.reserve(extra_leaders.size() + result.functions.size());
    for (const ConSanFunctionInfo &function : result.functions)
      extra_leaders.push_back(function.entry_text_offset);
    std::ranges::sort(extra_leaders);
    extra_leaders.erase(std::ranges::unique(extra_leaders).begin(), extra_leaders.end());

    std::vector<BasicBlock::CodeRange> code_ranges;
    code_ranges.reserve(code_object.functions().size());
    for (const AmdGpuFunctionInfo &function : code_object.functions()) {
      if (function.code_size != 0)
        code_ranges.push_back(
            {.start_offset = function.entry_text_offset, .size = function.code_size});
    }
    blocks = BasicBlock::build(code_object, *decoder, arch, extra_leaders, code_ranges);
    block_ptrs = block_ptrs_for(blocks);

    const BlockOffsetIndex block_index = build_block_offset_index(blocks);
    std::vector<uint64_t> kernel_entries;
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (kernel.has_text_range)
        kernel_entries.push_back(kernel.entry_text_offset);
    }
    std::ranges::sort(kernel_entries);
    kernel_entries.erase(std::ranges::unique(kernel_entries).begin(), kernel_entries.end());
    reservoir_owner_contexts.reserve(result.kernels.size());
    for (const ConSanKernelInfo &kernel : result.kernels) {
      if (!kernel.has_text_range || kernel.descriptor_file_offset > original_bytes.size() ||
          sizeof(KD) > original_bytes.size() - kernel.descriptor_file_offset)
        continue;
      auto scope =
          build_kernel_cfg_scope(blocks, block_index,
                                 KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                                                    .additional_entry_offsets = {}},
                                 kernel_entries, pristine_text);
      if (!scope)
        continue;
      KD descriptor{};
      std::memcpy(&descriptor, original_bytes.data() + kernel.descriptor_file_offset,
                  sizeof(descriptor));
      reservoir_owner_contexts.push_back(
          {.descriptor_file_offset = kernel.descriptor_file_offset,
           .scope = std::move(*scope),
           .liveness = nullptr,
           .wavefront_size = descriptor_wavefront_size(arch, descriptor)});
      ScReservoirOwnerContext &context = reservoir_owner_contexts.back();
      context.liveness = std::make_unique<LivenessAnalysis>(KernelBlockScope(context.scope.blocks),
                                                            LivenessAnalysisOptions{},
                                                            context.scope.liveness_edges);
    }
  }

  struct LdsCheckTrapCandidate {
    const ConSanLdsSite *site = nullptr;
    const LocalNopCave *local_cave = nullptr;
    const LocalNopCave *entry_island = nullptr;
    bool entry_island_is_appended = false;
    uint64_t kernel_entry_text_offset = 0;
    uint64_t descriptor_file_offset = 0;
    uint64_t appended_cave_text_offset = 0;
    uint16_t scratch_vgpr = 0;
    uint16_t vcc_save_sgpr = 0;
    std::optional<uint16_t> indirect_pc_sgpr;
    std::optional<uint16_t> indirect_scc_save_sgpr;
    uint64_t requested_words = 0;
    bool use_local_cave = false;
    bool use_appended_cave = false;
    std::optional<uint16_t> required_vgpr_allocation_count;
    std::optional<DbiPatchPlacement> placement;
    std::vector<uint64_t> entry_relay_offsets = {};
  };

  // A wide relay donor combines a relocatable two-dword instruction with a
  // pristine six-dword sequence. The sequence's appended body preserves its
  // ordinary execution path; its skipped payload preserves the two-dword
  // instruction and contributes three capacity-one relay words.
  struct ScRelayAnchorCandidate {
    uint64_t anchor_offset = 0;
    std::array<uint32_t, 2> anchor_original_words{};
  };
  struct ScWideRelayHostCandidate {
    uint64_t host_offset = 0;
    std::vector<uint32_t> host_original_words;
    std::vector<uint32_t> prefix_word_counts;
  };
  struct ScWideCompositeBranchRelayDonor {
    uint64_t anchor_offset = 0;
    std::array<uint32_t, 2> anchor_original_words{};
    uint64_t host_offset = 0;
    std::vector<uint32_t> host_original_words;
    DbiPatchPlacement host_placement;
  };
  struct ScRelayReservoirCandidate {
    uint64_t anchor_offset = 0;
    uint32_t original_size = 0;
    const BasicBlock *block = nullptr;
    uint16_t pc_sgpr_pair = 0;
    uint16_t saved_vcc_sgpr = 0;
    uint16_t return_pc_sgpr_pair = 0;
    uint16_t return_saved_vcc_sgpr = 0;
    uint16_t required_sgpr_count = 0;
    std::vector<uint64_t> owner_descriptor_file_offsets;
  };
  struct ScRelayReservoir {
    uint64_t anchor_offset = 0;
    uint32_t original_size = 0;
    uint16_t pc_sgpr_pair = 0;
    uint16_t saved_vcc_sgpr = 0;
    uint16_t return_pc_sgpr_pair = 0;
    uint16_t return_saved_vcc_sgpr = 0;
    uint16_t required_sgpr_count = 0;
    std::vector<uint64_t> owner_descriptor_file_offsets;
    DbiPatchPlacement placement;
  };

  const auto range_is_in_selected_container = [&](uint64_t offset, uint64_t size) {
    const auto selected = [&](const auto &container) {
      if (!options.test_kernel_name_filter.empty() &&
          container.name.find(options.test_kernel_name_filter) == std::string::npos)
        return false;
      return offset >= container.entry_text_offset && size <= container.code_size &&
             offset - container.entry_text_offset <= container.code_size - size;
    };
    if (std::ranges::any_of(result.kernels, [&](const ConSanKernelInfo &kernel) {
          return kernel.preflight_action != ConSanPreflightAction::Reject && selected(kernel);
        }))
      return true;
    return std::ranges::any_of(result.functions, selected);
  };
  const auto overlaps_future_flat_probe = [&](uint64_t offset, uint64_t size) {
    if (!options.probe_flat_check_trap)
      return false;
    const ByteRange candidate{offset, offset + size};
    const auto container_overlaps = [&](const auto &container) {
      if (!options.test_kernel_name_filter.empty() &&
          container.name.find(options.test_kernel_name_filter) == std::string::npos)
        return false;
      return std::ranges::any_of(container.flat_sites, [&](const ConSanFlatSite &site) {
        return is_supported_flat_check_trap_site(site, options.flat_provenance_mode) &&
               ranges_overlap(candidate, {site.text_offset, site.text_offset + site.size});
      });
    };
    return std::ranges::any_of(result.kernels, container_overlaps) ||
           std::ranges::any_of(result.functions, container_overlaps);
  };
  const auto overlaps_supported_lds_probe = [&](uint64_t offset, uint64_t size) {
    const ByteRange candidate{offset, offset + size};
    const auto container_overlaps = [&](const auto &container) {
      if (!options.test_kernel_name_filter.empty() &&
          container.name.find(options.test_kernel_name_filter) == std::string::npos)
        return false;
      return std::ranges::any_of(container.lds_sites, [&](const ConSanLdsSite &site) {
        return is_supported_check_trap_site(site) &&
               ranges_overlap(candidate, {site.text_offset, site.text_offset + site.size});
      });
    };
    return std::ranges::any_of(result.kernels, container_overlaps) ||
           std::ranges::any_of(result.functions, container_overlaps);
  };

  std::vector<const Instruction *> ordered_instructions;
  for (BasicBlock *block : block_ptrs) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions())
      ordered_instructions.push_back(&inst);
  }
  std::ranges::sort(ordered_instructions, {},
                    [](const Instruction *inst) { return inst->src_loc(); });
  std::unordered_set<uint64_t> s_clause_blocked_offsets;
  uint32_t clause_remaining = 0;
  std::optional<uint64_t> preceding_end;
  for (const Instruction *inst : ordered_instructions) {
    if (preceding_end && inst->src_loc() != *preceding_end)
      clause_remaining = 0;
    if (clause_remaining > 0) {
      s_clause_blocked_offsets.insert(inst->src_loc());
      --clause_remaining;
    }
    if (is_s_clause(*inst))
      clause_remaining = s_clause_following_instruction_count(*inst);
    preceding_end = inst->src_loc() + static_cast<uint64_t>(inst->size());
  }

  std::vector<ScRelayAnchorCandidate> relay_anchor_candidates;
  for (BasicBlock *block : block_ptrs) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      const uint64_t offset = inst.src_loc();
      if (inst.size() != 2 * static_cast<int>(sizeof(uint32_t)) ||
          std::string_view(inst.mnemonic()).starts_with("ds_") || inst.raw_encoding() == nullptr ||
          !is_relocatable_anchor(inst, offset, pristine_text, arch) ||
          offset % sizeof(uint32_t) != 0 || !range_is_in_selected_container(offset, inst.size()) ||
          s_clause_blocked_offsets.contains(offset) ||
          overlaps_future_flat_probe(offset, inst.size()) ||
          overlaps_preapplied_reserved_range(
              options, {offset, offset + static_cast<uint64_t>(inst.size())})) {
        continue;
      }
      ScRelayAnchorCandidate candidate;
      candidate.anchor_offset = offset;
      std::memcpy(candidate.anchor_original_words.data(), pristine_text.data() + offset,
                  inst.size());
      relay_anchor_candidates.push_back(candidate);
    }
  }
  std::ranges::sort(relay_anchor_candidates, {}, &ScRelayAnchorCandidate::anchor_offset);

  std::vector<ScWideRelayHostCandidate> relay_wide_host_candidates;
  for (BasicBlock *block : block_ptrs) {
    if (block == nullptr)
      continue;
    auto &instructions = block->instructions();
    for (auto host_it = instructions.begin(); host_it != instructions.end(); ++host_it) {
      const Instruction &host = *host_it;
      const uint64_t host_offset = host.src_loc();
      if (host.size() != 3 * static_cast<int>(sizeof(uint32_t)) ||
          (!std::string_view(host.mnemonic()).starts_with("flat_load") &&
           !std::string_view(host.mnemonic()).starts_with("flat_store")) ||
          host.raw_encoding() == nullptr || s_clause_blocked_offsets.contains(host_offset))
        continue;
      constexpr uint32_t kMaxWideRelayHostWords = 512u;
      uint32_t accumulated_size = 0;
      std::vector<uint32_t> prefix_word_counts;
      auto sequence_it = host_it;
      while (sequence_it != instructions.end() &&
             accumulated_size < kMaxWideRelayHostWords * sizeof(uint32_t)) {
        const Instruction &inst = *sequence_it;
        if (inst.src_loc() != host_offset + accumulated_size || inst.size() <= 0 ||
            !is_sc_relay_reservoir_instruction(inst, inst.src_loc(), pristine_text, arch) ||
            s_clause_blocked_offsets.contains(inst.src_loc()) ||
            overlaps_supported_lds_probe(inst.src_loc(), static_cast<uint64_t>(inst.size())) ||
            overlaps_future_flat_probe(inst.src_loc(), static_cast<uint64_t>(inst.size())) ||
            overlaps_preapplied_reserved_range(
                options, {inst.src_loc(), inst.src_loc() + static_cast<uint64_t>(inst.size())}) ||
            !range_is_in_selected_container(inst.src_loc(), static_cast<uint64_t>(inst.size())) ||
            static_cast<uint32_t>(inst.size()) >
                kMaxWideRelayHostWords * sizeof(uint32_t) - accumulated_size) {
          break;
        }
        accumulated_size += static_cast<uint32_t>(inst.size());
        if (accumulated_size >= 6u * sizeof(uint32_t))
          prefix_word_counts.push_back(accumulated_size / sizeof(uint32_t));
        ++sequence_it;
      }
      if (prefix_word_counts.empty())
        continue;
      ScWideRelayHostCandidate candidate;
      candidate.host_offset = host_offset;
      candidate.host_original_words.resize(accumulated_size / sizeof(uint32_t));
      candidate.prefix_word_counts = std::move(prefix_word_counts);
      std::memcpy(candidate.host_original_words.data(), pristine_text.data() + host_offset,
                  accumulated_size);
      relay_wide_host_candidates.push_back(candidate);
    }
  }
  std::ranges::sort(relay_wide_host_candidates, {}, &ScWideRelayHostCandidate::host_offset);

  // A reservoir relocates one pristine straight-line sequence and turns every
  // skipped word after its entry branch into an independent relay vertex. Keep
  // candidate windows bounded to avoid pathological code growth, while still
  // admitting the hundreds of vertices required by dense generated kernels.
  constexpr uint32_t kMaxScRelayReservoirWords = 512u;
  std::vector<ScRelayReservoirCandidate> relay_reservoir_candidates;
  const auto reservoir_instruction_is_admissible = [&](const Instruction &inst) {
    const uint64_t offset = inst.src_loc();
    const int size = inst.size();
    const std::string_view mnemonic = inst.mnemonic();
    if (size <= 0 || size % static_cast<int>(sizeof(uint32_t)) != 0 ||
        inst.raw_encoding() == nullptr || offset % sizeof(uint32_t) != 0 ||
        mnemonic.starts_with("ds_") || mnemonic == "s_clause" || mnemonic == "s_delay_alu" ||
        s_clause_blocked_offsets.contains(offset) ||
        !range_is_in_selected_container(offset, static_cast<uint64_t>(size)) ||
        overlaps_supported_lds_probe(offset, static_cast<uint64_t>(size)) ||
        overlaps_future_flat_probe(offset, static_cast<uint64_t>(size)) ||
        overlaps_preapplied_reserved_range(options,
                                           {offset, offset + static_cast<uint64_t>(size)})) {
      return false;
    }
    return is_sc_relay_reservoir_instruction(inst, offset, pristine_text, arch);
  };
  for (BasicBlock *block : block_ptrs) {
    if (block == nullptr)
      continue;
    std::vector<std::pair<uint64_t, uint32_t>> run;
    const auto flush_run = [&]() {
      for (size_t begin = 0; begin < run.size(); ++begin) {
        uint32_t bytes = 0;
        size_t end = begin;
        while (end < run.size() &&
               run[end].second <= kMaxScRelayReservoirWords * sizeof(uint32_t) - bytes) {
          bytes += run[end].second;
          ++end;
        }
        if (bytes >= 2u * sizeof(uint32_t))
          relay_reservoir_candidates.push_back({.anchor_offset = run[begin].first,
                                                .original_size = bytes,
                                                .block = block,
                                                .pc_sgpr_pair = 0,
                                                .saved_vcc_sgpr = 0,
                                                .return_pc_sgpr_pair = 0,
                                                .return_saved_vcc_sgpr = 0,
                                                .required_sgpr_count = 0,
                                                .owner_descriptor_file_offsets = {}});
      }
      run.clear();
    };
    std::optional<uint64_t> expected_offset;
    for (const Instruction &inst : block->instructions()) {
      if (!reservoir_instruction_is_admissible(inst) ||
          (expected_offset && inst.src_loc() != *expected_offset)) {
        flush_run();
      }
      if (!reservoir_instruction_is_admissible(inst)) {
        expected_offset.reset();
        continue;
      }
      run.emplace_back(inst.src_loc(), static_cast<uint32_t>(inst.size()));
      expected_offset = inst.src_loc() + static_cast<uint64_t>(inst.size());
    }
    flush_run();
  }
  std::unordered_map<uint64_t, const Instruction *> reservoir_instruction_by_offset;
  for (BasicBlock *block : block_ptrs) {
    if (block == nullptr)
      continue;
    for (const Instruction &instruction : block->instructions())
      reservoir_instruction_by_offset.emplace(instruction.src_loc(), &instruction);
  }
  std::unordered_map<const BasicBlock *, std::vector<const ScReservoirOwnerContext *>>
      reservoir_owners_by_block;
  for (const ScReservoirOwnerContext &context : reservoir_owner_contexts) {
    for (const BasicBlock *block : context.scope.blocks)
      reservoir_owners_by_block[block].push_back(&context);
  }
  std::erase_if(relay_reservoir_candidates, [&](ScRelayReservoirCandidate &candidate) {
    if (candidate.original_size <
            (kScRelayEntryWords + kScRelayTailRestoreWords + 1u) * sizeof(uint32_t) ||
        reservoir_owner_contexts.empty())
      return true;
    const auto entry_it = reservoir_instruction_by_offset.find(candidate.anchor_offset);
    const Instruction *entry_instruction =
        entry_it == reservoir_instruction_by_offset.end() ? nullptr : entry_it->second;
    if (entry_instruction == nullptr)
      return true;
    const auto boundary_it =
        reservoir_instruction_by_offset.find(candidate.anchor_offset + candidate.original_size);
    const Instruction *boundary =
        boundary_it == reservoir_instruction_by_offset.end() ? nullptr : boundary_it->second;
    const BasicBlock *candidate_block = candidate.block;
    if (candidate_block == nullptr)
      return true;
    const auto owners_it = reservoir_owners_by_block.find(candidate_block);
    if (owners_it == reservoir_owners_by_block.end() || owners_it->second.empty())
      return true;
    const std::vector<const ScReservoirOwnerContext *> &owners = owners_it->second;
    if (!std::ranges::all_of(owners,
                             [](const auto *owner) { return owner->wavefront_size == 32u; }))
      return true;
    const auto register_is_dead_at_endpoint = [&](const ScReservoirOwnerContext &owner,
                                                  RegisterRef reg, bool entry) {
      if (entry)
        return !owner.liveness->is_live_before(*entry_instruction, reg);
      if (boundary != nullptr)
        return !owner.liveness->is_live_before(*boundary, reg);
      return !owner.liveness->block_liveness(*candidate_block).live_out.contains(reg);
    };
    const auto find_saved_vcc_sgpr = [&](bool entry) -> std::optional<uint16_t> {
      for (uint16_t sgpr = 0u; sgpr < 104u; ++sgpr) {
        const RegisterRef ref{RegClass::SGPR, sgpr, 1};
        if (std::ranges::all_of(owners, [&](const auto *owner) {
              return register_is_dead_at_endpoint(*owner, ref, entry);
            }))
          return sgpr;
      }
      return std::nullopt;
    };
    const auto entry_saved_vcc = find_saved_vcc_sgpr(true);
    if (!entry_saved_vcc)
      return true;
    const std::optional<uint16_t> return_saved_vcc = find_saved_vcc_sgpr(false);
    if (!return_saved_vcc)
      return true;
    candidate.pc_sgpr_pair = kRdna4VccLo;
    candidate.saved_vcc_sgpr = *entry_saved_vcc;
    candidate.return_pc_sgpr_pair = kRdna4VccLo;
    candidate.return_saved_vcc_sgpr = *return_saved_vcc;
    candidate.required_sgpr_count =
        std::max<uint16_t>(static_cast<uint16_t>(*entry_saved_vcc + 1u),
                           static_cast<uint16_t>(*return_saved_vcc + 1u));
    for (const ScReservoirOwnerContext *owner : owners)
      candidate.owner_descriptor_file_offsets.push_back(owner->descriptor_file_offset);
    std::ranges::sort(candidate.owner_descriptor_file_offsets);
    return false;
  });
  std::ranges::sort(relay_reservoir_candidates, [](const auto &lhs, const auto &rhs) {
    if (lhs.anchor_offset != rhs.anchor_offset)
      return lhs.anchor_offset < rhs.anchor_offset;
    return lhs.original_size < rhs.original_size;
  });
  relay_reservoir_candidates.erase(std::ranges::unique(relay_reservoir_candidates, {},
                                                       [](const auto &candidate) {
                                                         return std::pair(candidate.anchor_offset,
                                                                          candidate.original_size);
                                                       })
                                       .begin(),
                                   relay_reservoir_candidates.end());

  std::vector<LdsCheckTrapCandidate> inline_candidates;
  std::vector<LdsCheckTrapCandidate> local_cave_candidates;
  std::vector<LdsCheckTrapCandidate> appended_cave_candidates;
  std::vector<LdsCheckTrapCandidate> generated_island_seed_candidates;
  bool skipped_supported_site_for_excessive_delay = false;
  size_t supported_candidate_count = 0;
  size_t scratchable_candidate_count = 0;
  size_t append_cave_reachable_candidate_count = 0;
  size_t indirect_pc_resource_failed_site_count = 0;
  size_t indirect_island_unreachable_site_count = 0;
  size_t indirect_candidate_placement_failure_count = 0;
  size_t local_cave_reachable_candidate_count = 0;
  size_t s_clause_blocked_candidate_count = 0;
  uint32_t max_observed_padding_words = 0;
  auto reachable_local_caves = [&](const ConSanLdsSite &site, uint64_t owner_entry_text_offset,
                                   uint64_t requested_words) {
    std::vector<const LocalNopCave *> caves;
    const uint64_t cave_words = requested_words + 1u;
    for (const LocalNopCave &cave : local_nop_caves) {
      if (cave.owner_entry_text_offset != owner_entry_text_offset || cave.word_count < cave_words)
        continue;
      const uint64_t return_branch_pc = cave.text_offset + requested_words * sizeof(uint32_t);
      if (compute_sopp_branch_simm16(site.text_offset, cave.text_offset) &&
          compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size))
        caves.push_back(&cave);
    }
    return caves;
  };
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.preflight_action == ConSanPreflightAction::Reject)
      continue;
    if (!options.test_kernel_name_filter.empty() &&
        kernel.name.find(options.test_kernel_name_filter) == std::string::npos)
      continue;
    const auto owner_context_it =
        std::ranges::find(reservoir_owner_contexts, kernel.descriptor_file_offset,
                          &ScReservoirOwnerContext::descriptor_file_offset);
    if (owner_context_it == reservoir_owner_contexts.end() || owner_context_it->liveness == nullptr)
      continue;
    const KernelBlockScope kernel_scope(owner_context_it->scope.blocks);
    const LivenessAnalysis *kernel_liveness = owner_context_it->liveness.get();
    const auto max_auto_scratch_vgpr =
        descriptor_vgpr_allocation_count(original_bytes, kernel.descriptor_file_offset, arch);
    std::optional<uint16_t> min_auto_scratch_vgpr;
    std::optional<uint16_t> min_preferred_vcc_save_sgpr;
    const KernelMaxRegisterRefs max_refs = max_register_refs_in_kernel(kernel, kernel_scope);
    if (!options.scratch_vgpr) {
      if (max_refs.vgpr && *max_refs.vgpr < 255)
        min_auto_scratch_vgpr = static_cast<uint16_t>(*max_refs.vgpr + 1u);
    }
    if (max_refs.sgpr && *max_refs.sgpr + 1u < REGISTER_SET_ALLOCATABLE_SGPRS)
      min_preferred_vcc_save_sgpr = static_cast<uint16_t>(*max_refs.sgpr + 1u);
    for (const ConSanLdsSite &site : kernel.lds_sites) {
      if (!is_supported_check_trap_site(site))
        continue;
      ++supported_candidate_count;
      auto required_vgprs = lds_dword_count(site);
      if (!required_vgprs)
        continue;
      auto action_words =
          mismatch_action_word_count(options, arch, result.errors, "ConSan LDS check/trap proof");
      if (!action_words)
        return;
      const uint32_t scratch_vgprs = *required_vgprs + report_action_scratch_vgprs(options) +
                                     (lds_load_clobbers_address(site) ? 1u : 0u);
      const uint64_t requested_words = (lds_load_clobbers_address(site) ? 1u : 0u) + 2u +
                                       lds_check_trap_predelay_setup_words(site) +
                                       static_cast<uint64_t>(*delay_words) + 2u + 1u + 1u +
                                       (2u + *action_words) * *required_vgprs + 1u;
      if (requested_words > 256u) {
        skipped_supported_site_for_excessive_delay = true;
        continue;
      }
      const uint32_t padding_words = static_cast<uint32_t>(requested_words - 2u);
      const Instruction *site_inst =
          find_instruction_at_text_offset(kernel_scope, site.text_offset);
      auto scratch =
          choose_scratch_vgpr(site, options, site_inst, kernel_liveness, min_auto_scratch_vgpr,
                              max_auto_scratch_vgpr, scratch_vgprs);
      if (!scratch && !options.scratch_vgpr && max_auto_scratch_vgpr) {
        scratch = choose_scratch_vgpr(site, options, site_inst, kernel_liveness,
                                      min_auto_scratch_vgpr, std::nullopt, scratch_vgprs);
      }
      if (!scratch)
        continue;
      auto required_allocation =
          required_descriptor_vgpr_allocation_for_scratch(site, *scratch, scratch_vgprs);
      if (!required_allocation)
        continue;
      std::optional<uint16_t> descriptor_growth;
      if (max_auto_scratch_vgpr && *required_allocation > *max_auto_scratch_vgpr)
        descriptor_growth = *required_allocation;
      auto vcc_save = choose_vcc_save_sgpr(site_inst, kernel_liveness, min_preferred_vcc_save_sgpr);
      if (!vcc_save)
        continue;
      ++scratchable_candidate_count;
      uint32_t observed_padding =
          count_nop_padding(original_bytes, site.file_offset + site.size, padding_words, arch);
      if (overlaps_preapplied_reserved_range(
              options,
              {site.text_offset,
               site.text_offset + requested_words * static_cast<uint64_t>(sizeof(uint32_t))})) {
        observed_padding = 0;
      }
      max_observed_padding_words = std::max(max_observed_padding_words, observed_padding);
      if (text_offset_is_inside_s_clause(kernel_scope, site.text_offset)) {
        ++s_clause_blocked_candidate_count;
      } else {
        bool appended_cave_reachable = false;
        if (appended_cave_offset != 0) {
          const uint64_t return_branch_pc =
              appended_cave_offset + requested_words * sizeof(uint32_t);
          if (compute_sopp_branch_simm16(site.text_offset, appended_cave_offset) &&
              compute_sopp_branch_simm16(return_branch_pc, site.text_offset + site.size)) {
            appended_cave_reachable = true;
            LdsCheckTrapCandidate candidate{&site,
                                            nullptr,
                                            nullptr,
                                            false,
                                            kernel.entry_text_offset,
                                            kernel.descriptor_file_offset,
                                            appended_cave_offset,
                                            *scratch,
                                            *vcc_save,
                                            std::nullopt,
                                            std::nullopt,
                                            requested_words,
                                            false,
                                            true,
                                            descriptor_growth,
                                            std::nullopt};
            appended_cave_candidates.push_back(candidate);
          }
        }
        const std::vector<const LocalNopCave *> caves =
            reachable_local_caves(site, kernel.entry_text_offset, requested_words);
        if (!caves.empty())
          ++local_cave_reachable_candidate_count;
        for (const LocalNopCave *cave : caves) {
          LdsCheckTrapCandidate candidate{&site,
                                          cave,
                                          nullptr,
                                          false,
                                          kernel.entry_text_offset,
                                          kernel.descriptor_file_offset,
                                          0,
                                          *scratch,
                                          *vcc_save,
                                          std::nullopt,
                                          std::nullopt,
                                          requested_words,
                                          true,
                                          false,
                                          descriptor_growth,
                                          std::nullopt};
          local_cave_candidates.push_back(candidate);
        }
        if (appended_cave_offset != 0) {
          uint16_t indirect_vcc_save_sgpr = *vcc_save;
          auto indirect_pc_sgpr = choose_lds_check_trap_indirect_pc_sgpr(
              site_inst, kernel_liveness, indirect_vcc_save_sgpr,
              min_preferred_vcc_save_sgpr.value_or(0));
          std::optional<uint16_t> indirect_scc_save_sgpr;
          if (indirect_pc_sgpr) {
            indirect_scc_save_sgpr = choose_lds_check_trap_indirect_scc_save_sgpr(
                site_inst, kernel_liveness, indirect_vcc_save_sgpr, *indirect_pc_sgpr,
                min_preferred_vcc_save_sgpr.value_or(0));
          }
          if (!indirect_pc_sgpr || !indirect_scc_save_sgpr) {
            if (auto joint_scratch = choose_lds_check_trap_indirect_scalar_scratch(
                    site_inst, kernel_liveness, min_preferred_vcc_save_sgpr)) {
              indirect_vcc_save_sgpr = joint_scratch->vcc_save_sgpr;
              indirect_pc_sgpr = joint_scratch->pc_sgpr_pair;
              indirect_scc_save_sgpr = joint_scratch->scc_save_sgpr;
            }
          }
          // Prefer four fresh scalar words above every guest reference. Keep
          // dead-register reuse as a bounded fallback for synthetic or
          // maximally allocated kernels until scalar spilling is available.
          if ((!indirect_pc_sgpr || !indirect_scc_save_sgpr) && min_preferred_vcc_save_sgpr) {
            if (auto joint_scratch = choose_lds_check_trap_indirect_scalar_scratch(
                    site_inst, kernel_liveness, std::nullopt)) {
              indirect_vcc_save_sgpr = joint_scratch->vcc_save_sgpr;
              indirect_pc_sgpr = joint_scratch->pc_sgpr_pair;
              indirect_scc_save_sgpr = joint_scratch->scc_save_sgpr;
            }
          }
          if (!indirect_pc_sgpr || !indirect_scc_save_sgpr) {
            if (!appended_cave_reachable)
              ++indirect_pc_resource_failed_site_count;
          } else {
            LdsCheckTrapCandidate generated_seed{&site,
                                                 nullptr,
                                                 nullptr,
                                                 true,
                                                 kernel.entry_text_offset,
                                                 kernel.descriptor_file_offset,
                                                 appended_cave_offset,
                                                 *scratch,
                                                 indirect_vcc_save_sgpr,
                                                 *indirect_pc_sgpr,
                                                 *indirect_scc_save_sgpr,
                                                 requested_words,
                                                 false,
                                                 true,
                                                 descriptor_growth,
                                                 std::nullopt};
            generated_island_seed_candidates.push_back(generated_seed);

            if (!appended_cave_reachable) {
              bool found_reachable_island = false;
              for (const LocalNopCave &cave : local_entry_islands) {
                if (cave.owner_entry_text_offset != kernel.entry_text_offset ||
                    !compute_sopp_branch_simm16(site.text_offset, cave.text_offset)) {
                  continue;
                }
                found_reachable_island = true;
                LdsCheckTrapCandidate candidate{&site,
                                                nullptr,
                                                &cave,
                                                false,
                                                kernel.entry_text_offset,
                                                kernel.descriptor_file_offset,
                                                appended_cave_offset,
                                                *scratch,
                                                indirect_vcc_save_sgpr,
                                                *indirect_pc_sgpr,
                                                *indirect_scc_save_sgpr,
                                                requested_words,
                                                false,
                                                true,
                                                descriptor_growth,
                                                std::nullopt};
                appended_cave_candidates.push_back(candidate);
              }
              if (found_reachable_island)
                appended_cave_reachable = true;
              else
                ++indirect_island_unreachable_site_count;
            }
          }
        }
        if (appended_cave_reachable)
          ++append_cave_reachable_candidate_count;
      }
      if (observed_padding < padding_words)
        continue;
      LdsCheckTrapCandidate candidate{&site,
                                      nullptr,
                                      nullptr,
                                      false,
                                      kernel.entry_text_offset,
                                      kernel.descriptor_file_offset,
                                      0,
                                      *scratch,
                                      *vcc_save,
                                      std::nullopt,
                                      std::nullopt,
                                      requested_words,
                                      false,
                                      false,
                                      descriptor_growth,
                                      std::nullopt};
      inline_candidates.push_back(candidate);
    }
  }

  auto candidate_order = [](const LdsCheckTrapCandidate &lhs, const LdsCheckTrapCandidate &rhs) {
    const bool lhs_grows_descriptor = lhs.required_vgpr_allocation_count.has_value();
    const bool rhs_grows_descriptor = rhs.required_vgpr_allocation_count.has_value();
    if (lhs_grows_descriptor != rhs_grows_descriptor)
      return !lhs_grows_descriptor;
    if (lhs.site->file_offset != rhs.site->file_offset)
      return lhs.site->file_offset < rhs.site->file_offset;
    if (lhs.use_local_cave != rhs.use_local_cave)
      return !lhs.use_local_cave;
    if ((lhs.entry_island != nullptr) != (rhs.entry_island != nullptr))
      return lhs.entry_island == nullptr;
    const uint64_t lhs_cave = lhs.local_cave ? lhs.local_cave->file_offset : 0;
    const uint64_t rhs_cave = rhs.local_cave ? rhs.local_cave->file_offset : 0;
    if (lhs_cave != rhs_cave)
      return lhs_cave < rhs_cave;
    const uint64_t lhs_island = lhs.entry_island ? lhs.entry_island->file_offset : 0;
    const uint64_t rhs_island = rhs.entry_island ? rhs.entry_island->file_offset : 0;
    return lhs_island < rhs_island;
  };
  std::ranges::sort(inline_candidates, candidate_order);
  std::ranges::sort(local_cave_candidates, candidate_order);
  std::ranges::sort(appended_cave_candidates, candidate_order);
  std::ranges::sort(generated_island_seed_candidates, [](const auto &lhs, const auto &rhs) {
    return lhs.site->text_offset < rhs.site->text_offset;
  });

  const uint32_t max_patches = std::max<uint32_t>(options.max_patches, 1u);
  std::vector<LdsCheckTrapCandidate> selected_candidates;
  selected_candidates.reserve(max_patches);
  DbiPatchPlacementPlanner placement_planner(arch, appended_cave_offset);
  std::unordered_map<const ConSanLdsSite *, std::vector<uint64_t>> direct_body_relay_slots;
  auto try_select_candidate = [&](LdsCheckTrapCandidate candidate, bool allow_descriptor_growth) {
    if (selected_candidates.size() >= max_patches)
      return;
    if (!allow_descriptor_growth && candidate.required_vgpr_allocation_count)
      return;
    const ConSanLdsSite &site = *candidate.site;
    DbiPatchPlacementRequest request;
    request.anchor_offset = site.text_offset;
    request.original_size = site.size;
    request.body_size =
        (candidate.requested_words +
         (candidate.entry_island != nullptr ? kLdsCheckTrapIndirectJumpWords : 0u)) *
        sizeof(uint32_t);
    request.inline_capacity =
        !candidate.use_local_cave && !candidate.use_appended_cave ? request.body_size : 0u;
    request.allow_appended_cave = candidate.use_appended_cave;
    if (candidate.use_local_cave && candidate.local_cave)
      request.local_cave = DbiPatchLocalCave{
          candidate.local_cave->text_offset,
          static_cast<uint64_t>(candidate.local_cave->word_count) * sizeof(uint32_t)};
    std::optional<DbiPatchPlacement> placement;
    if (candidate.entry_island != nullptr) {
      if (!candidate.indirect_pc_sgpr || !candidate.indirect_scc_save_sgpr)
        return;
      DbiPatchPlacementPlanner tentative_planner = placement_planner;
      if (!candidate.entry_island_is_appended &&
          !tentative_planner.reserve_existing_range(candidate.entry_island->text_offset,
                                                    kLdsCheckTrapIndirectJumpWords *
                                                        sizeof(uint32_t))) {
        ++indirect_candidate_placement_failure_count;
        return;
      }
      placement =
          tentative_planner.plan_indirect_appended(site.text_offset, site.size, request.body_size);
      if (placement)
        placement_planner = std::move(tentative_planner);
      else
        ++indirect_candidate_placement_failure_count;
    } else if (auto relay_slot = direct_body_relay_slots.find(candidate.site);
               relay_slot != direct_body_relay_slots.end() && candidate.use_appended_cave) {
      DbiPatchPlacementPlanner tentative_planner = placement_planner;
      placement = tentative_planner.plan(request);
      if (!placement || relay_slot->second.empty() ||
          tentative_planner.appended_end() != relay_slot->second.front() ||
          !tentative_planner.reserve_appended_prefix(relay_slot->second.size() *
                                                     sizeof(uint32_t))) {
        placement.reset();
      } else {
        placement_planner = std::move(tentative_planner);
      }
    } else {
      placement = placement_planner.plan(request);
    }
    if (!placement)
      return;
    if ((candidate.use_local_cave && placement->kind != DbiPatchPlacementKind::LocalCave) ||
        (candidate.use_appended_cave && placement->kind != DbiPatchPlacementKind::AppendedCave) ||
        (!candidate.use_local_cave && !candidate.use_appended_cave &&
         placement->kind != DbiPatchPlacementKind::Inline))
      return;
    candidate.placement = placement;

    if (candidate.use_local_cave) {
      if (candidate.local_cave == nullptr)
        return;
    }
    selected_candidates.push_back(candidate);
  };

  for (const LdsCheckTrapCandidate &candidate : inline_candidates)
    try_select_candidate(candidate, false);
  for (const LdsCheckTrapCandidate &candidate : inline_candidates)
    try_select_candidate(candidate, true);

  const auto site_is_selected = [&](const ConSanLdsSite *site) {
    return std::ranges::any_of(selected_candidates,
                               [&](const auto &selected) { return selected.site == site; });
  };
  bool needs_generated_island_bank = false;
  {
    DbiPatchPlacementPlanner simulated_planner = placement_planner;
    size_t simulated_selected_count = selected_candidates.size();
    std::unordered_set<const ConSanLdsSite *> simulated_selected_sites;
    for (const LdsCheckTrapCandidate &selected : selected_candidates)
      simulated_selected_sites.insert(selected.site);

    // Model the density-maximizing local-island phase before deciding whether
    // the subsequent direct-appended sequence will drift out of SOPP range.
    // Use a tentative planner so a failed island/body pair reserves neither
    // half of the placement.
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      if (candidate.entry_island == nullptr || simulated_selected_sites.contains(candidate.site) ||
          simulated_selected_count >= max_patches)
        continue;
      DbiPatchPlacementPlanner tentative_planner = simulated_planner;
      if (!tentative_planner.reserve_existing_range(candidate.entry_island->text_offset,
                                                    kLdsCheckTrapIndirectJumpWords *
                                                        sizeof(uint32_t)) ||
          !tentative_planner.plan_indirect_appended(
              candidate.site->text_offset, candidate.site->size,
              (candidate.requested_words + kLdsCheckTrapIndirectJumpWords) * sizeof(uint32_t)))
        continue;
      simulated_planner = std::move(tentative_planner);
      simulated_selected_sites.insert(candidate.site);
      ++simulated_selected_count;
    }
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      if (candidate.entry_island != nullptr || simulated_selected_sites.contains(candidate.site) ||
          simulated_selected_count >= max_patches)
        continue;
      DbiPatchPlacementRequest request;
      request.anchor_offset = candidate.site->text_offset;
      request.original_size = candidate.site->size;
      request.body_size = candidate.requested_words * sizeof(uint32_t);
      request.allow_appended_cave = true;
      if (!simulated_planner.plan(request)) {
        needs_generated_island_bank = true;
        break;
      }
      simulated_selected_sites.insert(candidate.site);
      ++simulated_selected_count;
    }
  }
  std::vector<LocalNopCave> generated_entry_islands;
  std::vector<LdsCheckTrapCandidate> generated_island_candidates;
  std::vector<ScWideCompositeBranchRelayDonor> selected_wide_composite_relay_donors;
  std::vector<ScRelayReservoir> selected_relay_reservoirs;
  generated_entry_islands.reserve(generated_island_seed_candidates.size());
  generated_island_candidates.reserve(generated_island_seed_candidates.size());
  uint64_t generated_island_bank_offset = appended_cave_offset;
  uint64_t generated_island_cursor = generated_island_bank_offset;
  uint64_t generated_relay_bank_offset = appended_cave_offset;
  size_t generated_relay_bank_word_count = 0;
  std::unordered_set<const ConSanLdsSite *> relay_routed_sites;
  bool use_relay_island_bank = local_nop_caves.empty() &&
                               indirect_island_unreachable_site_count != 0 &&
                               max_patches >= supported_candidate_count;
  if (use_relay_island_bank) {
    std::vector<LdsCheckTrapCandidate> relay_owner_seeds;
    relay_owner_seeds.reserve(generated_island_seed_candidates.size());
    for (const LdsCheckTrapCandidate &candidate : generated_island_seed_candidates) {
      if (!site_is_selected(candidate.site))
        relay_owner_seeds.push_back(candidate);
    }
    std::unordered_set<const ConSanLdsSite *> direct_sites;
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      if (candidate.entry_island == nullptr)
        direct_sites.insert(candidate.site);
    }
    for (const LdsCheckTrapCandidate &candidate : relay_owner_seeds) {
      if (!direct_sites.contains(candidate.site))
        relay_routed_sites.insert(candidate.site);
    }

    // Place wide-host bodies at the pristine append cursor, before the much
    // larger direct SC bodies consume their SOPP reach. Each host keeps an
    // intermediate eight-byte instruction's original bytes and local return
    // in its otherwise skipped payload, creating three relay vertices while
    // preserving both normal execution paths.
    DbiPatchPlacementPlanner relay_prefix_planner = placement_planner;
    std::unordered_set<uint64_t> prefixed_anchor_offsets;
    std::unordered_set<uint64_t> prefixed_host_offsets;
    const size_t wide_prefix_limit = relay_owner_seeds.size();
    for (auto host_it = relay_wide_host_candidates.rbegin();
         host_it != relay_wide_host_candidates.rend(); ++host_it) {
      if (selected_wide_composite_relay_donors.size() >= wide_prefix_limit)
        break;
      const ScWideRelayHostCandidate &host = *host_it;
      if (prefixed_host_offsets.contains(host.host_offset))
        continue;
      DbiPatchPlacementPlanner host_planner = relay_prefix_planner;
      std::optional<DbiPatchPlacement> host_placement;
      uint32_t selected_host_words = 0;
      for (auto prefix_it = host.prefix_word_counts.rbegin();
           prefix_it != host.prefix_word_counts.rend(); ++prefix_it) {
        DbiPatchPlacementPlanner tentative_host_planner = relay_prefix_planner;
        DbiPatchPlacementRequest host_request;
        host_request.anchor_offset = host.host_offset;
        host_request.original_size = *prefix_it * sizeof(uint32_t);
        host_request.body_size = *prefix_it * sizeof(uint32_t);
        host_request.allow_appended_cave = true;
        auto placement = tentative_host_planner.plan(host_request);
        if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave)
          continue;
        host_planner = std::move(tentative_host_planner);
        host_placement = *placement;
        selected_host_words = *prefix_it;
        break;
      }
      if (!host_placement)
        continue;
      const uint64_t target_distance =
          selected_wide_composite_relay_donors.empty() ? 110000u : 65536u;
      const uint64_t target_anchor =
          host.host_offset > target_distance ? host.host_offset - target_distance : 0u;
      auto anchor_it = std::ranges::lower_bound(relay_anchor_candidates, target_anchor, {},
                                                &ScRelayAnchorCandidate::anchor_offset);
      for (; anchor_it != relay_anchor_candidates.end() &&
             anchor_it->anchor_offset < host.host_offset;
           ++anchor_it) {
        const ScRelayAnchorCandidate &anchor = *anchor_it;
        if (prefixed_anchor_offsets.contains(anchor.anchor_offset) ||
            !compute_sopp_branch_simm16(anchor.anchor_offset,
                                        host.host_offset + sizeof(uint32_t)) ||
            !compute_sopp_branch_simm16(host.host_offset + 3u * sizeof(uint32_t),
                                        anchor.anchor_offset + 2u * sizeof(uint32_t)))
          continue;
        DbiPatchPlacementPlanner tentative_planner = host_planner;
        if (!tentative_planner.reserve_existing_range(anchor.anchor_offset, 2u * sizeof(uint32_t)))
          continue;
        ScWideCompositeBranchRelayDonor wide;
        wide.anchor_offset = anchor.anchor_offset;
        wide.anchor_original_words = anchor.anchor_original_words;
        wide.host_offset = host.host_offset;
        wide.host_original_words.assign(host.host_original_words.begin(),
                                        host.host_original_words.begin() + selected_host_words);
        wide.host_placement = *host_placement;
        selected_wide_composite_relay_donors.push_back(wide);
        prefixed_anchor_offsets.insert(anchor.anchor_offset);
        prefixed_host_offsets.insert(host.host_offset);
        relay_prefix_planner = std::move(tentative_planner);
        break;
      }
    }

    const auto replay_support_prefix =
        [&](std::vector<ScWideCompositeBranchRelayDonor> &wide_donors,
            std::string *error_out) -> std::optional<DbiPatchPlacementPlanner> {
      DbiPatchPlacementPlanner planner = placement_planner;
      for (ScWideCompositeBranchRelayDonor &donor : wide_donors) {
        const uint32_t host_size =
            static_cast<uint32_t>(donor.host_original_words.size() * sizeof(uint32_t));
        DbiPatchPlacementRequest request;
        request.anchor_offset = donor.host_offset;
        request.original_size = host_size;
        request.body_size = host_size;
        request.allow_appended_cave = true;
        auto placement = planner.plan(request, error_out);
        if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave ||
            !planner.reserve_existing_range(donor.anchor_offset, 2u * sizeof(uint32_t), error_out))
          return std::nullopt;
        donor.host_placement = *placement;
      }
      return planner;
    };

    // Every support prefix changes the later direct bodies and island bank.
    // Rebuild that complete layout for every trial and let exact max flow,
    // rather than geometric capacity guesses, decide whether a reservoir is
    // useful.
    struct RelayLayoutAttempt {
      std::unordered_set<const ConSanLdsSite *> routed_sites;
      std::unordered_map<const ConSanLdsSite *, std::vector<uint64_t>> body_relay_slots;
      std::vector<LdsCheckTrapCandidate> relay_seeds;
      std::vector<uint64_t> source_offsets;
      uint64_t relay_bank_offset = 0;
      size_t relay_bank_word_count = 0;
      uint64_t island_bank_offset = 0;
      SoppBranchRelayPlan relay_plan;
    };
    const std::unordered_set<const ConSanLdsSite *> required_relay_routed_sites =
        relay_routed_sites;
    const auto evaluate_relay_layout =
        [&](const DbiPatchPlacementPlanner &prefix_planner, size_t relay_bank_word_count,
            std::string *error_out) -> std::optional<RelayLayoutAttempt> {
      std::unordered_set<const ConSanLdsSite *> routed_sites = required_relay_routed_sites;
      DbiPatchPlacementPlanner direct_layout = prefix_planner;
      std::unordered_map<const ConSanLdsSite *, std::vector<uint64_t>> body_relay_slots;
      for (;;) {
        DbiPatchPlacementPlanner direct_simulation = prefix_planner;
        std::vector<const ConSanLdsSite *> newly_routed;
        std::unordered_map<const ConSanLdsSite *, std::vector<uint64_t>> simulated_slots;
        for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
          if (candidate.entry_island != nullptr || routed_sites.contains(candidate.site) ||
              site_is_selected(candidate.site))
            continue;
          DbiPatchPlacementRequest request;
          request.anchor_offset = candidate.site->text_offset;
          request.original_size = candidate.site->size;
          request.body_size = candidate.requested_words * sizeof(uint32_t);
          request.allow_appended_cave = true;
          if (!direct_simulation.plan(request)) {
            newly_routed.push_back(candidate.site);
            continue;
          }
          const uint64_t relay_slot_begin = direct_simulation.appended_end();
          if (!direct_simulation.reserve_appended_prefix(sizeof(uint32_t))) {
            if (error_out != nullptr)
              *error_out = "could not reserve a direct-body relay slot";
            return std::nullopt;
          }
          std::vector<uint64_t> relay_slots;
          relay_slots.push_back(relay_slot_begin);
          simulated_slots.emplace(candidate.site, std::move(relay_slots));
        }
        if (newly_routed.empty()) {
          direct_layout = std::move(direct_simulation);
          body_relay_slots = std::move(simulated_slots);
          break;
        }
        for (const ConSanLdsSite *site : newly_routed)
          routed_sites.insert(site);
      }

      std::vector<LdsCheckTrapCandidate> relay_seeds;
      relay_seeds.reserve(routed_sites.size());
      for (const LdsCheckTrapCandidate &candidate : relay_owner_seeds) {
        if (routed_sites.contains(candidate.site))
          relay_seeds.push_back(candidate);
      }
      std::vector<uint64_t> source_offsets;
      source_offsets.reserve(relay_seeds.size());
      for (const LdsCheckTrapCandidate &candidate : relay_seeds)
        source_offsets.push_back(candidate.site->text_offset);

      std::vector<uint64_t> relay_offsets;
      size_t reservoir_slot_count = 0;
      for (const ScRelayReservoir &reservoir : selected_relay_reservoirs)
        reservoir_slot_count +=
            sc_relay_reservoir_slot_offsets(reservoir.anchor_offset, reservoir.original_size)
                .size();
      relay_offsets.reserve(relay_owner_seeds.size() + body_relay_slots.size() +
                            3u * selected_wide_composite_relay_donors.size() +
                            reservoir_slot_count);
      for (const LdsCheckTrapCandidate &candidate : relay_owner_seeds)
        relay_offsets.push_back(candidate.site->text_offset + sizeof(uint32_t));
      for (const auto &[site, relay_site_offsets] : body_relay_slots) {
        (void)site;
        relay_offsets.insert(relay_offsets.end(), relay_site_offsets.begin(),
                             relay_site_offsets.end());
      }
      for (const ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
        relay_offsets.push_back(donor.anchor_offset + sizeof(uint32_t));
        for (size_t word = 4u; word < donor.host_original_words.size(); ++word)
          relay_offsets.push_back(donor.host_offset + word * sizeof(uint32_t));
      }
      for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
        const std::vector<uint64_t> slots =
            sc_relay_reservoir_slot_offsets(reservoir.anchor_offset, reservoir.original_size);
        relay_offsets.insert(relay_offsets.end(), slots.begin(), slots.end());
      }
      const uint64_t relay_bank_offset = direct_layout.appended_end();
      if (!direct_layout.reserve_appended_prefix(relay_bank_word_count * sizeof(uint32_t),
                                                 error_out))
        return std::nullopt;
      for (size_t index = 0; index < relay_bank_word_count; ++index)
        relay_offsets.push_back(relay_bank_offset + index * sizeof(uint32_t));
      std::vector<uint64_t> island_offsets;
      island_offsets.reserve(relay_seeds.size());
      uint64_t cursor = direct_layout.appended_end();
      const uint64_t island_bank_offset = cursor;
      for (size_t index = 0; index < relay_seeds.size(); ++index) {
        island_offsets.push_back(cursor);
        cursor += kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t);
      }
      const uint64_t island_bank_bytes = cursor - island_bank_offset;
      if (!direct_layout.reserve_appended_prefix(island_bank_bytes, error_out))
        return std::nullopt;
      for (ScRelayReservoir &reservoir : selected_relay_reservoirs) {
        auto placement = direct_layout.plan_indirect_appended(
            reservoir.anchor_offset, reservoir.original_size,
            reservoir.original_size + kScRelayAppendedOverheadWords * sizeof(uint32_t), error_out);
        if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave)
          return std::nullopt;
        placement->return_target = reservoir.anchor_offset + reservoir.original_size -
                                   kScRelayTailRestoreWords * sizeof(uint32_t);
        reservoir.placement = *placement;
      }
      auto plan =
          plan_forward_sopp_branch_relays(source_offsets, relay_offsets, island_offsets, error_out);
      if (!plan)
        return std::nullopt;
      return RelayLayoutAttempt{std::move(routed_sites), std::move(body_relay_slots),
                                std::move(relay_seeds),  std::move(source_offsets),
                                relay_bank_offset,       relay_bank_word_count,
                                island_bank_offset,      std::move(*plan)};
    };

    std::string relay_error;
    size_t relay_bank_word_count = 0;
    auto relay_attempt =
        evaluate_relay_layout(relay_prefix_planner, relay_bank_word_count, &relay_error);
    if (!relay_attempt) {
      result.errors.emplace_back("ConSan LDS check/trap relay planning failed: " + relay_error);
      return;
    }

    // A cut in pristine text needs more pristine vertices. Relocate a
    // straight-line wave32 range with an indirect entry/return pair. Each
    // endpoint uses VCC as the PC pair, saves VCC_LO in one endpoint-dead
    // SGPR, and uses architecturally inactive EXEC_HI to preserve SCC. Expose
    // only the skipped interior words as relay
    // capacity. Exact replay and max flow remain the acceptance test for every
    // trial.
    size_t last_ranked_reservoir_count = 0;
    size_t last_exact_reservoir_trial_count = 0;
    while (!relay_attempt->relay_plan.rejected_source_indices.empty()) {
      struct RankedReservoir {
        const ScRelayReservoirCandidate *candidate = nullptr;
        size_t cut_words = 0;
      };
      std::vector<RankedReservoir> ranked;
      constexpr uint64_t kMaximumForwardSoppBranchBytes = 131072u;
      uint64_t cut_neighborhood_begin = 0u;
      uint64_t cut_neighborhood_end = 0u;
      if (!relay_attempt->relay_plan.min_cut_relay_offsets.empty()) {
        const auto [minimum_cut_relay, maximum_cut_relay] =
            std::ranges::minmax(relay_attempt->relay_plan.min_cut_relay_offsets);
        cut_neighborhood_begin = minimum_cut_relay > kMaximumForwardSoppBranchBytes
                                     ? minimum_cut_relay - kMaximumForwardSoppBranchBytes
                                     : 0u;
        cut_neighborhood_end = maximum_cut_relay > std::numeric_limits<uint64_t>::max() -
                                                       kMaximumForwardSoppBranchBytes
                                   ? std::numeric_limits<uint64_t>::max()
                                   : maximum_cut_relay + kMaximumForwardSoppBranchBytes;
      } else {
        // A graph with no existing relay on its source-to-island frontier has
        // no saturated relay vertex to report. Seed that empty frontier from
        // the rejected sources through the reserved island bank; exact replay
        // below still decides whether any ranked reservoir adds real flow.
        cut_neighborhood_begin = std::numeric_limits<uint64_t>::max();
        for (size_t rejected : relay_attempt->relay_plan.rejected_source_indices) {
          if (rejected >= relay_attempt->source_offsets.size())
            continue;
          cut_neighborhood_begin =
              std::min(cut_neighborhood_begin, relay_attempt->source_offsets[rejected]);
        }
        if (cut_neighborhood_begin == std::numeric_limits<uint64_t>::max())
          break;
        cut_neighborhood_end =
            relay_attempt->island_bank_offset +
            relay_attempt->relay_seeds.size() * kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t);
      }
      for (const ScRelayReservoirCandidate &candidate : relay_reservoir_candidates) {
        const bool already_selected =
            std::ranges::any_of(selected_relay_reservoirs, [&](const auto &selected) {
              return candidate.anchor_offset < selected.anchor_offset + selected.original_size &&
                     selected.anchor_offset < candidate.anchor_offset + candidate.original_size;
            });
        if (already_selected)
          continue;
        const uint64_t payload_begin =
            candidate.anchor_offset + kScRelayEntryWords * sizeof(uint32_t);
        const uint64_t payload_end = candidate.anchor_offset + candidate.original_size -
                                     kScRelayTailRestoreWords * sizeof(uint32_t);
        const uint64_t overlap_begin = std::max(payload_begin, cut_neighborhood_begin);
        const uint64_t overlap_end = std::min(payload_end, cut_neighborhood_end);
        const size_t cut_words =
            overlap_end > overlap_begin
                ? static_cast<size_t>((overlap_end - overlap_begin) / sizeof(uint32_t))
                : 0u;
        if (cut_words != 0)
          ranked.push_back({&candidate, cut_words});
      }
      std::ranges::sort(ranked, [](const auto &lhs, const auto &rhs) {
        if (lhs.cut_words != rhs.cut_words)
          return lhs.cut_words > rhs.cut_words;
        if (lhs.candidate->original_size != rhs.candidate->original_size)
          return lhs.candidate->original_size > rhs.candidate->original_size;
        return lhs.candidate->anchor_offset < rhs.candidate->anchor_offset;
      });
      last_ranked_reservoir_count = ranked.size();
      const size_t old_rejected = relay_attempt->relay_plan.rejected_source_indices.size();
      std::optional<RelayLayoutAttempt> best_attempt;
      std::optional<ScRelayReservoir> best_reservoir;
      DbiPatchPlacementPlanner best_prefix = relay_prefix_planner;
      const auto append_candidate_reservoir = [&](const ScRelayReservoirCandidate &candidate) {
        selected_relay_reservoirs.push_back({candidate.anchor_offset,
                                             candidate.original_size,
                                             candidate.pc_sgpr_pair,
                                             candidate.saved_vcc_sgpr,
                                             candidate.return_pc_sgpr_pair,
                                             candidate.return_saved_vcc_sgpr,
                                             candidate.required_sgpr_count,
                                             candidate.owner_descriptor_file_offsets,
                                             {}});
      };
      constexpr size_t kMaxExactReservoirTrialsPerIteration = 8u;
      size_t exact_reservoir_trials = 0;
      for (const RankedReservoir &ranked_candidate : ranked) {
        const ScRelayReservoirCandidate &candidate = *ranked_candidate.candidate;
        append_candidate_reservoir(candidate);
        std::string trial_error;
        auto trial_prefix =
            replay_support_prefix(selected_wide_composite_relay_donors, &trial_error);
        if (!trial_prefix) {
          selected_relay_reservoirs.pop_back();
          continue;
        }
        ++exact_reservoir_trials;
        last_exact_reservoir_trial_count = exact_reservoir_trials;
        auto trial = evaluate_relay_layout(*trial_prefix, 0u, &trial_error);
        ScRelayReservoir trial_reservoir = selected_relay_reservoirs.back();
        selected_relay_reservoirs.pop_back();
        if (!trial || trial->relay_plan.rejected_source_indices.size() >= old_rejected) {
          if (exact_reservoir_trials >= kMaxExactReservoirTrialsPerIteration)
            break;
          else
            continue;
        }
        if (!best_attempt || trial->relay_plan.rejected_source_indices.size() <
                                 best_attempt->relay_plan.rejected_source_indices.size()) {
          best_attempt = std::move(trial);
          best_reservoir = std::move(trial_reservoir);
          best_prefix = std::move(*trial_prefix);
          if (best_attempt->relay_plan.rejected_source_indices.empty())
            break;
        }
        if (exact_reservoir_trials >= kMaxExactReservoirTrialsPerIteration)
          break;
      }
      bool accepted_batch = false;
      if (!best_attempt) {
        const size_t old_reservoir_count = selected_relay_reservoirs.size();
        size_t batch_size = 0;
        size_t next_trial_size = 2u;
        constexpr size_t kMaxBatchReservoirs = 256u;
        for (const RankedReservoir &ranked_candidate : ranked) {
          const ScRelayReservoirCandidate &candidate = *ranked_candidate.candidate;
          const bool overlaps_batch =
              std::ranges::any_of(selected_relay_reservoirs, [&](const auto &selected) {
                return candidate.anchor_offset < selected.anchor_offset + selected.original_size &&
                       selected.anchor_offset < candidate.anchor_offset + candidate.original_size;
              });
          if (overlaps_batch)
            continue;
          append_candidate_reservoir(candidate);
          ++batch_size;
          if (batch_size < next_trial_size && batch_size < kMaxBatchReservoirs)
            continue;
          std::string trial_error;
          auto trial_prefix =
              replay_support_prefix(selected_wide_composite_relay_donors, &trial_error);
          ++exact_reservoir_trials;
          last_exact_reservoir_trial_count = exact_reservoir_trials;
          auto trial =
              trial_prefix ? evaluate_relay_layout(*trial_prefix, 0u, &trial_error) : std::nullopt;
          if (trial && trial->relay_plan.rejected_source_indices.size() < old_rejected) {
            relay_prefix_planner = std::move(*trial_prefix);
            relay_attempt = std::move(trial);
            accepted_batch = true;
            break;
          }
          if (batch_size >= kMaxBatchReservoirs)
            break;
          next_trial_size = std::min(kMaxBatchReservoirs, next_trial_size * 2u);
        }
        if (!accepted_batch)
          selected_relay_reservoirs.resize(old_reservoir_count);
      }
      if (accepted_batch)
        continue;
      if (!best_attempt || !best_reservoir)
        break;
      selected_relay_reservoirs.push_back(*best_reservoir);
      relay_prefix_planner = std::move(best_prefix);
      relay_attempt = std::move(best_attempt);
    }

    if (relay_attempt->relay_plan.rejected_source_indices.empty()) {
      std::unordered_set<uint64_t> used_support_slots;
      for (const SoppBranchRelayRoute &route : relay_attempt->relay_plan.routes) {
        for (uint64_t offset : route.relay_offsets)
          used_support_slots.insert(offset);
      }
      const auto wide_is_used = [&](const ScWideCompositeBranchRelayDonor &donor) {
        if (used_support_slots.contains(donor.anchor_offset + sizeof(uint32_t)))
          return true;
        for (size_t word = 4u; word < donor.host_original_words.size(); ++word) {
          if (used_support_slots.contains(donor.host_offset + word * sizeof(uint32_t)))
            return true;
        }
        return false;
      };
      const auto reservoir_is_used = [&](const ScRelayReservoir &reservoir) {
        for (uint64_t offset :
             sc_relay_reservoir_slot_offsets(reservoir.anchor_offset, reservoir.original_size)) {
          if (used_support_slots.contains(offset))
            return true;
        }
        return false;
      };
      std::vector<ScWideCompositeBranchRelayDonor> pruned_wide;
      std::ranges::copy_if(selected_wide_composite_relay_donors, std::back_inserter(pruned_wide),
                           wide_is_used);
      std::vector<ScRelayReservoir> pruned_reservoirs;
      std::ranges::copy_if(selected_relay_reservoirs, std::back_inserter(pruned_reservoirs),
                           reservoir_is_used);
      if (pruned_wide.size() != selected_wide_composite_relay_donors.size() ||
          pruned_reservoirs.size() != selected_relay_reservoirs.size()) {
        auto original_wide = selected_wide_composite_relay_donors;
        auto original_reservoirs = selected_relay_reservoirs;
        selected_wide_composite_relay_donors = std::move(pruned_wide);
        selected_relay_reservoirs = std::move(pruned_reservoirs);
        std::string prune_error;
        auto pruned_prefix =
            replay_support_prefix(selected_wide_composite_relay_donors, &prune_error);
        auto pruned_attempt =
            pruned_prefix
                ? evaluate_relay_layout(*pruned_prefix, relay_bank_word_count, &prune_error)
                : std::nullopt;
        if (pruned_attempt && pruned_attempt->relay_plan.rejected_source_indices.empty()) {
          relay_prefix_planner = std::move(*pruned_prefix);
          relay_attempt = std::move(pruned_attempt);
        } else {
          selected_wide_composite_relay_donors = std::move(original_wide);
          selected_relay_reservoirs = std::move(original_reservoirs);
        }
      }
    }

    relay_routed_sites = relay_attempt->routed_sites;
    direct_body_relay_slots = relay_attempt->body_relay_slots;
    std::vector<LdsCheckTrapCandidate> relay_seeds = relay_attempt->relay_seeds;
    std::vector<uint64_t> source_offsets = relay_attempt->source_offsets;
    generated_relay_bank_offset = relay_attempt->relay_bank_offset;
    generated_relay_bank_word_count = relay_attempt->relay_bank_word_count;
    generated_island_bank_offset = relay_attempt->island_bank_offset;
    auto relay_plan = std::move(relay_attempt->relay_plan);

    if (!relay_plan.rejected_source_indices.empty()) {
      uint64_t min_rejected_source = std::numeric_limits<uint64_t>::max();
      uint64_t max_rejected_source = 0;
      uint64_t min_cut_from = std::numeric_limits<uint64_t>::max();
      uint64_t max_cut_from = 0;
      uint64_t min_cut_to = std::numeric_limits<uint64_t>::max();
      uint64_t max_cut_to = 0;
      uint64_t min_cut_relay = std::numeric_limits<uint64_t>::max();
      uint64_t max_cut_relay = 0;
      size_t min_cut_direct_relay_count = 0;
      size_t min_cut_owner_relay_count = 0;
      size_t min_cut_wide_relay_count = 0;
      size_t selected_wide_relay_slot_count = 0;
      size_t max_selected_wide_host_words = 0;
      for (const ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
        selected_wide_relay_slot_count += donor.host_original_words.size() - 3u;
        max_selected_wide_host_words =
            std::max(max_selected_wide_host_words, donor.host_original_words.size());
      }
      for (size_t rejected_index : relay_plan.rejected_source_indices) {
        min_rejected_source = std::min(min_rejected_source, source_offsets[rejected_index]);
        max_rejected_source = std::max(max_rejected_source, source_offsets[rejected_index]);
      }
      for (const auto [from, to] : relay_plan.min_cut_hops) {
        min_cut_from = std::min(min_cut_from, from);
        max_cut_from = std::max(max_cut_from, from);
        min_cut_to = std::min(min_cut_to, to);
        max_cut_to = std::max(max_cut_to, to);
      }
      for (uint64_t relay : relay_plan.min_cut_relay_offsets) {
        min_cut_relay = std::min(min_cut_relay, relay);
        max_cut_relay = std::max(max_cut_relay, relay);
        for (const auto &[site, offsets] : direct_body_relay_slots) {
          (void)site;
          min_cut_direct_relay_count += std::ranges::find(offsets, relay) != offsets.end();
        }
        min_cut_owner_relay_count +=
            std::ranges::any_of(relay_owner_seeds, [&](const LdsCheckTrapCandidate &candidate) {
              return candidate.site->text_offset + sizeof(uint32_t) == relay;
            });
        min_cut_wide_relay_count +=
            std::ranges::any_of(selected_wide_composite_relay_donors,
                                [&](const ScWideCompositeBranchRelayDonor &donor) {
                                  return donor.anchor_offset + sizeof(uint32_t) == relay ||
                                         donor.host_offset + 4u * sizeof(uint32_t) == relay ||
                                         donor.host_offset + 5u * sizeof(uint32_t) == relay;
                                });
      }
      result.warnings.emplace_back(
          "ConSan LDS check/trap relay bank could not close " +
          std::to_string(relay_plan.rejected_source_indices.size()) +
          " source route(s); retaining ordinary placement; "
          "relay_anchor_candidates=" +
          std::to_string(relay_anchor_candidates.size()) + " wide_host_candidates=" +
          std::to_string(relay_wide_host_candidates.size()) + " selected_wide_composite_donors=" +
          std::to_string(selected_wide_composite_relay_donors.size()) + " rejected_source_range=" +
          std::to_string(min_rejected_source) + "-" + std::to_string(max_rejected_source) +
          " relay_reservoir_candidates=" + std::to_string(relay_reservoir_candidates.size()) +
          " selected_relay_reservoirs=" + std::to_string(selected_relay_reservoirs.size()) +
          " ranked_relay_reservoirs=" + std::to_string(last_ranked_reservoir_count) +
          " exact_relay_reservoir_trials=" + std::to_string(last_exact_reservoir_trial_count) +
          " generated_relay_bank_words=" + std::to_string(generated_relay_bank_word_count) +
          " selected_wide_relay_slots=" + std::to_string(selected_wide_relay_slot_count) +
          " max_selected_wide_host_words=" + std::to_string(max_selected_wide_host_words) +
          " min_cut_hops=" + std::to_string(relay_plan.min_cut_hops.size()) +
          " min_cut_from_range=" + std::to_string(min_cut_from) + "-" +
          std::to_string(max_cut_from) + " min_cut_to_range=" + std::to_string(min_cut_to) + "-" +
          std::to_string(max_cut_to) +
          " min_cut_relays=" + std::to_string(relay_plan.min_cut_relay_offsets.size()) +
          " min_cut_direct_relays=" + std::to_string(min_cut_direct_relay_count) +
          " min_cut_owner_relays=" + std::to_string(min_cut_owner_relay_count) +
          " min_cut_wide_relays=" + std::to_string(min_cut_wide_relay_count) +
          " min_cut_relay_range=" + std::to_string(min_cut_relay) + "-" +
          std::to_string(max_cut_relay));
      generated_entry_islands.clear();
      selected_wide_composite_relay_donors.clear();
      selected_relay_reservoirs.clear();
      generated_island_bank_offset = appended_cave_offset;
      generated_island_cursor = generated_island_bank_offset;
      generated_relay_bank_offset = appended_cave_offset;
      generated_relay_bank_word_count = 0;
      use_relay_island_bank = false;
    } else {
      generated_island_cursor = generated_island_bank_offset;
      for (const LdsCheckTrapCandidate &candidate : relay_seeds) {
        generated_entry_islands.push_back(LocalNopCave{
            .text_offset = generated_island_cursor,
            .file_offset = 0,
            .word_count = static_cast<uint32_t>(kLdsCheckTrapIndirectJumpWords),
            .owner_entry_text_offset = candidate.kernel_entry_text_offset,
        });
        generated_island_cursor += kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t);
      }
      for (const SoppBranchRelayRoute &route : relay_plan.routes) {
        LdsCheckTrapCandidate candidate = relay_seeds[route.source_index];
        const uint64_t island_index = (route.island_offset - generated_island_bank_offset) /
                                      (kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t));
        if (island_index >= generated_entry_islands.size() ||
            generated_entry_islands[island_index].text_offset != route.island_offset) {
          result.errors.emplace_back(
              "ConSan LDS check/trap relay planner returned an unknown island");
          return;
        }
        candidate.entry_island = &generated_entry_islands[island_index];
        candidate.entry_relay_offsets = route.relay_offsets;
        generated_island_candidates.push_back(std::move(candidate));
      }
      indirect_island_unreachable_site_count = 0;
    }
  }
  if (!use_relay_island_bank && needs_generated_island_bank) {
    for (LdsCheckTrapCandidate candidate : generated_island_seed_candidates) {
      if (site_is_selected(candidate.site) ||
          !compute_sopp_branch_simm16(candidate.site->text_offset, generated_island_cursor))
        continue;
      generated_entry_islands.push_back(LocalNopCave{
          .text_offset = generated_island_cursor,
          .file_offset = 0,
          .word_count = static_cast<uint32_t>(kLdsCheckTrapIndirectJumpWords),
          .owner_entry_text_offset = candidate.kernel_entry_text_offset,
      });
      candidate.entry_island = &generated_entry_islands.back();
      generated_island_candidates.push_back(candidate);
      generated_island_cursor += kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t);
    }
  }
  const uint64_t generated_island_bank_bytes =
      generated_entry_islands.size() * kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t);
  if (!use_relay_island_bank &&
      !placement_planner.reserve_appended_prefix(generated_island_bank_bytes)) {
    result.errors.emplace_back(
        "ConSan LDS check/trap proof could not reserve its entry-island bank");
    return;
  }
  std::ranges::sort(generated_island_candidates, candidate_order);

  // A long local NOP cave can hold one complete check body or several
  // seven-word entry islands whose bodies live in appended text. Prefer the
  // latter representation before consuming any cave with a full body: it is
  // the only choice that can maximize covered sites in large generated
  // objects.
  for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
    if (candidate.entry_island != nullptr)
      try_select_candidate(candidate, true);
  }
  for (const LdsCheckTrapCandidate &candidate : local_cave_candidates)
    try_select_candidate(candidate, false);
  for (const LdsCheckTrapCandidate &candidate : local_cave_candidates)
    try_select_candidate(candidate, true);
  if (use_relay_island_bank) {
    for (ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
      const uint32_t host_size =
          static_cast<uint32_t>(donor.host_original_words.size() * sizeof(uint32_t));
      DbiPatchPlacementRequest request;
      request.anchor_offset = donor.host_offset;
      request.original_size = host_size;
      request.body_size = host_size;
      request.allow_appended_cave = true;
      auto placement = placement_planner.plan(request);
      if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave ||
          placement->body_offset != donor.host_placement.body_offset ||
          placement->return_target != donor.host_placement.return_target ||
          !placement_planner.reserve_existing_range(donor.anchor_offset, 2u * sizeof(uint32_t))) {
        result.errors.emplace_back(
            "ConSan LDS check/trap direct sequence diverged from its wide composite relay plan");
        return;
      }
      donor.host_placement = *placement;
    }
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      if (candidate.entry_island == nullptr && !relay_routed_sites.contains(candidate.site))
        try_select_candidate(candidate, true);
    }
    if (placement_planner.appended_end() != generated_relay_bank_offset ||
        !placement_planner.reserve_appended_prefix(generated_relay_bank_word_count *
                                                   sizeof(uint32_t)) ||
        placement_planner.appended_end() != generated_island_bank_offset ||
        !placement_planner.reserve_appended_prefix(generated_island_bank_bytes)) {
      result.errors.emplace_back(
          "ConSan LDS check/trap direct sequence diverged from its relay-bank plan");
      return;
    }
    for (ScRelayReservoir &reservoir : selected_relay_reservoirs) {
      auto placement = placement_planner.plan_indirect_appended(
          reservoir.anchor_offset, reservoir.original_size,
          reservoir.original_size + kScRelayAppendedOverheadWords * sizeof(uint32_t));
      if (placement)
        placement->return_target = reservoir.anchor_offset + reservoir.original_size -
                                   kScRelayTailRestoreWords * sizeof(uint32_t);
      if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave ||
          placement->body_offset != reservoir.placement.body_offset ||
          placement->body_size != reservoir.placement.body_size ||
          placement->return_branch_offset != reservoir.placement.return_branch_offset ||
          placement->return_target != reservoir.placement.return_target) {
        result.errors.emplace_back(
            "ConSan LDS check/trap direct sequence diverged from its relay-reservoir plan");
        return;
      }
      reservoir.placement = *placement;
    }
    for (const LdsCheckTrapCandidate &candidate : generated_island_candidates)
      try_select_candidate(candidate, true);
  } else {
    for (const LdsCheckTrapCandidate &candidate : generated_island_candidates)
      try_select_candidate(candidate, true);
    for (const LdsCheckTrapCandidate &candidate : appended_cave_candidates) {
      if (candidate.entry_island == nullptr)
        try_select_candidate(candidate, true);
    }
  }

  if (!selected_candidates.empty() && selected_candidates.size() < supported_candidate_count &&
      max_patches >= supported_candidate_count) {
    result.warnings.emplace_back(
        "ConSan LDS check/trap proof selected only " + std::to_string(selected_candidates.size()) +
        "/" + std::to_string(supported_candidate_count) +
        " supported sites; scratchable_candidates=" + std::to_string(scratchable_candidate_count) +
        " append_cave_reachable_candidates=" +
        std::to_string(append_cave_reachable_candidate_count) +
        " indirect_pc_resource_failed_sites=" +
        std::to_string(indirect_pc_resource_failed_site_count) +
        " indirect_island_unreachable_sites=" +
        std::to_string(indirect_island_unreachable_site_count) +
        " indirect_candidate_placement_failures=" +
        std::to_string(indirect_candidate_placement_failure_count) +
        " uncovered_nop_caves=" + std::to_string(local_nop_caves.size()) +
        " local_entry_islands=" + std::to_string(local_entry_islands.size()) +
        " local_cave_reachable_candidates=" + std::to_string(local_cave_reachable_candidate_count));
  }

  if (selected_candidates.empty()) {
    if (skipped_supported_site_for_excessive_delay) {
      result.warnings.emplace_back(
          "ConSan LDS check/trap proof skipped: requested delay needs too much "
          "padding for the supported LDS site");
      return;
    }
    result.warnings.emplace_back(
        "ConSan LDS check/trap proof found no padded or local-cave-reachable "
        "ds_load_b{32,64,128}, ds_load_2addr_b{32,64}, ds_load_u16_d16(_hi), or "
        "ds_store_b{32,64,128}, or ds_store_2addr_b{32,64} site; supported_candidates=" +
        std::to_string(supported_candidate_count) +
        " scratchable_candidates=" + std::to_string(scratchable_candidate_count) +
        " max_observed_padding_words=" + std::to_string(max_observed_padding_words) +
        " append_cave_reachable_candidates=" +
        std::to_string(append_cave_reachable_candidate_count) +
        " indirect_pc_resource_failed_sites=" +
        std::to_string(indirect_pc_resource_failed_site_count) +
        " indirect_island_unreachable_sites=" +
        std::to_string(indirect_island_unreachable_site_count) +
        " indirect_candidate_placement_failures=" +
        std::to_string(indirect_candidate_placement_failure_count) +
        " uncovered_nop_caves=" + std::to_string(local_nop_caves.size()) +
        " max_uncovered_nop_cave_words=" + std::to_string(max_uncovered_nop_cave_words) +
        " local_cave_reachable_candidates=" + std::to_string(local_cave_reachable_candidate_count) +
        " s_clause_blocked_candidates=" + std::to_string(s_clause_blocked_candidate_count));
    return;
  }

  struct PlannedLdsCheckTrapPatch {
    const ConSanLdsSite *site = nullptr;
    const LocalNopCave *local_cave = nullptr;
    const LocalNopCave *entry_island = nullptr;
    std::vector<uint32_t> words;
    std::vector<uint32_t> cave_words;
    std::vector<uint32_t> entry_island_words;
    std::array<uint32_t, 2> anchor_words{};
    ConSanPatchInfo info;
    bool use_local_cave = false;
    bool use_appended_cave = false;
    uint64_t descriptor_file_offset = 0;
    std::optional<uint16_t> required_vgpr_allocation_count;
    std::optional<DbiPatchPlacement> placement;
    std::vector<uint64_t> entry_relay_offsets = {};
    std::vector<uint64_t> trailing_relay_offsets;
  };

  std::vector<PlannedLdsCheckTrapPatch> planned_patches;
  planned_patches.reserve(selected_candidates.size());
  for (const LdsCheckTrapCandidate &candidate : selected_candidates) {
    const ConSanLdsSite &site = *candidate.site;
    auto words =
        build_lds_check_trap_words(original_bytes, site, arch, options, *delay_words,
                                   candidate.scratch_vgpr, candidate.vcc_save_sgpr, result.errors);
    if (!words)
      return;

    PlannedLdsCheckTrapPatch planned;
    planned.site = &site;
    planned.local_cave = candidate.local_cave;
    planned.entry_island = candidate.entry_island;
    planned.words = std::move(*words);
    planned.use_local_cave = candidate.use_local_cave;
    planned.use_appended_cave = candidate.use_appended_cave;
    planned.descriptor_file_offset = candidate.descriptor_file_offset;
    planned.required_vgpr_allocation_count = candidate.required_vgpr_allocation_count;
    planned.placement = candidate.placement;
    planned.entry_relay_offsets = candidate.entry_relay_offsets;
    if (auto relay = direct_body_relay_slots.find(candidate.site);
        relay != direct_body_relay_slots.end()) {
      planned.trailing_relay_offsets = relay->second;
    }
    planned.info.kind =
        lds_check_trap_patch_kind(site, candidate.use_local_cave || candidate.use_appended_cave);
    planned.info.anchor_offset = site.text_offset;
    planned.info.scratch_vgpr = candidate.scratch_vgpr;

    if (candidate.use_local_cave || candidate.use_appended_cave) {
      if (!candidate.placement) {
        result.errors.emplace_back("ConSan LDS check/trap proof selected missing placement");
        return;
      }
      if (candidate.use_local_cave && candidate.local_cave == nullptr) {
        result.errors.emplace_back("ConSan LDS check/trap proof selected missing local cave");
        return;
      }
      if (site.size != 2u * sizeof(uint32_t)) {
        result.errors.emplace_back(
            "ConSan LDS check/trap proof local cave expects an 8-byte LDS site");
        return;
      }

      const uint64_t trampoline_text_offset = candidate.placement->body_offset;
      planned.cave_words = planned.words;
      uint64_t entry_target = candidate.placement->body_offset;
      if (candidate.entry_island != nullptr) {
        if (!candidate.indirect_pc_sgpr || !candidate.indirect_scc_save_sgpr) {
          result.errors.emplace_back(
              "ConSan LDS check/trap proof selected missing indirect SGPR scratch");
          return;
        }
        const uint16_t required_sgpr_count = std::max<uint16_t>(
            std::max<uint16_t>(static_cast<uint16_t>(*candidate.indirect_pc_sgpr + 2u),
                               static_cast<uint16_t>(candidate.vcc_save_sgpr + 1u)),
            static_cast<uint16_t>(*candidate.indirect_scc_save_sgpr + 1u));
        planned.info.indirect_pc_sgpr = *candidate.indirect_pc_sgpr;
        planned.info.indirect_saved_scc_sgpr = *candidate.indirect_scc_save_sgpr;
        planned.info.indirect_saved_vcc_sgpr = candidate.vcc_save_sgpr;
        planned.info.indirect_return_offset =
            candidate.placement->body_offset + planned.cave_words.size() * sizeof(uint32_t);
        planned.info.indirect_required_sgpr_count = required_sgpr_count;
        planned.info.owner_descriptor_file_offsets = {candidate.descriptor_file_offset};
        if (!append_lds_check_trap_indirect_return(
                planned.cave_words, candidate.placement->body_offset,
                candidate.placement->return_target, *candidate.indirect_pc_sgpr,
                *candidate.indirect_scc_save_sgpr, arch) ||
            planned.cave_words.size() * sizeof(uint32_t) != candidate.placement->body_size) {
          result.errors.emplace_back(
              "ConSan LDS check/trap proof could not encode its long return");
          return;
        }
        entry_target = candidate.entry_relay_offsets.empty()
                           ? candidate.entry_island->text_offset
                           : candidate.entry_relay_offsets.front();
        if (!append_lds_check_trap_indirect_jump(
                planned.entry_island_words, candidate.entry_island->text_offset,
                candidate.placement->body_offset, *candidate.indirect_pc_sgpr,
                *candidate.indirect_scc_save_sgpr, arch) ||
            planned.entry_island_words.size() != kLdsCheckTrapIndirectJumpWords) {
          result.errors.emplace_back(
              "ConSan LDS check/trap proof could not encode its entry island");
          return;
        }
      } else {
        if (planned.words.size() * sizeof(uint32_t) != candidate.placement->body_size) {
          result.errors.emplace_back(
              "ConSan LDS check/trap proof body size changed after placement");
          return;
        }
        const auto ret = compute_sopp_branch_simm16(candidate.placement->return_branch_offset,
                                                    candidate.placement->return_target);
        if (!ret) {
          result.errors.emplace_back(
              "ConSan LDS check/trap proof direct trampoline return exceeds s_branch simm16");
          return;
        }
        planned.cave_words.push_back(build_s_branch(*ret, arch));
      }
      const auto fwd = compute_sopp_branch_simm16(candidate.placement->anchor_offset, entry_target);
      if (!fwd) {
        result.errors.emplace_back(
            "ConSan LDS check/trap proof trampoline entry exceeds s_branch simm16");
        return;
      }
      if (candidate.use_local_cave &&
          candidate.local_cave->word_count < planned.cave_words.size()) {
        result.errors.emplace_back("ConSan LDS check/trap proof local cave is too small");
        return;
      }

      planned.anchor_words = {build_s_branch(*fwd, arch), build_s_nop(0, arch)};
      const uint64_t anchor_bytes = planned.anchor_words.size() * sizeof(uint32_t);
      const uint64_t cave_bytes = planned.cave_words.size() * sizeof(uint32_t);
      if (site.file_offset > original_bytes.size() ||
          anchor_bytes > original_bytes.size() - site.file_offset ||
          (candidate.use_local_cave &&
           (candidate.local_cave->file_offset > original_bytes.size() ||
            cave_bytes > original_bytes.size() - candidate.local_cave->file_offset)) ||
          (candidate.entry_island != nullptr &&
           (candidate.entry_island->file_offset > original_bytes.size() ||
            kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t) >
                original_bytes.size() - candidate.entry_island->file_offset))) {
        result.errors.emplace_back("ConSan LDS check/trap proof exceeds ELF bytes");
        return;
      }

      planned.info.trampoline_offset = trampoline_text_offset;
      planned.info.original_size = site.size;
      planned.info.trampoline_size =
          static_cast<uint32_t>(planned.cave_words.size() * sizeof(uint32_t) +
                                planned.trailing_relay_offsets.size() * sizeof(uint32_t));
    } else {
      const uint64_t patch_bytes = static_cast<uint64_t>(planned.words.size() * sizeof(uint32_t));
      if (site.file_offset > original_bytes.size() ||
          patch_bytes > original_bytes.size() - site.file_offset) {
        result.errors.emplace_back("ConSan LDS check/trap proof exceeds ELF bytes");
        return;
      }
      planned.info.trampoline_offset = site.text_offset + site.size;
      planned.info.original_size = static_cast<uint32_t>(patch_bytes);
      planned.info.trampoline_size = 0;
    }

    planned_patches.push_back(std::move(planned));
  }

  std::unordered_set<uint64_t> selected_relay_slots;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    if ((planned.use_local_cave || planned.use_appended_cave) &&
        planned.site->size == 2u * sizeof(uint32_t)) {
      selected_relay_slots.insert(planned.site->text_offset + sizeof(uint32_t));
    }
    selected_relay_slots.insert(planned.trailing_relay_offsets.begin(),
                                planned.trailing_relay_offsets.end());
  }
  for (const ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
    selected_relay_slots.insert(donor.anchor_offset + sizeof(uint32_t));
    for (size_t word = 4u; word < donor.host_original_words.size(); ++word)
      selected_relay_slots.insert(donor.host_offset + word * sizeof(uint32_t));
  }
  for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
    for (uint64_t offset :
         sc_relay_reservoir_slot_offsets(reservoir.anchor_offset, reservoir.original_size))
      selected_relay_slots.insert(offset);
  }
  for (size_t index = 0; index < generated_relay_bank_word_count; ++index)
    selected_relay_slots.insert(generated_relay_bank_offset + index * sizeof(uint32_t));
  std::unordered_map<uint64_t, uint32_t> entry_relay_words;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    for (size_t relay_index = 0; relay_index < planned.entry_relay_offsets.size(); ++relay_index) {
      const uint64_t relay_offset = planned.entry_relay_offsets[relay_index];
      if (!selected_relay_slots.contains(relay_offset)) {
        result.errors.emplace_back("ConSan LDS check/trap relay belongs to an unselected anchor");
        return;
      }
      const uint64_t target = relay_index + 1u < planned.entry_relay_offsets.size()
                                  ? planned.entry_relay_offsets[relay_index + 1u]
                                  : planned.entry_island->text_offset;
      const auto branch = compute_sopp_branch_simm16(relay_offset, target);
      if (!branch ||
          !entry_relay_words.emplace(relay_offset, build_s_branch(*branch, arch)).second) {
        result.errors.emplace_back(
            "ConSan LDS check/trap relay path is unreachable or overlapping");
        return;
      }
    }
  }
  const bool uses_appended_cave = std::ranges::any_of(planned_patches,
                                                      [](const PlannedLdsCheckTrapPatch &patch) {
                                                        return patch.use_appended_cave;
                                                      }) ||
                                  !selected_relay_reservoirs.empty();
  std::vector<DescriptorVgprGrowth> descriptor_growths;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    if (planned.required_vgpr_allocation_count) {
      descriptor_growths.push_back(
          {planned.descriptor_file_offset, *planned.required_vgpr_allocation_count});
    }
  }
  std::vector<DescriptorSgprGrowth> descriptor_sgpr_growths;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    if (planned.info.indirect_required_sgpr_count != 0) {
      descriptor_sgpr_growths.push_back(
          {planned.descriptor_file_offset, planned.info.indirect_required_sgpr_count});
    }
  }
  for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
    for (uint64_t descriptor_file_offset : reservoir.owner_descriptor_file_offsets) {
      descriptor_sgpr_growths.push_back({descriptor_file_offset, reservoir.required_sgpr_count});
    }
  }
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_descriptor_vgpr_growths_to_patcher(patcher, original_bytes, descriptor_growths, arch,
                                                  result.errors)) {
      result.patches.clear();
      return;
    }
    if (!apply_descriptor_sgpr_growths_to_patcher(patcher, descriptor_sgpr_growths,
                                                  result.errors)) {
      result.patches.clear();
      return;
    }
    std::span<const uint8_t> text_bytes = patcher.text_bytes();
    std::vector<uint8_t> new_text(text_bytes.begin(), text_bytes.end());
    if (new_text.size() != appended_cave_offset ||
        generated_island_bank_offset < appended_cave_offset ||
        generated_island_bank_offset > std::numeric_limits<size_t>::max() ||
        generated_island_bank_bytes >
            std::numeric_limits<size_t>::max() - generated_island_bank_offset) {
      result.errors.emplace_back("ConSan LDS check/trap proof found an invalid entry-island bank");
      result.patches.clear();
      return;
    }
    const uint32_t padding_nop = build_s_nop(0, arch);
    const size_t relay_bank_begin = static_cast<size_t>(generated_relay_bank_offset);
    const size_t relay_bank_end =
        relay_bank_begin + generated_relay_bank_word_count * sizeof(uint32_t);
    const size_t bank_begin = static_cast<size_t>(generated_island_bank_offset);
    const size_t bank_end = bank_begin + static_cast<size_t>(generated_island_bank_bytes);
    const auto append_island_bank_if_due = [&]() {
      if (new_text.size() != bank_begin)
        return;
      new_text.resize(bank_end);
      for (size_t offset = bank_begin; offset < bank_end; offset += sizeof(uint32_t))
        std::memcpy(new_text.data() + offset, &padding_nop, sizeof(padding_nop));
    };
    bool generated_relay_bank_emitted = false;
    const auto append_generated_relay_bank_if_due = [&]() {
      if (generated_relay_bank_emitted || generated_relay_bank_word_count == 0)
        return;
      if (new_text.size() != relay_bank_begin)
        return;
      new_text.resize(relay_bank_end);
      for (size_t offset = relay_bank_begin; offset < relay_bank_end; offset += sizeof(uint32_t))
        std::memcpy(new_text.data() + offset, &padding_nop, sizeof(padding_nop));
      ConSanPatchInfo info;
      info.kind = ConSanPatchKind::TrampolineScRelayReservoir;
      info.trampoline_offset = generated_relay_bank_offset;
      info.trampoline_size =
          static_cast<uint32_t>(generated_relay_bank_word_count * sizeof(uint32_t));
      result.patches.push_back(std::move(info));
      generated_relay_bank_emitted = true;
    };
    bool wide_composite_relay_donors_emitted = false;
    const auto append_wide_composite_relay_donors_if_due = [&]() -> bool {
      if (wide_composite_relay_donors_emitted || selected_wide_composite_relay_donors.empty())
        return true;
      if (new_text.size() !=
          selected_wide_composite_relay_donors.front().host_placement.body_offset)
        return true;
      for (const ScWideCompositeBranchRelayDonor &donor : selected_wide_composite_relay_donors) {
        if (new_text.size() != donor.host_placement.body_offset)
          return false;
        const auto anchor_to_host =
            compute_sopp_branch_simm16(donor.anchor_offset, donor.host_offset + sizeof(uint32_t));
        const auto anchor_return = compute_sopp_branch_simm16(
            donor.host_offset + 3u * sizeof(uint32_t), donor.anchor_offset + 2u * sizeof(uint32_t));
        const auto host_to_body =
            compute_sopp_branch_simm16(donor.host_offset, donor.host_placement.body_offset);
        const auto host_return = compute_sopp_branch_simm16(
            donor.host_placement.return_branch_offset, donor.host_placement.return_target);
        if (!anchor_to_host || !anchor_return || !host_to_body || !host_return ||
            donor.host_original_words.size() < 6u ||
            donor.host_placement.return_target !=
                donor.host_offset + donor.host_original_words.size() * sizeof(uint32_t)) {
          result.errors.emplace_back(
              "ConSan LDS check/trap wide composite relay exceeds branch reach");
          return false;
        }
        const std::array<uint32_t, 2> anchor_words = {build_s_branch(*anchor_to_host, arch),
                                                      padding_nop};
        std::vector<uint32_t> host_words(donor.host_original_words.size(), padding_nop);
        host_words[0] = build_s_branch(*host_to_body, arch);
        host_words[1] = donor.anchor_original_words[0];
        host_words[2] = donor.anchor_original_words[1];
        host_words[3] = build_s_branch(*anchor_return, arch);
        std::memcpy(new_text.data() + donor.anchor_offset, anchor_words.data(),
                    anchor_words.size() * sizeof(uint32_t));
        std::memcpy(new_text.data() + donor.host_offset, host_words.data(),
                    host_words.size() * sizeof(uint32_t));
        append_words(new_text, donor.host_original_words);
        const uint32_t host_return_word = build_s_branch(*host_return, arch);
        append_words(new_text, std::span<const uint32_t>(&host_return_word, 1));

        ConSanPatchInfo anchor_info;
        anchor_info.kind = ConSanPatchKind::TrampolineScBranchRelayDonor;
        anchor_info.anchor_offset = donor.anchor_offset;
        anchor_info.trampoline_offset = donor.host_offset + sizeof(uint32_t);
        anchor_info.original_size = 2u * sizeof(uint32_t);
        anchor_info.trampoline_size = 3u * sizeof(uint32_t);
        result.patches.push_back(std::move(anchor_info));
        ConSanPatchInfo host_info;
        host_info.kind = ConSanPatchKind::TrampolineScBranchRelayDonor;
        host_info.anchor_offset = donor.host_offset;
        host_info.trampoline_offset = donor.host_placement.body_offset;
        host_info.original_size =
            static_cast<uint32_t>(donor.host_original_words.size() * sizeof(uint32_t));
        host_info.trampoline_size = host_info.original_size + sizeof(uint32_t);
        result.patches.push_back(std::move(host_info));
      }
      wide_composite_relay_donors_emitted = true;
      return true;
    };
    bool relay_reservoirs_emitted = false;
    const auto append_relay_reservoirs_if_due = [&]() -> bool {
      if (relay_reservoirs_emitted || selected_relay_reservoirs.empty())
        return true;
      if (new_text.size() != selected_relay_reservoirs.front().placement.body_offset)
        return true;
      for (const ScRelayReservoir &reservoir : selected_relay_reservoirs) {
        if (new_text.size() != reservoir.placement.body_offset ||
            reservoir.anchor_offset > pristine_text.size() ||
            reservoir.original_size > pristine_text.size() - reservoir.anchor_offset)
          return false;
        std::vector<uint32_t> entry_words;
        entry_words.push_back(build_s_mov_b32(reservoir.saved_vcc_sgpr, kRdna4VccLo, arch));
        if (!append_lds_check_trap_indirect_jump(entry_words, reservoir.anchor_offset,
                                                 reservoir.placement.body_offset,
                                                 reservoir.pc_sgpr_pair, kRdna4ExecHi, arch) ||
            entry_words.size() != kScRelayEntryWords ||
            reservoir.original_size < entry_words.size() * sizeof(uint32_t) ||
            reservoir.placement.return_target != reservoir.anchor_offset + reservoir.original_size -
                                                     kScRelayTailRestoreWords * sizeof(uint32_t)) {
          result.errors.emplace_back(
              "ConSan LDS check/trap could not encode its relay-reservoir entry");
          return false;
        }
        std::memcpy(new_text.data() + reservoir.anchor_offset, entry_words.data(),
                    entry_words.size() * sizeof(uint32_t));
        for (uint64_t offset = reservoir.anchor_offset + entry_words.size() * sizeof(uint32_t);
             offset < reservoir.placement.return_target; offset += sizeof(uint32_t))
          std::memcpy(new_text.data() + offset, &padding_nop, sizeof(padding_nop));
        const uint32_t restore_return_vcc =
            build_s_mov_b32(kRdna4VccLo, reservoir.return_saved_vcc_sgpr, arch);
        std::memcpy(new_text.data() + reservoir.placement.return_target, &restore_return_vcc,
                    sizeof(uint32_t));
        const uint32_t restore_entry_vcc =
            build_s_mov_b32(kRdna4VccLo, reservoir.saved_vcc_sgpr, arch);
        append_words(new_text, std::span<const uint32_t>(&restore_entry_vcc, 1u));
        new_text.insert(new_text.end(), pristine_text.begin() + reservoir.anchor_offset,
                        pristine_text.begin() + reservoir.anchor_offset + reservoir.original_size);
        std::vector<uint32_t> return_words;
        return_words.push_back(build_s_mov_b32(reservoir.return_saved_vcc_sgpr, kRdna4VccLo, arch));
        if (!append_lds_check_trap_indirect_jump(
                return_words,
                reservoir.placement.body_offset + sizeof(uint32_t) + reservoir.original_size,
                reservoir.placement.return_target, reservoir.return_pc_sgpr_pair, kRdna4ExecHi,
                arch) ||
            return_words.size() + 1u != kScRelayAppendedOverheadWords) {
          result.errors.emplace_back(
              "ConSan LDS check/trap could not encode its relay-reservoir return");
          return false;
        }
        append_words(new_text, return_words);

        ConSanPatchInfo info;
        info.kind = ConSanPatchKind::TrampolineScRelayReservoir;
        info.anchor_offset = reservoir.anchor_offset;
        info.trampoline_offset = reservoir.placement.body_offset;
        info.original_size = reservoir.original_size;
        info.trampoline_size =
            reservoir.original_size + kScRelayAppendedOverheadWords * sizeof(uint32_t);
        info.indirect_pc_sgpr = reservoir.pc_sgpr_pair;
        info.indirect_saved_vcc_sgpr = reservoir.saved_vcc_sgpr;
        info.indirect_return_pc_sgpr = reservoir.return_pc_sgpr_pair;
        info.indirect_return_saved_vcc_sgpr = reservoir.return_saved_vcc_sgpr;
        info.indirect_required_sgpr_count = reservoir.required_sgpr_count;
        info.owner_descriptor_file_offsets = reservoir.owner_descriptor_file_offsets;
        result.patches.push_back(std::move(info));
      }
      relay_reservoirs_emitted = true;
      return true;
    };
    for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
      const ConSanLdsSite &site = *planned.site;
      if (planned.use_local_cave) {
        std::memcpy(new_text.data() + site.text_offset, planned.anchor_words.data(),
                    planned.anchor_words.size() * sizeof(uint32_t));
        std::memcpy(new_text.data() + planned.local_cave->text_offset, planned.cave_words.data(),
                    planned.cave_words.size() * sizeof(uint32_t));
      } else if (planned.use_appended_cave) {
        if (!append_wide_composite_relay_donors_if_due()) {
          result.patches.clear();
          return;
        }
        append_generated_relay_bank_if_due();
        append_island_bank_if_due();
        if (!append_relay_reservoirs_if_due()) {
          result.patches.clear();
          return;
        }
        if (!planned.placement || new_text.size() != planned.placement->body_offset) {
          result.errors.emplace_back(
              "ConSan LDS check/trap proof emitted a stale appended-cave mapping");
          result.patches.clear();
          return;
        }
        if (planned.entry_island != nullptr) {
          const uint64_t island_bytes = planned.entry_island_words.size() * sizeof(uint32_t);
          if (island_bytes != kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t) ||
              planned.entry_island->text_offset > new_text.size() ||
              island_bytes > new_text.size() - planned.entry_island->text_offset) {
            result.errors.emplace_back(
                "ConSan LDS check/trap proof emitted an invalid entry-island mapping");
            result.patches.clear();
            return;
          }
          std::memcpy(new_text.data() + planned.entry_island->text_offset,
                      planned.entry_island_words.data(), island_bytes);
        }
        std::memcpy(new_text.data() + site.text_offset, planned.anchor_words.data(),
                    planned.anchor_words.size() * sizeof(uint32_t));
        append_words(new_text, planned.cave_words);
        for (uint64_t trailing_relay_offset : planned.trailing_relay_offsets) {
          if (new_text.size() != trailing_relay_offset) {
            result.errors.emplace_back(
                "ConSan LDS check/trap emitted a stale direct-body relay slot");
            result.patches.clear();
            return;
          }
          append_words(new_text, std::span<const uint32_t>(&padding_nop, 1));
        }
      } else {
        std::memcpy(new_text.data() + site.text_offset, planned.words.data(),
                    planned.words.size() * sizeof(uint32_t));
      }
      if (planned.entry_island != nullptr) {
        ConSanPatchInfo island_info;
        island_info.kind = ConSanPatchKind::TrampolineScIndirectBranchIsland;
        island_info.anchor_offset = site.text_offset;
        island_info.trampoline_offset = planned.entry_island->text_offset;
        island_info.trampoline_size =
            static_cast<uint32_t>(kLdsCheckTrapIndirectJumpWords * sizeof(uint32_t));
        result.patches.push_back(std::move(island_info));
      }
      result.patches.push_back(planned.info);
    }
    if (!append_wide_composite_relay_donors_if_due()) {
      result.patches.clear();
      return;
    }
    append_generated_relay_bank_if_due();
    append_island_bank_if_due();
    if (!append_relay_reservoirs_if_due()) {
      result.patches.clear();
      return;
    }
    for (const auto &[relay_offset, relay_word] : entry_relay_words) {
      if (relay_offset > new_text.size() || sizeof(relay_word) > new_text.size() - relay_offset) {
        result.errors.emplace_back("ConSan LDS check/trap relay exceeds executable text");
        result.patches.clear();
        return;
      }
      std::memcpy(new_text.data() + relay_offset, &relay_word, sizeof(relay_word));
    }
    if (!patcher.replace_text(new_text)) {
      result.errors.emplace_back("ConSan LDS check/trap proof could not append trampoline text");
      result.patches.clear();
      return;
    }
    result.elf_bytes = patcher.emit();
    result.modified = true;
    return;
  }

  if (result.elf_bytes.empty())
    result.elf_bytes.assign(original_bytes.begin(), original_bytes.end());
  if (!apply_descriptor_vgpr_growths_to_bytes(result.elf_bytes, descriptor_growths, arch,
                                              result.errors)) {
    result.patches.clear();
    return;
  }
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    const ConSanLdsSite &site = *planned.site;
    if (planned.use_local_cave) {
      const uint64_t anchor_bytes = planned.anchor_words.size() * sizeof(uint32_t);
      const uint64_t cave_bytes = planned.cave_words.size() * sizeof(uint32_t);
      std::memcpy(result.elf_bytes.data() + site.file_offset, planned.anchor_words.data(),
                  static_cast<size_t>(anchor_bytes));
      std::memcpy(result.elf_bytes.data() + planned.local_cave->file_offset,
                  planned.cave_words.data(), static_cast<size_t>(cave_bytes));
    } else {
      const uint64_t patch_bytes = static_cast<uint64_t>(planned.words.size() * sizeof(uint32_t));
      std::memcpy(result.elf_bytes.data() + site.file_offset, planned.words.data(),
                  static_cast<size_t>(patch_bytes));
    }
    result.patches.push_back(planned.info);
  }
  result.modified = true;
}
