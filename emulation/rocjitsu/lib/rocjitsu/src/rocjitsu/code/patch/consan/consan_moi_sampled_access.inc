void try_apply_direct_sampled_watchpoint_patch(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options, rj_code_arch_t arch,
                                               ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI sampled probe currently supports only RDNA4");
    return;
  }

  const ConSanMoiReportBufferLayout layout = resolve_sampled_report_layout(options);
  const uint32_t total_budget = std::min(options.max_patches, layout.sampled_watchpoint_capacity);
  uint32_t qualified_atomic_sequences = 0;
  if (options.moi_track_atomics) {
    for (const ConSanSyncSequence &sequence : result.sync_sequences) {
      if (sequence.kind != ConSanSyncSequenceKind::Atomic || sequence.width_bits != 32u ||
          !sequence.raw_scope || (*sequence.raw_scope != 2u && *sequence.raw_scope != 3u) ||
          !consan_sync_confidence_meets(sequence.confidence,
                                        ConSanSemanticConfidence::Conservative) ||
          !consan_sync_confidence_meets(sequence.memory_role_confidence,
                                        ConSanSemanticConfidence::Conservative) ||
          (sequence.memory_role != ConSanSyncMemoryRole::Release &&
           sequence.memory_role != ConSanSyncMemoryRole::Acquire &&
           sequence.memory_role != ConSanSyncMemoryRole::AcquireRelease) ||
          (sequence.rmw_outcome != ConSanSyncRmwOutcome::NoReturn &&
           sequence.rmw_outcome != ConSanSyncRmwOutcome::ReturnsOldValue &&
           sequence.rmw_outcome != ConSanSyncRmwOutcome::CompareExchange))
        continue;
      bool eligible = false;
      for (const std::string &identity : sequence.member_event_identities) {
        const auto event =
            std::ranges::find(result.sync_events, identity, &ConSanSyncEvent::identity);
        if (event == result.sync_events.end() || event->kind != ConSanSyncEventKind::Atomic)
          continue;
        const ConSanAtomicSite *site = nullptr;
        if (event->in_kernel) {
          const auto kernel =
              std::ranges::find(result.kernels, event->container_name, &ConSanKernelInfo::name);
          if (kernel != result.kernels.end()) {
            const auto found = std::ranges::find(kernel->atomic_sites, event->text_offset,
                                                 &ConSanAtomicSite::text_offset);
            if (found != kernel->atomic_sites.end())
              site = &*found;
          }
        } else {
          const auto function =
              std::ranges::find(result.functions, event->container_name, &ConSanFunctionInfo::name);
          if (function != result.functions.end()) {
            const auto found = std::ranges::find(function->atomic_sites, event->text_offset,
                                                 &ConSanAtomicSite::text_offset);
            if (found != function->atomic_sites.end())
              site = &*found;
          }
        }
        if (site != nullptr &&
            (site->mnemonic.starts_with("flat_atomic") ||
             site->mnemonic.starts_with("global_atomic")) &&
            (sequence.rmw_outcome != ConSanSyncRmwOutcome::CompareExchange ||
             site->returns_old_value.value_or(false))) {
          eligible = true;
          break;
        }
      }
      if (eligible && qualified_atomic_sequences != std::numeric_limits<uint32_t>::max())
        ++qualified_atomic_sequences;
    }
  }
  const uint32_t reserved_atomic_slots = std::min(total_budget, qualified_atomic_sequences);
  const uint32_t max_candidates = total_budget - reserved_atomic_slots;
  if (max_candidates == 0) {
    if (total_budget == 0) {
      result.warnings.emplace_back(
          "ConSan MOI sampled probe requires room for the report header and one sampled entry");
    }
    return;
  }
  if (options.moi_sample_stride == 0) {
    result.warnings.emplace_back("ConSan MOI sampled probe requires a nonzero sample stride");
    return;
  }
  if (options.moi_sample_offset >= options.moi_sample_stride) {
    result.warnings.emplace_back("ConSan MOI sampled probe sample offset must be less than stride");
    return;
  }
  if (options.moi_runtime_sample_stride == 0 || options.moi_runtime_sample_stride > (1u << 24u) ||
      (options.moi_runtime_sample_stride & (options.moi_runtime_sample_stride - 1u)) != 0 ||
      options.moi_runtime_sample_offset >= options.moi_runtime_sample_stride) {
    result.warnings.emplace_back(
        "ConSan MOI runtime sample stride must be a power of two in 1..16777216 and offset must "
        "be less than stride");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates =
      find_first_light_access_record_candidates(result, bytes);
  apply_test_kernel_filter(candidates, options);
  std::ranges::sort(candidates, [](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
    return lhs->text_offset < rhs->text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled probe found no native LDS or likely group flat load/store candidate");
    return;
  }

  uint32_t sampled_filter_candidate_count = 0;
  uint32_t sampled_filter_range_count = 0;
  for (uint32_t i = 0; i < candidates.size(); ++i) {
    if (i % options.moi_sample_stride == options.moi_sample_offset) {
      ++sampled_filter_candidate_count;
      const auto ranges = candidate_access_ranges(bytes, *candidates[i]);
      if (ranges &&
          ranges->size() <= std::numeric_limits<uint32_t>::max() - sampled_filter_range_count)
        sampled_filter_range_count += static_cast<uint32_t>(ranges->size());
    }
  }
  uint32_t window_bank_count = 1;
  if (options.moi_runtime_sample_stride > 1 && sampled_filter_range_count != 0) {
    const uint32_t available_per_range = max_candidates / sampled_filter_range_count;
    while (window_bank_count < 8u && window_bank_count * 2u <= available_per_range)
      window_bank_count *= 2u;
  }

  struct PlannedSampledPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    uint32_t first_record_index = 0;
    uint32_t access_range_count = 0;
    std::optional<uint32_t> prior_first_record_index;
    uint32_t prior_access_range_count = 0;
    uint32_t runtime_sample_index = 0;
    DbiPatchPlacement placement;
    std::optional<uint64_t> entry_island_offset;
    uint64_t entry_island_word_count = 0;
    bool entry_island_at_anchor = false;
    std::vector<uint32_t> displaced_tail_words;
    uint64_t workgroup_descriptor_file_offset = 0;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
    bool runtime_fast_gate = false;
    std::optional<uint32_t> private_epoch_offset;
    std::optional<uint32_t> private_sample_sequence_offset;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedSampledPatch> planned_patches;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  DbiPatchPlacementPlanner placement_planner(arch, original_text_size);
  constexpr uint64_t kIndirectIslandWords = 7u;
  constexpr uint64_t kSampledAccessIslandWords = 40u;
  const uint64_t access_island_word_count =
      options.moi_runtime_sample_stride > 1u ? kSampledAccessIslandWords : kIndirectIslandWords;
  const bool has_function_candidate = std::ranges::any_of(
      candidates, [](const ConSanMoiCandidate *candidate) { return !candidate->in_kernel; });
  const bool use_indirect_appended =
      options.moi_exec_save_sgpr.has_value() && (candidates.size() > 8u || has_function_candidate);
  uint64_t reserved_barrier_island_count = 0u;
  if (use_indirect_appended && options.moi_track_barriers) {
    for (const ConSanKernelInfo &kernel : result.kernels)
      reserved_barrier_island_count += kernel.barrier_sites.size();
    for (const ConSanFunctionInfo &function : result.functions)
      reserved_barrier_island_count += function.barrier_sites.size();
    reserved_barrier_island_count =
        std::min<uint64_t>(reserved_barrier_island_count, options.max_patches);
  }
  const uint64_t indirect_body_begin =
      original_text_size + reserved_barrier_island_count * kIndirectIslandWords * sizeof(uint32_t) +
      candidates.size() * access_island_word_count * sizeof(uint32_t);
  const uint64_t access_island_begin =
      original_text_size + reserved_barrier_island_count * kIndirectIslandWords * sizeof(uint32_t);
  uint64_t indirect_body_cursor = indirect_body_begin;
  std::span<const uint8_t> original_text;
  if (code_object.text_sections().size() == 1) {
    const Section *text = code_object.text_sections().front();
    original_text =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
  }
  std::optional<MoiLocalNopIslandAllocator> local_entry_islands;
  if (use_indirect_appended && !original_text.empty())
    local_entry_islands.emplace(original_text, result, arch, kIndirectIslandWords);
  uint32_t sampled_candidate_index = 0;
  uint32_t next_record_index = 0;
  bool capacity_limited = false;
  MoiSpillManagers spill_managers;
  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const uint32_t current_sample_index = sampled_candidate_index++;
    if (current_sample_index % options.moi_sample_stride != options.moi_sample_offset)
      continue;

    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const auto access_ranges = candidate_access_ranges(bytes, candidate);
    if (!access_ranges || access_ranges->empty())
      continue;
    const uint32_t range_count = static_cast<uint32_t>(access_ranges->size());
    const uint32_t required_slots = range_count * window_bank_count;
    if (required_slots > max_candidates - next_record_index) {
      capacity_limited = true;
      continue;
    }
    const auto resources = resolve_moi_scratch(result, candidate, options,
                                               direct_sampled_scratch_count(options, candidate));
    if (!resources) {
      continue;
    }
    std::optional<uint64_t> workgroup_descriptor = candidate.kernel_descriptor_file_offset;
    if (!workgroup_descriptor) {
      workgroup_descriptor =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
    }
    if (!workgroup_descriptor)
      continue;
    std::optional<MoiPrivateEpochLayout> private_layout;
    if (options.automatic_moi_private_epoch) {
      private_layout = build_moi_private_epoch_layout(result, *resources, result.warnings);
      if (!private_layout)
        continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(
          result, *resources, options, spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt);
      if (!spill)
        continue;
    }
    std::optional<uint64_t> entry_island_offset;
    bool uses_local_entry_island = false;
    bool entry_island_at_anchor = false;
    std::vector<uint32_t> displaced_tail_words;
    uint32_t displaced_original_size = candidate.size;
    std::vector<std::string> candidate_errors;
    if (use_indirect_appended) {
      entry_island_offset = access_island_begin +
                            planned_patches.size() * access_island_word_count * sizeof(uint32_t);
      if (candidate.size >= sizeof(uint32_t) &&
          !compute_sopp_branch_simm16(candidate.text_offset, *entry_island_offset)) {
        entry_island_offset = local_entry_islands
                                  ? local_entry_islands->claim_reachable(candidate.text_offset)
                                  : std::nullopt;
        uses_local_entry_island = entry_island_offset.has_value();
      }
      if (!entry_island_offset && !original_text.empty()) {
        std::vector<uint32_t> displaced_words;
        while (displaced_words.size() < kIndirectIslandWords) {
          auto decoded = decode_relocatable_entry_instruction(
              original_text, candidate.text_offset + displaced_words.size() * sizeof(uint32_t),
              arch, candidate_errors);
          if (!decoded) {
            displaced_words.clear();
            break;
          }
          displaced_words.insert(displaced_words.end(), decoded->begin(), decoded->end());
        }
        if (displaced_words.size() >= kIndirectIslandWords &&
            displaced_words.size() * sizeof(uint32_t) <= std::numeric_limits<uint32_t>::max()) {
          const size_t candidate_words = candidate.size / sizeof(uint32_t);
          if (candidate_words <= displaced_words.size()) {
            displaced_tail_words.assign(displaced_words.begin() + candidate_words,
                                        displaced_words.end());
            displaced_original_size =
                static_cast<uint32_t>(displaced_words.size() * sizeof(uint32_t));
            entry_island_offset = candidate.text_offset;
            uses_local_entry_island = true;
            entry_island_at_anchor = true;
          }
        }
      }
      if (candidate.size < sizeof(uint32_t) || !entry_island_offset) {
        result.warnings.emplace_back(
            "ConSan MOI sampled probe skipped load/store site: entry island is unreachable");
        continue;
      }
    }
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    if (private_layout) {
      const uint16_t end = static_cast<uint16_t>(resources->base + resources->count);
      if (private_layout->sample_sequence_offset) {
        candidate_options.moi_epoch_vgpr = static_cast<uint16_t>(end - 2u);
        candidate_options.moi_sample_sequence_vgpr = static_cast<uint16_t>(end - 1u);
      } else {
        candidate_options.moi_epoch_vgpr = static_cast<uint16_t>(end - 1u);
      }
    }
    const bool runtime_fast_gate = use_indirect_appended && !uses_local_entry_island &&
                                   options.moi_runtime_sample_stride > 1u &&
                                   options.moi_runtime_sample_offset <= 1024u;
    ConSanMoiCandidate lowered_candidate = candidate;
    lowered_candidate.kernel_descriptor_file_offset = workgroup_descriptor;
    const auto prior_patch = std::find_if(
        planned_patches.rbegin(), planned_patches.rend(), [&](const PlannedSampledPatch &planned) {
          return planned.workgroup_descriptor_file_offset == *workgroup_descriptor;
        });
    const std::optional<uint32_t> prior_first_record_index =
        prior_patch == planned_patches.rend()
            ? std::nullopt
            : std::optional<uint32_t>(prior_patch->first_record_index);
    const uint32_t prior_access_range_count =
        prior_patch == planned_patches.rend() ? 0u : prior_patch->access_range_count;
    auto words = build_direct_sampled_watchpoint_words(
        bytes, lowered_candidate, candidate_options, arch, next_record_index,
        prior_first_record_index, prior_access_range_count, window_bank_count,
        layout.sampled_causal_windows_offset, layout.sampled_watchpoints_offset,
        layout.sampled_pending_acquires_offset,
        private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        private_layout ? private_layout->sample_sequence_offset : std::nullopt, candidate_errors);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t spill_bytes =
        spill ? static_cast<uint64_t>((spill->save_words.size() + spill->restore_words.size()) *
                                      sizeof(uint32_t))
              : 0;
    const uint64_t patch_bytes =
        static_cast<uint64_t>((words->size() + displaced_tail_words.size()) * sizeof(uint32_t)) +
        spill_bytes;
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    std::optional<DbiPatchPlacement> placement;
    if (use_indirect_appended) {
      placement = DbiPatchPlacement{
          .kind = DbiPatchPlacementKind::AppendedCave,
          .anchor_offset = candidate.text_offset,
          .original_size = displaced_original_size,
          .body_offset = indirect_body_cursor,
          .body_size = patch_bytes,
          .return_branch_offset = 0,
          .return_target = candidate.text_offset + displaced_original_size,
      };
      indirect_body_cursor += patch_bytes + 6u * sizeof(uint32_t);
    } else {
      DbiPatchPlacementRequest placement_request;
      placement_request.anchor_offset = candidate.text_offset;
      placement_request.original_size = candidate.size;
      placement_request.body_size = patch_bytes;
      placement_request.inline_capacity = spill || private_layout ? 0u : available_bytes;
      std::string placement_error;
      placement = placement_planner.plan(placement_request, &placement_error);
      if (!placement) {
        result.warnings.emplace_back("ConSan MOI sampled probe skipped load/store site: " +
                                     placement_error);
        continue;
      }
    }

    PlannedSampledPatch planned;
    planned.candidate = candidate_ptr;
    planned.first_record_index = next_record_index;
    planned.access_range_count = range_count;
    planned.prior_first_record_index = prior_first_record_index;
    planned.prior_access_range_count = prior_access_range_count;
    planned.runtime_sample_index = current_sample_index;
    planned.placement = *placement;
    planned.entry_island_offset = entry_island_offset;
    planned.entry_island_word_count =
        uses_local_entry_island ? kIndirectIslandWords : access_island_word_count;
    planned.entry_island_at_anchor = entry_island_at_anchor;
    planned.displaced_tail_words = std::move(displaced_tail_words);
    planned.workgroup_descriptor_file_offset = *workgroup_descriptor;
    planned.resources = *resources;
    planned.spill = spill;
    planned.runtime_fast_gate = runtime_fast_gate;
    if (private_layout)
      planned.private_epoch_offset = private_layout->epoch_offset;
    if (private_layout)
      planned.private_sample_sequence_offset = private_layout->sample_sequence_offset;
    planned.required_private_bytes = std::max(private_layout ? private_layout->ephemeral_base : 0u,
                                              spill ? spill->total_private_bytes : 0u);
    planned_patches.push_back(std::move(planned));
    next_record_index += required_slots;
    if (next_record_index == max_candidates)
      break;
  }

  if (planned_patches.empty()) {
    if (capacity_limited) {
      result.warnings.emplace_back(
          "ConSan MOI sampled probe report capacity cannot retain every range of a selected "
          "load/store site");
    } else if (options.moi_sample_stride != 1 || options.moi_sample_offset != 0) {
      result.warnings.emplace_back(
          "ConSan MOI sampled probe selected no candidate after applying the sample filter");
    }
    return;
  }

  if ((capacity_limited || next_record_index == max_candidates) &&
      sampled_filter_candidate_count > planned_patches.size()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled probe limited sampled patches to " +
        std::to_string(planned_patches.size()) + " of " +
        std::to_string(sampled_filter_candidate_count) +
        " sample-filter-selected candidates by report capacity or max-patch budget");
  }

  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const PlannedSampledPatch &planned_patch : planned_patches) {
    note_descriptor_requirements(descriptor_requirements, planned_patch.resources);
    note_moi_sgpr_requirements(scalar_requirements, planned_patch.resources, options);
    if (planned_patch.required_private_bytes != 0u) {
      for (uint64_t descriptor_offset : planned_patch.resources.owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, planned_patch.required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, planned_patch.required_private_bytes);
      }
    }
  }

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedSampledPatch &patch) {
        return patch.placement.kind == DbiPatchPlacementKind::AppendedCave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_descriptor_requirements(patcher, code_object, bytes, result, descriptor_requirements,
                                       result.errors)) {
      return;
    }
    if (!apply_spill_descriptor_requirements(patcher, code_object, bytes, result,
                                             private_requirements, result.errors)) {
      return;
    }
    if (!apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements,
                                            result.errors))
      return;
    const std::span<const uint8_t> old_text = patcher.text_bytes();
    if (old_text.empty()) {
      result.errors.emplace_back("ConSan MOI sampled probe found no .text section");
      return;
    }
    std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
    if (use_indirect_appended) {
      while (new_text.size() < indirect_body_begin)
        append_word_bytes(new_text, build_s_nop(0, arch));
    }
    std::vector<ConSanPatchInfo> patches;
    patches.reserve(planned_patches.size());
    for (const PlannedSampledPatch &planned_patch : planned_patches) {
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      ConSanOptions candidate_options = options;
      candidate_options.scratch_vgpr = planned_patch.resources.base;
      if (planned_patch.private_epoch_offset) {
        const uint16_t end =
            static_cast<uint16_t>(planned_patch.resources.base + planned_patch.resources.count);
        if (planned_patch.private_sample_sequence_offset) {
          candidate_options.moi_epoch_vgpr = static_cast<uint16_t>(end - 2u);
          candidate_options.moi_sample_sequence_vgpr = static_cast<uint16_t>(end - 1u);
        } else {
          candidate_options.moi_epoch_vgpr = static_cast<uint16_t>(end - 1u);
        }
      }
      ConSanMoiCandidate lowered_candidate = candidate;
      lowered_candidate.kernel_descriptor_file_offset =
          planned_patch.workgroup_descriptor_file_offset;
      auto words = build_direct_sampled_watchpoint_words(
          bytes, lowered_candidate, candidate_options, arch, planned_patch.first_record_index,
          planned_patch.prior_first_record_index, planned_patch.prior_access_range_count,
          window_bank_count, layout.sampled_causal_windows_offset,
          layout.sampled_watchpoints_offset, layout.sampled_pending_acquires_offset,
          planned_patch.private_epoch_offset, planned_patch.private_sample_sequence_offset,
          result.errors);
      if (!words)
        return;
      const uint64_t spill_bytes =
          planned_patch.spill ? static_cast<uint64_t>((planned_patch.spill->save_words.size() +
                                                       planned_patch.spill->restore_words.size()) *
                                                      sizeof(uint32_t))
                              : 0;
      const uint64_t displaced_tail_bytes =
          planned_patch.displaced_tail_words.size() * sizeof(uint32_t);
      if (spill_bytes + displaced_tail_bytes > planned_patch.placement.body_size) {
        result.errors.emplace_back("ConSan MOI sampled probe has an invalid relocated body size");
        return;
      }
      const uint64_t planned_probe_bytes =
          planned_patch.placement.body_size - spill_bytes - displaced_tail_bytes;
      if (words->size() * sizeof(uint32_t) > planned_probe_bytes) {
        result.errors.emplace_back("ConSan MOI sampled probe final patch grew after planning");
        return;
      }
      words->resize(static_cast<size_t>(planned_probe_bytes / sizeof(uint32_t)),
                    build_s_nop(0, arch));

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiSampledWatchpointStore
                      : ConSanPatchKind::InlineMoiSampledWatchpointStore;
      info.anchor_offset = candidate.text_offset;
      info.sampled_first_slot = planned_patch.first_record_index;
      info.sampled_window_bank_count = window_bank_count;
      info.sampled_access_kind = candidate.kind;
      info.sampled_access_range_count = planned_patch.access_range_count;
      info.scratch_vgpr = planned_patch.resources.base;
      info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
      info.required_private_segment_size = planned_patch.required_private_bytes;
      info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
      if (planned_patch.spill) {
        info.spilled_vgpr_count = planned_patch.spill->vgpr_count;
      }

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe emitted a stale appended-cave mapping");
          return;
        }
        const uint64_t entry_target =
            planned_patch.entry_island_offset.value_or(planned_patch.placement.body_offset);
        const auto ret =
            planned_patch.entry_island_offset
                ? std::optional<int16_t>()
                : compute_sopp_branch_simm16(planned_patch.placement.return_branch_offset,
                                             planned_patch.placement.return_target);
        if (!write_moi_appended_anchor(new_text, planned_patch, entry_target, arch, "sampled probe",
                                       result.errors) ||
            (!planned_patch.entry_island_offset && !ret)) {
          result.errors.emplace_back(
              "ConSan MOI sampled probe appended cave branch is out of range");
          return;
        }

        if (planned_patch.entry_island_offset) {
          const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(options);
          std::vector<uint32_t> island_words;
          if (planned_patch.runtime_fast_gate) {
            std::vector<std::string> workgroup_errors;
            const auto workgroup_sources = moi_descriptor_workgroup_sources(
                bytes, planned_patch.workgroup_descriptor_file_offset, arch, workgroup_errors);
            if (!workgroup_sources) {
              result.errors.insert(result.errors.end(), workgroup_errors.begin(),
                                   workgroup_errors.end());
              return;
            }
            auto fast_gate = build_sampled_runtime_fast_gate_island_words(
                bytes, candidate, options, *workgroup_sources, planned_patch.runtime_sample_index,
                *planned_patch.entry_island_offset, cave_text_offset,
                planned_patch.placement.return_target, planned_patch.entry_island_word_count, arch);
            if (fast_gate)
              island_words = std::move(*fast_gate);
          } else if (append_moi_scc_preserving_indirect_jump(
                         island_words, *planned_patch.entry_island_offset, cave_text_offset,
                         *options.moi_exec_save_sgpr, special_state.scc_save_sgpr,
                         /*capture_scc=*/true, arch)) {
            island_words.resize(planned_patch.entry_island_word_count, build_s_nop(0, arch));
          }
          if (island_words.size() != planned_patch.entry_island_word_count) {
            result.errors.emplace_back(
                "ConSan MOI sampled probe could not encode its entry island");
            return;
          }
          if (!write_moi_entry_island(new_text, planned_patch, island_words, patches,
                                      "sampled probe", result.errors))
            return;
        }

        auto cave_words =
            assemble_moi_appended_body(planned_patch, *words, "sampled probe", result.errors);
        if (!cave_words)
          return;
        if (planned_patch.entry_island_offset) {
          const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(options);
          if (!append_moi_scc_preserving_indirect_jump(
                  *cave_words, cave_text_offset, planned_patch.placement.return_target,
                  *options.moi_exec_save_sgpr, special_state.scc_save_sgpr,
                  /*capture_scc=*/false, arch)) {
            result.errors.emplace_back("ConSan MOI sampled probe could not encode its long return");
            return;
          }
        } else {
          cave_words->push_back(build_s_branch(*ret, arch));
        }
        append_words_bytes(new_text, *cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = planned_patch.placement.original_size;
        info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.placement.body_size) {
          result.errors.emplace_back("ConSan MOI sampled probe final patch size changed");
          return;
        }
        if (candidate.text_offset > new_text.size() ||
            patch_bytes > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI sampled probe exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.text_offset, words->data(),
                    static_cast<size_t>(patch_bytes));
        info.trampoline_offset = candidate.text_offset + candidate.size;
        info.original_size = static_cast<uint32_t>(patch_bytes);
        info.trampoline_size = 0;
      }
      patches.push_back(info);
    }

    if (!patcher.replace_text(new_text)) {
      result.errors.emplace_back("ConSan MOI sampled probe could not grow .text");
      return;
    }
    result.elf_bytes = patcher.emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  if (!apply_inline_moi_access_patches(
          bytes, planned_patches, descriptor_requirements, scalar_requirements, "sampled probe",
          [&](const PlannedSampledPatch &planned_patch) {
            ConSanOptions candidate_options = options;
            candidate_options.scratch_vgpr = planned_patch.resources.base;
            if (planned_patch.private_epoch_offset) {
              const uint16_t end = static_cast<uint16_t>(planned_patch.resources.base +
                                                         planned_patch.resources.count);
              if (planned_patch.private_sample_sequence_offset) {
                candidate_options.moi_epoch_vgpr = static_cast<uint16_t>(end - 2u);
                candidate_options.moi_sample_sequence_vgpr = static_cast<uint16_t>(end - 1u);
              } else {
                candidate_options.moi_epoch_vgpr = static_cast<uint16_t>(end - 1u);
              }
            }
            ConSanMoiCandidate lowered_candidate = *planned_patch.candidate;
            lowered_candidate.kernel_descriptor_file_offset =
                planned_patch.workgroup_descriptor_file_offset;
            auto words = build_direct_sampled_watchpoint_words(
                bytes, lowered_candidate, candidate_options, arch, planned_patch.first_record_index,
                planned_patch.prior_first_record_index, planned_patch.prior_access_range_count,
                window_bank_count, layout.sampled_causal_windows_offset,
                layout.sampled_watchpoints_offset, layout.sampled_pending_acquires_offset,
                planned_patch.private_epoch_offset, planned_patch.private_sample_sequence_offset,
                result.errors);
            if (words) {
              const size_t planned_words =
                  static_cast<size_t>(planned_patch.placement.body_size / sizeof(uint32_t));
              if (words->size() > planned_words) {
                result.errors.emplace_back(
                    "ConSan MOI sampled probe final patch grew after planning");
                return std::optional<std::vector<uint32_t>>{};
              }
              words->resize(planned_words, build_s_nop(0, arch));
            }
            return words;
          },
          [&](const PlannedSampledPatch &planned_patch) {
            const ConSanMoiCandidate &candidate = *planned_patch.candidate;
            ConSanPatchInfo info;
            info.kind = ConSanPatchKind::InlineMoiSampledWatchpointStore;
            info.anchor_offset = candidate.text_offset;
            info.trampoline_offset = candidate.text_offset + candidate.size;
            info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
            info.sampled_first_slot = planned_patch.first_record_index;
            info.sampled_window_bank_count = window_bank_count;
            info.sampled_access_kind = candidate.kind;
            info.sampled_access_range_count = planned_patch.access_range_count;
            info.scratch_vgpr = planned_patch.resources.base;
            info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
            info.required_private_segment_size = planned_patch.required_private_bytes;
            info.owner_descriptor_file_offsets =
                planned_patch.resources.owner_descriptor_file_offsets;
            return info;
          },
          [](std::vector<uint8_t> &, ConSanResult &) { return true; }, result))
    return;
}
