void try_apply_first_light_access_record_patch(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options, rj_code_arch_t arch,
                                               ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI first-light probe currently supports only RDNA4");
    return;
  }
  if (options.moi_report_buffer_size < consan_moi_report_buffer_min_bytes(1, 0, 0, 0)) {
    result.warnings.emplace_back(
        "ConSan MOI first-light probe requires room for the report header and one access record");
    return;
  }

  const ConSanMoiReportBufferLayout layout = resolve_record_replay_report_layout(
      options, options.moi_track_barriers, options.moi_track_atomics, options.moi_track_atomics);
  if (options.max_patches == 0 || layout.access_record_capacity == 0)
    return;
  if (options.moi_dynamic_access_records && !options.moi_exec_save_sgpr) {
    result.warnings.emplace_back(
        "ConSan MOI dynamic access-record probe requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return;
  }
  if (options.moi_dynamic_access_records &&
      (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u)) {
    result.warnings.emplace_back(
        "ConSan MOI dynamic access-record probe requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in "
        "0..100");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates =
      find_first_light_access_record_candidates(result, bytes);
  apply_test_kernel_filter(candidates, options);
  std::ranges::sort(candidates, [](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
    return lhs->text_offset < rhs->text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI first-light probe found no native LDS or likely group "
                                 "flat load/store candidate");
    return;
  }

  struct PlannedAccessRecordPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    DbiPatchPlacement placement;
    std::optional<uint64_t> entry_island_offset;
    bool entry_island_at_anchor = false;
    std::vector<uint32_t> displaced_tail_words;
    uint32_t record_index = 0;
    uint32_t record_count = 0;
    ResolvedMoiScratchPlan resources;
    std::optional<VgprSpillSequence> spill;
  };
  std::vector<PlannedAccessRecordPatch> planned_patches;
  uint32_t planned_record_count = 0;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  DbiPatchPlacementPlanner placement_planner(arch, original_text_size);
  constexpr uint64_t kIndirectIslandWords = 7u;
  const bool has_unreachable_function_candidate =
      std::ranges::any_of(candidates, [&](const ConSanMoiCandidate *candidate) {
        return !candidate->in_kernel &&
               !compute_sopp_branch_simm16(candidate->text_offset, original_text_size);
      });
  const bool use_indirect_appended = options.moi_exec_save_sgpr.has_value() &&
                                     (candidates.size() > 8u || has_unreachable_function_candidate);
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
      original_text_size +
      (candidates.size() + reserved_barrier_island_count) * kIndirectIslandWords * sizeof(uint32_t);
  const uint64_t access_island_begin =
      original_text_size + reserved_barrier_island_count * kIndirectIslandWords * sizeof(uint32_t);
  uint64_t indirect_body_cursor = indirect_body_begin;
  std::optional<MoiLocalNopIslandAllocator> local_entry_islands;
  std::span<const uint8_t> original_text;
  if (use_indirect_appended && code_object.text_sections().size() == 1) {
    const Section *text = code_object.text_sections().front();
    const auto text_bytes =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
    original_text = text_bytes;
    local_entry_islands.emplace(text_bytes, result, arch, kIndirectIslandWords);
  }
  MoiSpillManagers spill_managers;
  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const uint16_t access_scratch_count = moi_access_scratch_vgpr_count(options, candidate);
    const auto resources = resolve_moi_scratch(result, candidate, options, access_scratch_count);
    if (!resources) {
      const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(result, candidate);
      if (plan != nullptr && plan->reason == ConSanRegisterPlanReason::ForbiddenOverlap &&
          options.scratch_vgpr) {
        std::vector<std::string> overlap_errors;
        const bool rejected = reject_candidate_scratch_range_overlap(
            candidate, *options.scratch_vgpr, access_scratch_count, overlap_errors);
        if (!rejected)
          overlap_errors.emplace_back("ConSan MOI probe scratch VGPRs overlap a forbidden range");
        result.warnings.insert(result.warnings.end(), overlap_errors.begin(), overlap_errors.end());
      }
      continue;
    }
    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      spill = build_moi_spill_sequence(result, *resources, options, spill_managers, arch,
                                       result.warnings);
      if (!spill)
        continue;
    }
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = resources->base;
    const std::optional<std::vector<ConSanMoiAccessRange>> access_ranges =
        candidate_access_ranges(bytes, candidate);
    if (!access_ranges || access_ranges->empty())
      continue;
    const uint32_t candidate_record_count = static_cast<uint32_t>(access_ranges->size());
    if (!options.moi_dynamic_access_records &&
        candidate_record_count > layout.access_record_capacity - planned_record_count) {
      result.warnings.emplace_back(
          "ConSan MOI first-light probe skipped a supported load/store site without enough "
          "report-buffer access-record slots");
      continue;
    }
    std::vector<std::string> candidate_errors;
    auto words =
        build_first_light_access_record_words(bytes, candidate, candidate_options, arch, 0, 0,
                                              layout.access_record_capacity, candidate_errors);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t spill_bytes =
        spill ? static_cast<uint64_t>((spill->save_words.size() + spill->restore_words.size()) *
                                      sizeof(uint32_t))
              : 0;
    std::vector<uint32_t> displaced_tail_words;
    uint32_t displaced_original_size = candidate.size;
    bool entry_island_at_anchor = false;
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    std::optional<DbiPatchPlacement> placement;
    std::optional<uint64_t> entry_island_offset;
    if (use_indirect_appended) {
      entry_island_offset =
          access_island_begin + planned_patches.size() * kIndirectIslandWords * sizeof(uint32_t);
      if (candidate.size < sizeof(uint32_t) ||
          !compute_sopp_branch_simm16(candidate.text_offset, *entry_island_offset)) {
        entry_island_offset = local_entry_islands
                                  ? local_entry_islands->claim_reachable(candidate.text_offset)
                                  : std::nullopt;
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
            entry_island_at_anchor = true;
          }
        }
      }
      if (candidate.size < sizeof(uint32_t) || !entry_island_offset) {
        result.warnings.emplace_back(
            "ConSan MOI first-light probe skipped load/store site: entry island is unreachable");
        continue;
      }
      const uint64_t patch_bytes =
          static_cast<uint64_t>((words->size() + displaced_tail_words.size()) * sizeof(uint32_t)) +
          spill_bytes;
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
      const uint64_t patch_bytes =
          static_cast<uint64_t>(words->size() * sizeof(uint32_t)) + spill_bytes;
      DbiPatchPlacementRequest placement_request;
      placement_request.anchor_offset = candidate.text_offset;
      placement_request.original_size = candidate.size;
      placement_request.body_size = patch_bytes;
      placement_request.inline_capacity = spill ? 0u : available_bytes;
      std::string placement_error;
      placement = placement_planner.plan(placement_request, &placement_error);
      if (!placement) {
        result.warnings.emplace_back("ConSan MOI first-light probe skipped load/store site: " +
                                     placement_error);
        continue;
      }
    }

    PlannedAccessRecordPatch planned;
    planned.candidate = candidate_ptr;
    planned.placement = *placement;
    planned.entry_island_offset = entry_island_offset;
    planned.entry_island_at_anchor = entry_island_at_anchor;
    planned.displaced_tail_words = std::move(displaced_tail_words);
    planned.record_index = planned_record_count;
    planned.record_count = candidate_record_count;
    planned.resources = *resources;
    planned.spill = spill;
    planned_patches.push_back(std::move(planned));
    planned_record_count += candidate_record_count;
    if (planned_patches.size() == options.max_patches)
      break;
  }

  if (planned_patches.empty())
    return;

  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
    note_descriptor_requirements(descriptor_requirements, planned_patch.resources);
    note_moi_sgpr_requirements(scalar_requirements, planned_patch.resources, options);
    if (planned_patch.spill) {
      note_spill_descriptor_requirements(private_requirements, planned_patch.resources,
                                         *planned_patch.spill);
    }
  }

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedAccessRecordPatch &patch) {
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
      result.errors.emplace_back("ConSan MOI first-light probe found no .text section");
      return;
    }
    std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
    if (use_indirect_appended) {
      while (new_text.size() < indirect_body_begin)
        append_word_bytes(new_text, build_s_nop(0, arch));
    }
    std::vector<ConSanPatchInfo> patches;
    patches.reserve(planned_patches.size());
    for (const PlannedAccessRecordPatch &planned_patch : planned_patches) {
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      ConSanOptions candidate_options = options;
      candidate_options.scratch_vgpr = planned_patch.resources.base;
      auto words = build_first_light_access_record_words(
          bytes, candidate, candidate_options, arch, planned_patch.record_index,
          planned_record_count, layout.access_record_capacity, result.errors);
      if (!words)
        return;

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiAccessRecordStore
                      : ConSanPatchKind::InlineMoiAccessRecordStore;
      info.anchor_offset = candidate.text_offset;
      info.scratch_vgpr = planned_patch.resources.base;
      info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
      if (planned_patch.spill) {
        info.spilled_vgpr_count = planned_patch.spill->vgpr_count;
        info.required_private_segment_size = planned_patch.spill->total_private_bytes;
      }

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe emitted a stale appended-cave mapping");
          return;
        }
        const uint64_t entry_target =
            planned_patch.entry_island_offset.value_or(planned_patch.placement.body_offset);
        const auto ret =
            planned_patch.entry_island_offset
                ? std::optional<int16_t>()
                : compute_sopp_branch_simm16(planned_patch.placement.return_branch_offset,
                                             planned_patch.placement.return_target);
        if (!write_moi_appended_anchor(new_text, planned_patch, entry_target, arch,
                                       "first-light probe", result.errors) ||
            (!planned_patch.entry_island_offset && !ret)) {
          result.errors.emplace_back(
              "ConSan MOI first-light probe appended cave branch is out of range");
          return;
        }

        if (planned_patch.entry_island_offset) {
          const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(options);
          std::vector<uint32_t> island_words;
          if (!append_moi_scc_preserving_indirect_jump(
                  island_words, *planned_patch.entry_island_offset, cave_text_offset,
                  *options.moi_exec_save_sgpr, special_state.scc_save_sgpr,
                  /*capture_scc=*/true, arch) ||
              island_words.size() != kIndirectIslandWords) {
            result.errors.emplace_back(
                "ConSan MOI first-light probe could not encode its entry island");
            return;
          }
          if (!write_moi_entry_island(new_text, planned_patch, island_words, patches,
                                      "first-light probe", result.errors))
            return;
        }

        auto cave_words =
            assemble_moi_appended_body(planned_patch, *words, "first-light probe", result.errors);
        if (!cave_words)
          return;
        if (planned_patch.entry_island_offset) {
          const MoiSpecialStateSgprs special_state = *moi_special_state_sgprs(options);
          if (!append_moi_scc_preserving_indirect_jump(
                  *cave_words, cave_text_offset, planned_patch.placement.return_target,
                  *options.moi_exec_save_sgpr, special_state.scc_save_sgpr,
                  /*capture_scc=*/false, arch)) {
            result.errors.emplace_back(
                "ConSan MOI first-light probe could not encode its long return");
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
          result.errors.emplace_back("ConSan MOI first-light probe final patch size changed");
          return;
        }
        if (candidate.text_offset > new_text.size() ||
            patch_bytes > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI first-light probe exceeds .text");
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
      result.errors.emplace_back("ConSan MOI first-light probe could not grow .text");
      return;
    }
    result.elf_bytes = patcher.emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  if (!apply_inline_moi_access_patches(
          bytes, planned_patches, descriptor_requirements, scalar_requirements, "first-light probe",
          [&](const PlannedAccessRecordPatch &planned_patch) {
            ConSanOptions candidate_options = options;
            candidate_options.scratch_vgpr = planned_patch.resources.base;
            return build_first_light_access_record_words(
                bytes, *planned_patch.candidate, candidate_options, arch,
                planned_patch.record_index, planned_record_count, layout.access_record_capacity,
                result.errors);
          },
          [](const PlannedAccessRecordPatch &planned_patch) {
            const ConSanMoiCandidate &candidate = *planned_patch.candidate;
            ConSanPatchInfo info;
            info.kind = ConSanPatchKind::InlineMoiAccessRecordStore;
            info.anchor_offset = candidate.text_offset;
            info.trampoline_offset = candidate.text_offset + candidate.size;
            info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
            info.scratch_vgpr = planned_patch.resources.base;
            info.owner_descriptor_file_offsets =
                planned_patch.resources.owner_descriptor_file_offsets;
            return info;
          },
          [](std::vector<uint8_t> &, ConSanResult &) { return true; }, result))
    return;
}
