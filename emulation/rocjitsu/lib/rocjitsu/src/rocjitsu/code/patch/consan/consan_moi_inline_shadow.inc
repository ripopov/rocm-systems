void try_apply_inline_shadow_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   rj_code_arch_t arch, ConSanResult &result) {
  if (!options.moi_report_buffer_address)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    result.warnings.emplace_back("ConSan MOI inline-shadow probe currently supports only RDNA4");
    return;
  }

  const ConSanMoiReportBufferLayout layout = resolve_inline_report_layout(options);
  if (options.max_patches == 0)
    return;
  if (!options.moi_inline_workgroup_shadow &&
      layout.exact_shadow_entry_capacity < kConSanMoiInlineShadowConservativeExactShadowEntries) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe requires exact-shadow capacity for the full 64 KiB "
        "LDS address range");
    return;
  }
  if (!validate_inline_shadow_exec_save_sgpr(options, result.warnings))
    return;
  if (!options.moi_owner_vgpr && !options.automatic_moi_private_epoch) {
    result.warnings.emplace_back("ConSan MOI inline-shadow probe requires persistent owner state");
    return;
  }
  std::vector<const ConSanMoiCandidate *> candidates =
      find_inline_shadow_access_candidates(result, bytes);
  apply_test_kernel_filter(candidates, options);
  std::ranges::sort(candidates, [](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
    return lhs->text_offset < rhs->text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe found no supported native LDS or normalized group-flat "
        "load/store candidate");
    return;
  }

  struct PlannedInlineShadowPatch {
    const ConSanMoiCandidate *candidate = nullptr;
    DbiPatchPlacement placement;
    std::optional<uint64_t> entry_island_offset;
    bool entry_island_at_anchor = false;
    std::vector<uint32_t> displaced_tail_words;
    ResolvedMoiScratchPlan resources;
    ConSanMoiWorkgroupSources workgroup_sources;
    std::optional<VgprSpillSequence> spill;
    std::optional<uint32_t> private_epoch_offset;
    std::optional<uint32_t> private_owner_offset;
    std::optional<uint16_t> private_owner_shift;
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    uint32_t required_private_bytes = 0;
  };
  std::vector<PlannedInlineShadowPatch> planned_patches;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  DbiPatchPlacementPlanner placement_planner(arch, original_text_size);
  constexpr uint64_t kIndirectIslandWords = 8u;
  const bool has_function_candidate = std::ranges::any_of(
      candidates, [](const ConSanMoiCandidate *candidate) { return !candidate->in_kernel; });
  const bool use_indirect_appended =
      options.moi_exec_save_sgpr.has_value() && (candidates.size() >= 8u || has_function_candidate);
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

  std::span<const uint8_t> original_text;
  if (code_object.text_sections().size() == 1) {
    const Section *text = code_object.text_sections().front();
    original_text =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
  }
  std::optional<MoiLocalNopIslandAllocator> local_entry_islands;
  if (use_indirect_appended && !original_text.empty())
    local_entry_islands.emplace(original_text, result, arch, kIndirectIslandWords);

  MoiSpillManagers spill_managers;
  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const uint16_t scratch_count = inline_shadow_scratch_count(options, candidate);
    const auto resources = resolve_moi_scratch(result, candidate, options, scratch_count);
    if (!resources)
      continue;

    std::optional<MoiPrivateEpochLayout> private_layout;
    std::optional<uint16_t> private_owner_shift;
    if (options.automatic_moi_private_epoch) {
      const bool persist_workitem_owner =
          options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId;
      private_layout =
          build_moi_private_epoch_layout(result, *resources, result.warnings,
                                         /*include_sample_sequence=*/false, persist_workitem_owner);
      if (!private_layout)
        continue;
      if (options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
        private_owner_shift = common_moi_private_owner_shift(bytes, *resources, result.warnings);
        if (!private_owner_shift)
          continue;
      }
    }

    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    if (options.moi_inline_workgroup_shadow) {
      workgroup_shadow = common_moi_workgroup_shadow_layout(bytes, *resources, result.warnings);
      if (!workgroup_shadow) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow access falls back to the external exact-shadow table");
      }
    }

    std::optional<uint64_t> workgroup_descriptor = candidate.kernel_descriptor_file_offset;
    if (!workgroup_descriptor) {
      workgroup_descriptor =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
    }
    if (!workgroup_descriptor)
      continue;
    std::vector<std::string> workgroup_errors;
    const auto workgroup_sources =
        moi_descriptor_workgroup_sources(bytes, *workgroup_descriptor, arch, workgroup_errors);
    if (!workgroup_sources) {
      result.warnings.insert(result.warnings.end(), workgroup_errors.begin(),
                             workgroup_errors.end());
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

    ConSanOptions candidate_options = options;
    candidate_options.moi_inline_workgroup_shadow = workgroup_shadow.has_value();
    candidate_options.scratch_vgpr = resources->base;
    std::vector<std::string> candidate_errors;
    auto words = build_inline_shadow_words(
        bytes, candidate, candidate_options, arch, layout, *workgroup_sources, workgroup_shadow,
        private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        private_layout ? private_layout->owner_offset : std::nullopt, private_owner_shift,
        candidate_errors);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t spill_bytes =
        spill ? static_cast<uint64_t>((spill->save_words.size() + spill->restore_words.size()) *
                                      sizeof(uint32_t))
              : 0;
    const uint32_t required_private_bytes =
        std::max(private_layout ? private_layout->ephemeral_base : 0u,
                 spill ? spill->total_private_bytes : 0u);
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size, arch);
    const uint64_t available_bytes =
        candidate.size + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    std::optional<DbiPatchPlacement> placement;
    std::optional<uint64_t> entry_island_offset;
    bool entry_island_at_anchor = false;
    std::vector<uint32_t> displaced_tail_words;
    uint32_t displaced_original_size = candidate.size;
    bool placed_through_indirect_island = false;
    if (use_indirect_appended) {
      uint64_t candidate_entry_island_offset =
          access_island_begin + planned_patches.size() * kIndirectIslandWords * sizeof(uint32_t);
      bool entry_island_reachable =
          candidate.size >= sizeof(uint32_t) &&
          compute_sopp_branch_simm16(candidate.text_offset, candidate_entry_island_offset);
      if (!entry_island_reachable) {
        if (const auto local_island =
                local_entry_islands ? local_entry_islands->claim_reachable(candidate.text_offset)
                                    : std::nullopt) {
          candidate_entry_island_offset = *local_island;
          entry_island_reachable = true;
        }
      }
      if (!entry_island_reachable && !original_text.empty()) {
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
            candidate_entry_island_offset = candidate.text_offset;
            entry_island_reachable = true;
            entry_island_at_anchor = true;
          }
        }
      }
      if (entry_island_reachable) {
        entry_island_offset = candidate_entry_island_offset;
        const uint64_t patch_bytes =
            static_cast<uint64_t>((words->size() + displaced_tail_words.size()) *
                                  sizeof(uint32_t)) +
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
        // The long return is getpc, a three-word signed delta builder, SCC
        // restore, a scalar-destination dependency wait, and setpc. The fixed
        // reservation keeps later body addresses stable while their
        // instruction words are regenerated during emission.
        indirect_body_cursor += patch_bytes + 7u * sizeof(uint32_t);
        placed_through_indirect_island = true;
      }
    }
    if (!placed_through_indirect_island) {
      const uint64_t patch_bytes =
          static_cast<uint64_t>(words->size() * sizeof(uint32_t)) + spill_bytes;
      DbiPatchPlacementRequest placement_request;
      placement_request.anchor_offset = candidate.text_offset;
      placement_request.original_size = candidate.size;
      placement_request.body_size = patch_bytes;
      placement_request.inline_capacity =
          spill || private_layout || workgroup_shadow ? 0u : available_bytes;
      std::string placement_error;
      placement = placement_planner.plan(placement_request, &placement_error);
      if (!placement) {
        result.warnings.emplace_back("ConSan MOI inline-shadow probe skipped access site: " +
                                     placement_error);
        continue;
      }
    }

    PlannedInlineShadowPatch planned;
    planned.candidate = candidate_ptr;
    planned.placement = *placement;
    planned.entry_island_offset = entry_island_offset;
    planned.entry_island_at_anchor = entry_island_at_anchor;
    planned.displaced_tail_words = std::move(displaced_tail_words);
    planned.resources = *resources;
    planned.workgroup_sources = *workgroup_sources;
    planned.spill = spill;
    if (private_layout)
      planned.private_epoch_offset = private_layout->epoch_offset;
    if (private_layout)
      planned.private_owner_offset = private_layout->owner_offset;
    planned.private_owner_shift = private_owner_shift;
    planned.workgroup_shadow = workgroup_shadow;
    planned.required_private_bytes = required_private_bytes;
    planned_patches.push_back(std::move(planned));
    if (planned_patches.size() == options.max_patches)
      break;
  }

  if (planned_patches.empty())
    return;

  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  MoiDescriptorLdsRequirements lds_requirements;
  for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
    note_descriptor_requirements(descriptor_requirements, planned_patch.resources);
    note_moi_sgpr_requirements(scalar_requirements, planned_patch.resources, options);
    if (planned_patch.workgroup_shadow) {
      note_moi_lds_requirements(lds_requirements, planned_patch.resources,
                                *planned_patch.workgroup_shadow);
    }
    if (planned_patch.required_private_bytes != 0) {
      for (uint64_t descriptor_offset : planned_patch.resources.owner_descriptor_file_offsets) {
        auto [it, inserted] =
            private_requirements.emplace(descriptor_offset, planned_patch.required_private_bytes);
        if (!inserted)
          it->second = std::max(it->second, planned_patch.required_private_bytes);
      }
    }
  }

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedInlineShadowPatch &patch) {
        return patch.placement.kind == DbiPatchPlacementKind::AppendedCave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_descriptor_requirements(patcher, code_object, bytes, result, descriptor_requirements,
                                       result.errors))
      return;
    if (!apply_spill_descriptor_requirements(patcher, code_object, bytes, result,
                                             private_requirements, result.errors))
      return;
    if (!apply_sgpr_descriptor_requirements(patcher, result, code_object, scalar_requirements,
                                            result.errors))
      return;
    if (!apply_lds_descriptor_requirements(patcher, result, code_object, lds_requirements,
                                           result.errors))
      return;
    const std::span<const uint8_t> old_text = patcher.text_bytes();
    if (old_text.empty()) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe found no .text section");
      return;
    }
    std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
    if (use_indirect_appended) {
      while (new_text.size() < indirect_body_begin)
        append_word_bytes(new_text, build_s_nop(0, arch));
    }
    std::vector<ConSanPatchInfo> patches;
    patches.reserve(planned_patches.size());
    for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      ConSanOptions candidate_options = options;
      candidate_options.moi_inline_workgroup_shadow = planned_patch.workgroup_shadow.has_value();
      candidate_options.scratch_vgpr = planned_patch.resources.base;
      auto words = build_inline_shadow_words(
          bytes, candidate, candidate_options, arch, layout, planned_patch.workgroup_sources,
          planned_patch.workgroup_shadow, planned_patch.private_epoch_offset,
          planned_patch.private_owner_offset, planned_patch.private_owner_shift, result.errors);
      if (!words)
        return;

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiExactShadowStore
                      : ConSanPatchKind::InlineMoiExactShadowStore;
      info.anchor_offset = candidate.text_offset;
      info.scratch_vgpr = planned_patch.resources.base;
      info.owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets;
      info.persistent_epoch_private_offset = planned_patch.private_epoch_offset;
      info.persistent_owner_private_offset = planned_patch.private_owner_offset;
      if (planned_patch.private_epoch_offset) {
        info.persistent_private_state_end =
            planned_patch.private_owner_offset
                ? *planned_patch.private_owner_offset + SpillManager::kSlotBytes
                : *planned_patch.private_epoch_offset + SpillManager::kSlotBytes;
      }
      info.required_private_segment_size = planned_patch.required_private_bytes;
      if (planned_patch.workgroup_shadow) {
        info.workgroup_shadow_base = planned_patch.workgroup_shadow->base;
        info.workgroup_shadow_size = planned_patch.workgroup_shadow->size;
        info.required_group_segment_size =
            planned_patch.workgroup_shadow->required_group_segment_size;
      }
      if (planned_patch.spill)
        info.spilled_vgpr_count = planned_patch.spill->vgpr_count;

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe emitted a stale appended-cave mapping");
          return;
        }
        const uint64_t entry_target =
            planned_patch.entry_island_offset.value_or(planned_patch.placement.body_offset);
        if (!write_moi_appended_anchor(new_text, planned_patch, entry_target, arch,
                                       "inline-shadow probe", result.errors))
          return;

        if (planned_patch.entry_island_offset) {
          const uint16_t jump_pc_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 12u);
          const uint16_t saved_scc_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 10u);
          std::vector<uint32_t> island_words;
          if (!append_moi_scc_preserving_indirect_jump(
                  island_words, *planned_patch.entry_island_offset, cave_text_offset, jump_pc_sgpr,
                  saved_scc_sgpr, /*capture_scc=*/true, arch) ||
              island_words.empty()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its entry island");
            return;
          }
          const auto wait_pc = build_s_wait_alu_sa_sdst0(arch);
          if (!wait_pc) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its entry dependency wait");
            return;
          }
          island_words.insert(island_words.end() - 1, *wait_pc);
          if (island_words.size() != kIndirectIslandWords) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe encoded an invalid entry island size");
            return;
          }
          if (!write_moi_entry_island(new_text, planned_patch, island_words, patches,
                                      "inline-shadow probe", result.errors))
            return;
        }

        auto cave_words =
            assemble_moi_appended_body(planned_patch, *words, "inline-shadow probe", result.errors);
        if (!cave_words)
          return;
        if (planned_patch.entry_island_offset) {
          const uint16_t jump_pc_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 12u);
          const uint16_t saved_scc_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 10u);
          if (!append_moi_scc_preserving_indirect_jump(
                  *cave_words, cave_text_offset, planned_patch.placement.return_target,
                  jump_pc_sgpr, saved_scc_sgpr, /*capture_scc=*/false, arch)) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its long return");
            return;
          }
          const auto wait_pc = build_s_wait_alu_sa_sdst0(arch);
          if (!wait_pc) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its return dependency wait");
            return;
          }
          cave_words->insert(cave_words->end() - 1, *wait_pc);
        } else {
          const auto ret = compute_sopp_branch_simm16(planned_patch.placement.return_branch_offset,
                                                      planned_patch.placement.return_target);
          if (!ret) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe appended cave return is out of range");
            return;
          }
          cave_words->push_back(build_s_branch(*ret, arch));
        }
        append_words_bytes(new_text, *cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = planned_patch.placement.original_size;
        info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.placement.body_size) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe final patch size changed");
          return;
        }
        if (candidate.text_offset > new_text.size() ||
            patch_bytes > new_text.size() - candidate.text_offset) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe exceeds .text");
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
      result.errors.emplace_back("ConSan MOI inline-shadow probe could not grow .text");
      return;
    }
    result.elf_bytes = patcher.emit();
    result.patches.insert(result.patches.end(), patches.begin(), patches.end());
    result.modified = true;
    return;
  }

  if (!apply_inline_moi_access_patches(
          bytes, planned_patches, descriptor_requirements, scalar_requirements,
          "inline-shadow probe",
          [&](const PlannedInlineShadowPatch &planned_patch) {
            ConSanOptions candidate_options = options;
            candidate_options.moi_inline_workgroup_shadow =
                planned_patch.workgroup_shadow.has_value();
            candidate_options.scratch_vgpr = planned_patch.resources.base;
            return build_inline_shadow_words(
                bytes, *planned_patch.candidate, candidate_options, arch, layout,
                planned_patch.workgroup_sources, planned_patch.workgroup_shadow,
                planned_patch.private_epoch_offset, planned_patch.private_owner_offset,
                planned_patch.private_owner_shift, result.errors);
          },
          [](const PlannedInlineShadowPatch &planned_patch) {
            const ConSanMoiCandidate &candidate = *planned_patch.candidate;
            ConSanPatchInfo info;
            info.kind = ConSanPatchKind::InlineMoiExactShadowStore;
            info.anchor_offset = candidate.text_offset;
            info.trampoline_offset = candidate.text_offset + candidate.size;
            info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
            info.scratch_vgpr = planned_patch.resources.base;
            info.owner_descriptor_file_offsets =
                planned_patch.resources.owner_descriptor_file_offsets;
            if (planned_patch.workgroup_shadow) {
              info.workgroup_shadow_base = planned_patch.workgroup_shadow->base;
              info.workgroup_shadow_size = planned_patch.workgroup_shadow->size;
              info.required_group_segment_size =
                  planned_patch.workgroup_shadow->required_group_segment_size;
            }
            return info;
          },
          [&](std::vector<uint8_t> &image, ConSanResult &current_result) {
            return apply_lds_descriptor_requirements(image, current_result, lds_requirements,
                                                     current_result.errors);
          },
          result))
    return;
}

[[nodiscard]] bool append_sampled_banked_address(std::vector<uint32_t> &words,
                                                 uint64_t first_address, uint32_t element_size,
                                                 uint32_t bank_count, uint16_t bank_vgpr,
                                                 uint16_t address_vgpr, uint16_t temporary_vgpr,
                                                 rj_code_arch_t arch);

[[nodiscard]] bool append_direct_sampled_immediate_check(
    std::vector<uint32_t> &words, const ConSanOptions &options, rj_code_arch_t arch,
    uint32_t prior_record_index, uint32_t window_bank_count, uint16_t bank_vgpr,
    size_t sampled_watchpoints_offset, uint16_t current_low_vgpr, uint16_t current_high_vgpr,
    std::vector<std::string> &errors) {
  if (!options.moi_sampled_check)
    return true;
  if (!options.moi_report_buffer_address || !options.scratch_vgpr || !options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI sampled checker is missing report or scalar resources");
    return false;
  }

  const uint16_t address_lo_vgpr = *options.scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 1u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t prior_low_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  const uint16_t prior_high_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 6u);
  if (window_bank_count == 0) {
    errors.emplace_back("ConSan MOI sampled checker has an invalid prior bank");
    return false;
  }
  const uint64_t prior_address = *options.moi_report_buffer_address + sampled_watchpoints_offset +
                                 static_cast<uint64_t>(prior_record_index) * sizeof(uint64_t);
  InstructionSequence sequence(words);
  const auto restore_vcc_label = sequence.make_label();

  const auto save_vcc = build_s_mov_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  const auto zero_low = build_v_mov_b32_e64_literal(prior_low_vgpr, 0u, arch);
  const auto zero_high = build_v_mov_b32_e64_literal(prior_high_vgpr, 0u, arch);
  const auto atomic_snapshot =
      build_flat_atomic_add_u64_vaddr_vsrc_vdst(address_lo_vgpr, prior_low_vgpr, prior_low_vgpr,
                                                /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  if (!save_vcc || !zero_low || !zero_high || !atomic_snapshot ||
      !append_sampled_banked_address(words, prior_address, sizeof(uint64_t), window_bank_count,
                                     bank_vgpr, address_lo_vgpr, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not atomically snapshot the prior slot");
    return false;
  }
  words.insert(words.end(), zero_low->begin(), zero_low->end());
  words.insert(words.end(), zero_high->begin(), zero_high->end());
  words.insert(words.end(), atomic_snapshot->begin(), atomic_snapshot->end());
  words.push_back(kWaitLoadcnt0);
  words.push_back(*save_vcc);

  auto append_required_predicate = [&](std::optional<uint32_t> compare) {
    if (!compare)
      return false;
    words.push_back(*compare);
    return sequence.emit_branch(restore_vcc_label, InstructionSequence::BranchKind::VccZero);
  };

  const auto prior_valid = build_v_and_b32_e32_literal(tmp_vgpr, 1u, prior_low_vgpr, arch);
  if (!prior_valid) {
    errors.emplace_back("ConSan MOI sampled checker could not decode validity");
    return false;
  }
  words.insert(words.end(), prior_valid->begin(), prior_valid->end());
  if (!append_required_predicate(
          build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch)) ||
      !append_extract_exact_shadow_field(words, address_lo_vgpr, current_low_vgpr,
                                         consan_moi_sampled_watchpoint::owner_shift,
                                         consan_moi_sampled_watchpoint::max_owner, arch) ||
      !append_extract_exact_shadow_field(words, address_hi_vgpr, prior_low_vgpr,
                                         consan_moi_sampled_watchpoint::owner_shift,
                                         consan_moi_sampled_watchpoint::max_owner, arch) ||
      !append_required_predicate(
          build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch))) {
    errors.emplace_back("ConSan MOI sampled checker could not compare owners");
    return false;
  }

  const uint32_t low_epoch_generation_mask =
      static_cast<uint32_t>(consan_moi_sampled_watchpoint::epoch_generation_mask);
  const uint32_t high_generation_mask =
      static_cast<uint32_t>(consan_moi_sampled_watchpoint::generation_mask >> 32u);
  const uint32_t high_range_mask = static_cast<uint32_t>(
      (consan_moi_sampled_watchpoint::start_mask | consan_moi_sampled_watchpoint::count_mask) >>
      32u);
  auto append_equal_masked = [&](uint16_t current_vgpr, uint16_t prior_vgpr, uint32_t mask) {
    const auto current = build_v_and_b32_e32_literal(address_lo_vgpr, mask, current_vgpr, arch);
    const auto prior = build_v_and_b32_e32_literal(address_hi_vgpr, mask, prior_vgpr, arch);
    if (!current || !prior)
      return false;
    words.insert(words.end(), current->begin(), current->end());
    words.insert(words.end(), prior->begin(), prior->end());
    return append_required_predicate(
        build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch));
  };
  if (!append_equal_masked(current_low_vgpr, prior_low_vgpr, low_epoch_generation_mask) ||
      !append_equal_masked(current_high_vgpr, prior_high_vgpr, high_generation_mask) ||
      !append_equal_masked(current_high_vgpr, prior_high_vgpr, high_range_mask)) {
    errors.emplace_back("ConSan MOI sampled checker could not compare ordering or ranges");
    return false;
  }

  if (!append_extract_exact_shadow_field(
          words, address_lo_vgpr, current_low_vgpr,
          consan_moi_sampled_watchpoint::access_kind_shift,
          (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, arch) ||
      !append_extract_exact_shadow_field(
          words, address_hi_vgpr, prior_low_vgpr, consan_moi_sampled_watchpoint::access_kind_shift,
          (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not decode access kinds");
    return false;
  }
  const auto both_read =
      build_v_and_b32_e32(tmp_vgpr, vector_source_vgpr(address_lo_vgpr), address_hi_vgpr, arch);
  if (!both_read) {
    errors.emplace_back("ConSan MOI sampled checker could not compare access kinds");
    return false;
  }
  words.push_back(*both_read);
  if (!append_required_predicate(build_v_cmp_ne_u32_e32_vcc(
          scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read)),
          tmp_vgpr, arch))) {
    errors.emplace_back("ConSan MOI sampled checker could not predicate conflicting kinds");
    return false;
  }

  const uint64_t immediate_conflict_count_address =
      *options.moi_report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter);
  if (!append_atomic_fetch_add_one_u32(words, immediate_conflict_count_address, tmp_vgpr,
                                       *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled checker could not increment the conflict counter");
    return false;
  }

  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, *options.moi_exec_save_sgpr, arch);
  if (!restore_vcc) {
    errors.emplace_back("ConSan MOI sampled checker could not restore VCC");
    return false;
  }
  if (!sequence.bind(restore_vcc_label))
    return false;
  words.push_back(*restore_vcc);
  if (!sequence.resolve_branches(arch)) {
    errors.emplace_back("ConSan MOI sampled checker skip branch is out of range");
    return false;
  }
  return true;
}

[[nodiscard]] bool
append_sampled_window_bank_index(std::vector<uint32_t> &words, const ConSanOptions &options,
                                 const ConSanMoiWorkgroupSources &workgroup_sources,
                                 uint32_t bank_count, uint16_t bank_vgpr, uint16_t temporary_vgpr,
                                 rj_code_arch_t arch) {
  if (bank_count == 0 || !std::has_single_bit(bank_count))
    return false;
  if (bank_count == 1) {
    const auto zero = build_v_mov_b32_e64_literal(bank_vgpr, 0, arch);
    if (!zero)
      return false;
    words.insert(words.end(), zero->begin(), zero->end());
    return true;
  }
  if (!options.moi_dispatch_id_sgpr)
    return false;
  words.push_back(build_v_mov_b32_e32(bank_vgpr, *options.moi_dispatch_id_sgpr, arch));
  words.push_back(build_v_mov_b32_e32(
      temporary_vgpr, static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u), arch));
  const auto mix = [&](uint16_t source) {
    const auto word = build_v_xor_b32_e32(bank_vgpr, vector_source_vgpr(source), bank_vgpr, arch);
    if (!word)
      return false;
    words.push_back(*word);
    return true;
  };
  if (!mix(temporary_vgpr))
    return false;
  for (const ConSanMoiWorkgroupSource &source :
       {workgroup_sources.x, workgroup_sources.y, workgroup_sources.z}) {
    if (!source.scalar_src)
      continue;
    if (!append_workgroup_source_value(words, source, temporary_vgpr, arch) || !mix(temporary_vgpr))
      return false;
  }
  const auto shift =
      build_v_lshrrev_b32_e32(temporary_vgpr, scalar_positive_inline_u32(16), bank_vgpr, arch);
  if (!shift)
    return false;
  words.push_back(*shift);
  if (!mix(temporary_vgpr))
    return false;
  const auto multiply = build_v_mul_lo_u32_vop3_literal(bank_vgpr, 0x85ebca6bu, bank_vgpr, arch);
  if (!multiply)
    return false;
  words.insert(words.end(), multiply->begin(), multiply->end());
  const auto mask = build_v_and_b32_e32_literal(bank_vgpr, bank_count - 1u, bank_vgpr, arch);
  if (!mask)
    return false;
  words.insert(words.end(), mask->begin(), mask->end());
  return true;
}

[[nodiscard]] bool append_sampled_banked_address(std::vector<uint32_t> &words,
                                                 uint64_t first_address, uint32_t element_size,
                                                 uint32_t bank_count, uint16_t bank_vgpr,
                                                 uint16_t address_vgpr, uint16_t temporary_vgpr,
                                                 rj_code_arch_t arch) {
  const auto low =
      build_v_mov_b32_e64_literal(address_vgpr, static_cast<uint32_t>(first_address), arch);
  const auto high = build_v_mov_b32_e64_literal(static_cast<uint16_t>(address_vgpr + 1u),
                                                static_cast<uint32_t>(first_address >> 32u), arch);
  if (!low || !high)
    return false;
  words.insert(words.end(), low->begin(), low->end());
  words.insert(words.end(), high->begin(), high->end());
  if (bank_count == 1)
    return true;
  const auto offset =
      build_v_mul_lo_u32_vop3_literal(temporary_vgpr, element_size, bank_vgpr, arch);
  const auto add = build_v_add_u64_vgpr_offset(address_vgpr, temporary_vgpr, arch);
  if (!offset || !add)
    return false;
  words.insert(words.end(), offset->begin(), offset->end());
  words.insert(words.end(), add->begin(), add->end());
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_range_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanMoiAccessRange &access_range, const ConSanOptions &options, rj_code_arch_t arch,
    uint32_t record_index, std::span<const uint32_t> prior_record_indices,
    uint32_t window_bank_count, bool relocate_instruction,
    std::optional<uint16_t> preserved_lds_byte_offset_vgpr, size_t sampled_causal_windows_offset,
    size_t sampled_watchpoints_offset, size_t sampled_pending_acquires_offset,
    std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI sampled probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  // A runtime fast-gate lowers the cave body with stride=1 after the outer
  // scalar gate has made the sampling decision.  Multi-bank publication still
  // needs its bank index to survive all ordinary temporaries in that cave.
  // Reserve the dedicated bank VGPR based on the emitted layout, not solely on
  // whether the inner vector sampler remains enabled.
  const bool dedicated_bank_vgpr = options.moi_runtime_sample_stride > 1 || window_bank_count > 1;
  const uint16_t base_scratch_count = static_cast<uint16_t>((options.moi_sampled_check ? 7u : 5u) +
                                                            (dedicated_bank_vgpr ? 1u : 0u));
  const uint16_t scratch_count = static_cast<uint16_t>(
      direct_sampled_scratch_count(options, candidate) +
      (window_bank_count > 1 && options.moi_runtime_sample_stride <= 1 ? 1u : 0u));
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back("ConSan MOI sampled probe exceeds the VGPR file");
    return std::nullopt;
  }
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (preserved_lds_byte_offset_vgpr)
    lds_byte_offset_vgpr = preserved_lds_byte_offset_vgpr;
  if (reject_candidate_scratch_range_overlap(candidate, *options.scratch_vgpr, scratch_count,
                                             errors))
    return std::nullopt;
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      (!options.automatic_moi_private_epoch &&
       reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                             scratch_count, "MOI epoch", errors)))
    return std::nullopt;
  const ConSanMoiLdsCellRange static_range =
      consan_moi_lds_cell_range_for_bytes(access_range.static_byte_offset, access_range.byte_count);
  std::optional<uint16_t> derived_owner_vgpr;
  std::optional<uint32_t> derived_owner_word;
  if (!options.moi_owner_vgpr && candidate.kernel_descriptor_file_offset) {
    const auto owner_shift =
        moi_descriptor_owner_shift(bytes, *candidate.kernel_descriptor_file_offset, errors);
    if (!owner_shift)
      return std::nullopt;
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    const auto owner_init = build_v_lshrrev_b32_e32(
        value_vgpr, scalar_positive_inline_u32(*owner_shift), kRdna4WorkitemIdX, arch);
    if (!owner_init) {
      errors.emplace_back("ConSan MOI sampled probe could not encode owner derivation");
      return std::nullopt;
    }
    derived_owner_vgpr = value_vgpr;
    derived_owner_word = *owner_init;
  }
  if (!candidate.kernel_descriptor_file_offset) {
    errors.emplace_back("ConSan MOI sampled probe requires exact workgroup-id sources");
    return std::nullopt;
  }
  const auto workgroup_sources = moi_descriptor_workgroup_sources(
      bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
  if (!workgroup_sources)
    return std::nullopt;

  const uint16_t low_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t high_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t owner_vgpr = options.moi_owner_vgpr.value_or(derived_owner_vgpr.value_or(0));
  const bool runtime_sampled = options.moi_runtime_sample_stride > 1;
  const uint16_t bank_vgpr =
      dedicated_bank_vgpr
          ? static_cast<uint16_t>(*options.scratch_vgpr + (options.moi_sampled_check ? 7u : 5u))
          : tmp_vgpr;
  if (runtime_sampled && !options.moi_owner_vgpr && !derived_owner_vgpr) {
    errors.emplace_back("ConSan MOI runtime sampled probe could not derive a wave owner");
    return std::nullopt;
  }
  if (runtime_sampled && !options.moi_dispatch_id_sgpr) {
    errors.emplace_back("ConSan MOI runtime sampled probe requires a hardware dispatch ID");
    return std::nullopt;
  }
  if (!options.moi_exec_save_sgpr || *options.moi_exec_save_sgpr > 98u ||
      *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back(
        "ConSan MOI sampled probe needs seven aligned VCC/EXEC/SCC-save SGPRs in 0..104");
    return std::nullopt;
  }
  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  const uint32_t generation =
      options.moi_report_generation & consan_moi_sampled_watchpoint::max_generation;
  const uint64_t generation_field = static_cast<uint64_t>(generation)
                                    << consan_moi_sampled_watchpoint::generation_shift;
  const uint32_t low_literal = static_cast<uint32_t>(
      consan_moi_sampled_watchpoint::valid_mask |
      (static_cast<uint64_t>(kind) << consan_moi_sampled_watchpoint::access_kind_shift) |
      generation_field);
  const uint32_t encoded_cell_count = encode_consan_moi_sampled_cell_count(static_range.cell_count)
                                      << (consan_moi_sampled_watchpoint::count_shift - 32u);
  const uint32_t encoded_generation_high = static_cast<uint32_t>(generation_field >> 32u);
  const uint64_t sampled_entry_address = *options.moi_report_buffer_address +
                                         sampled_watchpoints_offset +
                                         static_cast<uint64_t>(record_index) * sizeof(uint64_t);
  const uint64_t pending_acquire_address =
      *options.moi_report_buffer_address + sampled_pending_acquires_offset +
      static_cast<uint64_t>(record_index) * sizeof(ConSanMoiSampledPendingAcquireSlot);

  std::vector<uint32_t> words;
  words.reserve(candidate.size / sizeof(uint32_t) + 40u + (derived_owner_word ? 1u : 0u) +
                (options.moi_epoch_vgpr ? 2u : 0u) +
                (relocate_instruction && moi_load_clobbers_address(candidate) &&
                         !preserved_lds_byte_offset_vgpr
                     ? 1u
                     : 0u) +
                (access_range.static_byte_offset != 0u ? 3u : 0u));
  InstructionSequence sequence(words);
  const auto collision_label = sequence.make_label();
  const auto different_identity_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  ConSanMoiRecordEmitter record(words, *options.scratch_vgpr, tmp_vgpr, arch);
  // A DS load may overwrite its own address VGPR. Snapshot that address before
  // executing the displaced instruction so sampled publication remains tied
  // to the guest access rather than to the loaded payload.
  if (relocate_instruction && moi_load_clobbers_address(candidate) &&
      !preserved_lds_byte_offset_vgpr) {
    const uint16_t saved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(*options.scratch_vgpr + base_scratch_count);
    words.push_back(build_v_mov_b32_e32(saved_lds_byte_offset_vgpr,
                                        vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    lds_byte_offset_vgpr = saved_lds_byte_offset_vgpr;
  }
  if (derived_owner_word)
    words.push_back(*derived_owner_word);
  if (relocate_instruction) {
    for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
      words.push_back(word);
    }
    words.push_back(kWaitDscnt0);
  }

  const uint16_t original_exec_save_sgpr = *options.moi_exec_save_sgpr;
  const uint16_t selection_vcc_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u);
  const uint16_t publication_exec_save_sgpr =
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u);
  const uint16_t publication_scc_save_sgpr =
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
  const auto save_scc =
      build_rdna4_s_cselect_b32(publication_scc_save_sgpr, scalar_positive_inline_u32(1),
                                scalar_positive_inline_u32(0), arch);
  const auto save_vcc = build_s_mov_b64(selection_vcc_save_sgpr, kRdna4VccLo, arch);
  const auto save_exec = build_s_mov_b64(original_exec_save_sgpr, kRdna4ExecLo, arch);
  if (!save_scc || !save_vcc || !save_exec) {
    errors.emplace_back("ConSan MOI sampled probe could not save guest EXEC/VCC/SCC");
    return std::nullopt;
  }
  // Publication has several nested EXEC-narrowing paths. Some paths reuse the
  // publication-save pair after an earlier narrowing, so that pair is not a
  // reliable copy of the guest mask at the common exit. Keep the otherwise
  // unused leading pair as the immutable guest EXEC snapshot.
  words.push_back(*save_exec);
  words.push_back(*save_scc);
  words.push_back(*save_vcc);
  if (runtime_sampled) {
    if (access_range.static_byte_offset != 0u) {
      if (!append_compute_effective_lds_byte_offset(words, low_vgpr, *lds_byte_offset_vgpr,
                                                    access_range.static_byte_offset, arch))
        return std::nullopt;
    } else {
      words.push_back(
          build_v_mov_b32_e32(low_vgpr, vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    }
    const auto cell = build_v_lshrrev_b32_e32(
        low_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::granule_shift),
        low_vgpr, arch);
    const auto residue = build_v_and_b32_e32_literal(
        low_vgpr, options.moi_runtime_sample_stride - 1u, low_vgpr, arch);
    const auto selected_value =
        build_v_mov_b32_e64_literal(high_vgpr, options.moi_runtime_sample_offset, arch);
    const auto selected = build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(high_vgpr), low_vgpr, arch);
    const auto narrow_selected =
        build_s_and_saveexec_b64(publication_exec_save_sgpr, kRdna4VccLo, arch);
    if (!cell || !residue || !selected_value || !selected || !narrow_selected) {
      errors.emplace_back("ConSan MOI runtime sampled probe could not encode LDS-cell selector");
      return std::nullopt;
    }
    words.push_back(*cell);
    words.insert(words.end(), residue->begin(), residue->end());
    words.insert(words.end(), selected_value->begin(), selected_value->end());
    words.push_back(*selected);
    if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    words.push_back(*narrow_selected);
  }
  if (runtime_sampled || window_bank_count > 1) {
    if (!append_sampled_window_bank_index(words, options, *workgroup_sources, window_bank_count,
                                          bank_vgpr, high_vgpr, arch)) {
      errors.emplace_back("ConSan MOI sampled probe could not select a window bank");
      return std::nullopt;
    }
  } else {
    const auto zero_bank = build_v_mov_b32_e64_literal(bank_vgpr, 0, arch);
    if (!zero_bank)
      return std::nullopt;
    words.insert(words.end(), zero_bank->begin(), zero_bank->end());
  }
  if (!append_moi_delay_words(words, arch, options, errors, "ConSan MOI sampled probe"))
    return std::nullopt;

  // A ready deferred acquire deterministically selects the wave which may
  // publish its associated later read. Empty slots are ordinary reads. Odd,
  // malformed, or other-wave slots take the existing fail-closed drop path,
  // preventing independent first-publisher races from joining unrelated
  // atomic and LDS evidence.
  if (kind == ConSanMoiShadowAccessKind::Read) {
    const auto pending_gate_done_label = sequence.make_label();
    if (!append_sampled_banked_address(
            words, pending_acquire_address, sizeof(ConSanMoiSampledPendingAcquireSlot),
            window_bank_count, bank_vgpr, *options.scratch_vgpr, high_vgpr, arch))
      return std::nullopt;
    if (!append_load_u32_vgpr_at_offset(words, *options.scratch_vgpr,
                                        offsetof(ConSanMoiSampledPendingAcquireSlot, version),
                                        low_vgpr, arch))
      return std::nullopt;
    const auto empty = build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), low_vgpr, arch);
    const auto enter_gate = build_s_cbranch_vccz(1, arch);
    if (!empty || !enter_gate)
      return std::nullopt;
    words.push_back(*empty);
    words.push_back(*enter_gate);
    if (!sequence.emit_branch(pending_gate_done_label,
                              InstructionSequence::BranchKind::Unconditional))
      return std::nullopt;

    const auto ready = build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(2), low_vgpr, arch);
    if (!ready)
      return std::nullopt;
    words.push_back(*ready);
    if (!sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    if (!append_load_u32_vgpr_at_offset(words, *options.scratch_vgpr,
                                        offsetof(ConSanMoiSampledPendingAcquireSlot, owner_id),
                                        high_vgpr, arch))
      return std::nullopt;
    const auto owner_equal =
        build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(owner_vgpr), high_vgpr, arch);
    const auto narrow_owner =
        build_s_and_saveexec_b64(publication_exec_save_sgpr, kRdna4VccLo, arch);
    if (!owner_equal || !narrow_owner)
      return std::nullopt;
    words.push_back(*owner_equal);
    if (!sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    words.push_back(*narrow_owner);
    if (!sequence.bind(pending_gate_done_label))
      return std::nullopt;
  }

  const uint64_t causal_window_address =
      *options.moi_report_buffer_address + sampled_causal_windows_offset +
      static_cast<uint64_t>(record_index) * sizeof(ConSanMoiSampledCausalWindow);
  const auto claim_new = build_v_mov_b32_e64_literal(
      low_vgpr, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing), arch);
  const auto claim_compare = build_v_mov_b32_e64_literal(
      high_vgpr, static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Empty), arch);
  const auto claim = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      *options.scratch_vgpr, low_vgpr, low_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
      arch);
  const auto claimed =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(static_cast<uint32_t>(
                                     ConSanMoiSampledCausalPublicationState::Empty)),
                                 low_vgpr, arch);
  const auto narrow_winner =
      build_s_and_saveexec_b64(publication_exec_save_sgpr, kRdna4VccLo, arch);
  if (!claim_new || !claim_compare || !claim || !claimed || !narrow_winner ||
      !append_sampled_banked_address(
          words, causal_window_address + offsetof(ConSanMoiSampledCausalWindow, publication_state),
          sizeof(ConSanMoiSampledCausalWindow), window_bank_count, bank_vgpr, *options.scratch_vgpr,
          high_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not claim a causal window slot");
    return std::nullopt;
  }
  words.insert(words.end(), claim_new->begin(), claim_new->end());
  words.insert(words.end(), claim_compare->begin(), claim_compare->end());
  words.insert(words.end(), claim->begin(), claim->end());
  words.push_back(kWaitLoadcnt0);
  words.push_back(*claimed);
  if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::VccZero))
    return std::nullopt;
  words.push_back(*narrow_winner);

  if (!append_sampled_banked_address(words, causal_window_address,
                                     sizeof(ConSanMoiSampledCausalWindow), window_bank_count,
                                     bank_vgpr, *options.scratch_vgpr, high_vgpr, arch))
    return std::nullopt;
  const auto store_dynamic_record_index = [&]() {
    const auto dynamic_record_index =
        build_v_add_nc_u32_e32_literal(high_vgpr, record_index, bank_vgpr, arch);
    if (!dynamic_record_index)
      return false;
    words.insert(words.end(), dynamic_record_index->begin(), dynamic_record_index->end());
    return record.store_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), high_vgpr);
  };
  if (!record.store_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                            static_cast<uint32_t>(options.moi_report_generation)) ||
      !record.store_literal(offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
                            static_cast<uint32_t>(options.moi_report_generation >> 32u)) ||
      !(options.moi_dispatch_id_sgpr
            ? record.store_sgpr(offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                                *options.moi_dispatch_id_sgpr)
            : record.store_literal(offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                                   static_cast<uint32_t>(options.moi_report_dispatch_id))) ||
      !(options.moi_dispatch_id_sgpr
            ? record.store_sgpr(offsetof(ConSanMoiSampledCausalWindow, dispatch_id) + 4u,
                                static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u))
            : record.store_literal(offsetof(ConSanMoiSampledCausalWindow, dispatch_id) + 4u,
                                   static_cast<uint32_t>(options.moi_report_dispatch_id >> 32u))) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                              workgroup_sources->x,
                              ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                              workgroup_sources->y,
                              ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                              workgroup_sources->z,
                              ConSanMoiRecordEmitter::MissingWorkgroupSource::StoreZero) ||
      !(options.moi_epoch_vgpr
            ? record.store_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch),
                                *options.moi_epoch_vgpr)
            : record.store_literal(offsetof(ConSanMoiSampledCausalWindow, epoch), 0)) ||
      !store_dynamic_record_index() ||
      !record.store_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1) ||
      !record.store_literal(offsetof(ConSanMoiSampledCausalWindow, reserved), 0)) {
    errors.emplace_back("ConSan MOI sampled probe could not publish causal window metadata");
    return std::nullopt;
  }

  const auto mov_low = build_v_mov_b32_e64_literal(low_vgpr, low_literal, arch);
  if (!mov_low) {
    errors.emplace_back("ConSan MOI sampled probe could not encode sampled entry low word");
    return std::nullopt;
  }
  words.insert(words.end(), mov_low->begin(), mov_low->end());
  if ((options.moi_owner_vgpr || derived_owner_vgpr) &&
      !append_add_shifted_vgpr_field(words, low_vgpr, owner_vgpr,
                                     consan_moi_sampled_watchpoint::owner_shift,
                                     consan_moi_sampled_watchpoint::max_owner, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode owner field");
    return std::nullopt;
  }
  if (options.moi_epoch_vgpr &&
      !append_add_shifted_vgpr_field(words, low_vgpr, *options.moi_epoch_vgpr,
                                     consan_moi_sampled_watchpoint::epoch_shift,
                                     consan_moi_sampled_watchpoint::max_epoch, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode epoch field");
    return std::nullopt;
  }
  uint16_t effective_lds_byte_offset_vgpr = *lds_byte_offset_vgpr;
  if (access_range.static_byte_offset != 0u) {
    if (!append_compute_effective_lds_byte_offset(words, tmp_vgpr, *lds_byte_offset_vgpr,
                                                  access_range.static_byte_offset, arch)) {
      errors.emplace_back("ConSan MOI sampled probe could not encode effective LDS byte offset");
      return std::nullopt;
    }
    effective_lds_byte_offset_vgpr = tmp_vgpr;
  }
  const auto start_cell_shift = build_v_lshrrev_b32_e32(
      tmp_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::granule_shift),
      effective_lds_byte_offset_vgpr, arch);
  const auto high_from_start = build_v_lshlrev_b32_e32(
      high_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::start_shift - 32u),
      tmp_vgpr, arch);
  if (!start_cell_shift || !high_from_start) {
    errors.emplace_back("ConSan MOI sampled probe could not encode sampled entry start cell");
    return std::nullopt;
  }
  words.push_back(*start_cell_shift);
  words.push_back(*high_from_start);
  if (!append_add_literal_field(words, high_vgpr, encoded_cell_count | encoded_generation_high,
                                tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not encode cell-count field");
    return std::nullopt;
  }
  for (uint32_t prior_record_index : prior_record_indices) {
    if (!append_direct_sampled_immediate_check(
            words, options, arch, prior_record_index, window_bank_count, bank_vgpr,
            sampled_watchpoints_offset, low_vgpr, high_vgpr, errors))
      return std::nullopt;
  }
  // A site is shared by all selected waves. Publish the complete packed entry
  // with one device-scope atomic exchange so concurrent writers cannot leave a
  // low word from one wave paired with a high word from another.
  const auto atomic_publish = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      *options.scratch_vgpr, low_vgpr, low_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
      arch);
  if (!atomic_publish || !append_sampled_banked_address(
                             words, sampled_entry_address, sizeof(uint64_t), window_bank_count,
                             bank_vgpr, *options.scratch_vgpr, tmp_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not atomically publish sampled entry");
    return std::nullopt;
  }
  words.insert(words.end(), atomic_publish->begin(), atomic_publish->end());
  words.push_back(kWaitLoadcnt0);
  if (!append_sampled_banked_address(words, causal_window_address,
                                     sizeof(ConSanMoiSampledCausalWindow), window_bank_count,
                                     bank_vgpr, *options.scratch_vgpr, tmp_vgpr, arch))
    return std::nullopt;
  if (!record.store_literal(offsetof(ConSanMoiSampledCausalWindow, publication_state),
                            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready))) {
    errors.emplace_back("ConSan MOI sampled probe could not commit causal window metadata");
    return std::nullopt;
  }
  const uint64_t selected_count_address =
      *options.moi_report_buffer_address +
      offsetof(ConSanMoiReportHeader, sampled_causal_window_count);
  if (!append_atomic_fetch_add_one_u32(words, selected_count_address, tmp_vgpr,
                                       *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not count the claimed causal window");
    return std::nullopt;
  }
  const auto restore_winner_exec = build_s_mov_b64(kRdna4ExecLo, publication_exec_save_sgpr, arch);
  if (!restore_winner_exec)
    return std::nullopt;
  words.push_back(*restore_winner_exec);
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional))
    return std::nullopt;

  const auto save_collision_exec = build_s_mov_b64(publication_exec_save_sgpr, kRdna4ExecLo, arch);
  if (!save_collision_exec || !sequence.bind(collision_label))
    return std::nullopt;
  words.push_back(*save_collision_exec);

  // Repeated executions of the same selected causal window are idempotent:
  // the static slot already retains their representative sample. Only a
  // different exact dynamic identity occupying the slot is capacity loss.
  if (!append_sampled_banked_address(words, causal_window_address,
                                     sizeof(ConSanMoiSampledCausalWindow), window_bank_count,
                                     bank_vgpr, *options.scratch_vgpr, high_vgpr, arch))
    return std::nullopt;
  const auto reject_unless_equal_vgpr = [&](uint32_t offset, uint16_t expected_vgpr) -> bool {
    if (!record.load(offset, low_vgpr))
      return false;
    const auto equal =
        build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(expected_vgpr), low_vgpr, arch);
    if (!equal)
      return false;
    words.push_back(*equal);
    return sequence.emit_branch(different_identity_label, InstructionSequence::BranchKind::VccZero);
  };
  const auto reject_unless_equal_literal = [&](uint32_t offset, uint32_t literal) -> bool {
    const auto expected = build_v_mov_b32_e64_literal(high_vgpr, literal, arch);
    if (!expected)
      return false;
    words.insert(words.end(), expected->begin(), expected->end());
    return reject_unless_equal_vgpr(offset, high_vgpr);
  };
  const auto reject_unless_equal_sgpr = [&](uint32_t offset, uint16_t expected_sgpr) -> bool {
    words.push_back(build_v_mov_b32_e32(high_vgpr, expected_sgpr, arch));
    return reject_unless_equal_vgpr(offset, high_vgpr);
  };
  const auto reject_unless_equal_workgroup = [&](uint32_t offset,
                                                 const ConSanMoiWorkgroupSource &source) -> bool {
    if (!source.scalar_src)
      return reject_unless_equal_literal(offset, 0u);
    if (!append_workgroup_source_value(words, source, high_vgpr, arch))
      return false;
    return reject_unless_equal_vgpr(offset, high_vgpr);
  };
  const auto reject_unless_equal_dynamic_record_index = [&]() {
    const auto dynamic_record_index =
        build_v_add_nc_u32_e32_literal(high_vgpr, record_index, bank_vgpr, arch);
    if (!dynamic_record_index)
      return false;
    words.insert(words.end(), dynamic_record_index->begin(), dynamic_record_index->end());
    return reject_unless_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), high_vgpr);
  };
  if (!reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                                   static_cast<uint32_t>(options.moi_report_generation)) ||
      !reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
                                   static_cast<uint32_t>(options.moi_report_generation >> 32u)) ||
      !(options.moi_dispatch_id_sgpr
            ? reject_unless_equal_sgpr(offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                                       *options.moi_dispatch_id_sgpr)
            : reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                                          static_cast<uint32_t>(options.moi_report_dispatch_id))) ||
      !(options.moi_dispatch_id_sgpr
            ? reject_unless_equal_sgpr(offsetof(ConSanMoiSampledCausalWindow, dispatch_id) + 4u,
                                       static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u))
            : reject_unless_equal_literal(
                  offsetof(ConSanMoiSampledCausalWindow, dispatch_id) + 4u,
                  static_cast<uint32_t>(options.moi_report_dispatch_id >> 32u))) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                                     workgroup_sources->x) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                                     workgroup_sources->y) ||
      !reject_unless_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                                     workgroup_sources->z) ||
      !(options.moi_epoch_vgpr
            ? reject_unless_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch),
                                       *options.moi_epoch_vgpr)
            : reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, epoch), 0u)) ||
      !reject_unless_equal_dynamic_record_index() ||
      !reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1u) ||
      !reject_unless_equal_literal(
          offsetof(ConSanMoiSampledCausalWindow, publication_state),
          static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready)) ||
      !reject_unless_equal_literal(offsetof(ConSanMoiSampledCausalWindow, reserved), 0u)) {
    errors.emplace_back("ConSan MOI sampled probe could not compare a repeated causal identity");
    return std::nullopt;
  }
  const auto restore_duplicate_exec =
      build_s_mov_b64(kRdna4ExecLo, publication_exec_save_sgpr, arch);
  if (!restore_duplicate_exec)
    return std::nullopt;
  words.push_back(*restore_duplicate_exec);
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional))
    return std::nullopt;

  if (!sequence.bind(different_identity_label))
    return std::nullopt;
  const auto mbcnt_lo = build_v_mbcnt_lo_u32_b32(tmp_vgpr, publication_exec_save_sgpr,
                                                 scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi =
      build_v_mbcnt_hi_u32_b32(tmp_vgpr, static_cast<uint16_t>(publication_exec_save_sgpr + 1u),
                               vector_source_vgpr(tmp_vgpr), arch);
  const auto first_active_lane =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
  const auto narrow_collision =
      build_s_and_saveexec_b64(publication_exec_save_sgpr, kRdna4VccLo, arch);
  if (!mbcnt_lo || !mbcnt_hi || !first_active_lane || !narrow_collision) {
    errors.emplace_back("ConSan MOI sampled probe could not select a collision representative");
    return std::nullopt;
  }
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*narrow_collision);
  const uint64_t saturated_count_address =
      *options.moi_report_buffer_address +
      offsetof(ConSanMoiReportHeader, sampled_saturated_window_count);
  if (!append_atomic_fetch_add_one_u32(words, saturated_count_address, tmp_vgpr,
                                       *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled probe could not count a saturated causal window claim");
    return std::nullopt;
  }
  const auto restore_collision_exec =
      build_s_mov_b64(kRdna4ExecLo, publication_exec_save_sgpr, arch);
  if (!restore_collision_exec)
    return std::nullopt;
  words.push_back(*restore_collision_exec);

  if (!sequence.bind(restore_label))
    return std::nullopt;
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, selection_vcc_save_sgpr, arch);
  const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, original_exec_save_sgpr, arch);
  const auto restore_scc =
      build_rdna4_s_cmp_lg_u32(publication_scc_save_sgpr, scalar_positive_inline_u32(0), arch);
  if (!restore_exec || !restore_vcc || !restore_scc) {
    errors.emplace_back("ConSan MOI sampled probe could not restore guest EXEC/VCC/SCC");
    return std::nullopt;
  }
  words.push_back(*restore_exec);
  words.push_back(*restore_vcc);
  words.push_back(*restore_scc);
  if (!sequence.resolve_branches(arch)) {
    errors.emplace_back("ConSan MOI sampled probe local branch is out of range");
    return std::nullopt;
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, uint32_t first_record_index,
    std::optional<uint32_t> prior_first_record_index, uint32_t prior_access_range_count,
    uint32_t window_bank_count, size_t sampled_causal_windows_offset,
    size_t sampled_watchpoints_offset, size_t sampled_pending_acquires_offset,
    std::optional<uint32_t> private_epoch_offset,
    std::optional<uint32_t> private_sample_sequence_offset, std::vector<std::string> &errors) {
  const auto access_ranges = candidate_access_ranges(bytes, candidate);
  if (!access_ranges || access_ranges->empty()) {
    errors.emplace_back("ConSan MOI sampled probe requires at least one LDS access range");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  if (private_epoch_offset) {
    if (!options.automatic_moi_private_epoch || !options.moi_epoch_vgpr) {
      errors.emplace_back("ConSan MOI sampled probe has an invalid private epoch plan");
      return std::nullopt;
    }
    const auto load =
        build_address_free_scratch_load_b32(*options.moi_epoch_vgpr, *private_epoch_offset, arch);
    const auto wait = build_s_wait_loadcnt0(arch);
    if (!load || !wait) {
      errors.emplace_back("ConSan MOI sampled probe could not load private epoch state");
      return std::nullopt;
    }
    words.insert(words.end(), load->begin(), load->end());
    words.push_back(*wait);
  }
  if (private_sample_sequence_offset) {
    if (!options.moi_sample_sequence_vgpr) {
      errors.emplace_back("ConSan MOI sampled probe has no private sequence temporary");
      return std::nullopt;
    }
    const auto load = build_address_free_scratch_load_b32(*options.moi_sample_sequence_vgpr,
                                                          *private_sample_sequence_offset, arch);
    const auto wait = build_s_wait_loadcnt0(arch);
    if (!load || !wait)
      return std::nullopt;
    words.insert(words.end(), load->begin(), load->end());
    words.push_back(*wait);
  }
  std::optional<uint16_t> preserved_lds_byte_offset_vgpr;
  if (moi_load_clobbers_address(candidate)) {
    if (!options.scratch_vgpr)
      return std::nullopt;
    const uint16_t base_scratch_count = static_cast<uint16_t>(
        (options.moi_sampled_check ? 7u : 5u) +
        (options.moi_runtime_sample_stride > 1 || window_bank_count > 1 ? 1u : 0u));
    preserved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(*options.scratch_vgpr + base_scratch_count);
    const auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
    if (!lds_byte_offset_vgpr)
      return std::nullopt;
    words.push_back(build_v_mov_b32_e32(*preserved_lds_byte_offset_vgpr,
                                        vector_source_vgpr(*lds_byte_offset_vgpr), arch));
  }

  for (uint32_t range_index = 0; range_index < access_ranges->size(); ++range_index) {
    std::vector<uint32_t> prior_record_indices;
    if (prior_first_record_index) {
      prior_record_indices.reserve(prior_access_range_count);
      for (uint32_t prior_range_index = 0; prior_range_index < prior_access_range_count;
           ++prior_range_index) {
        prior_record_indices.push_back(*prior_first_record_index +
                                       prior_range_index * window_bank_count);
      }
    }
    auto range_words = build_direct_sampled_watchpoint_range_words(
        bytes, candidate, (*access_ranges)[range_index], options, arch,
        first_record_index + range_index * window_bank_count, prior_record_indices,
        window_bank_count, range_index == 0u, preserved_lds_byte_offset_vgpr,
        sampled_causal_windows_offset, sampled_watchpoints_offset, sampled_pending_acquires_offset,
        errors);
    if (!range_words)
      return std::nullopt;
    words.insert(words.end(), range_words->begin(), range_words->end());
  }
  if (private_sample_sequence_offset) {
    const auto store = build_address_free_scratch_store_b32(*options.moi_sample_sequence_vgpr,
                                                            *private_sample_sequence_offset, arch);
    const auto wait = build_s_wait_storecnt0(arch);
    if (!store || !wait)
      return std::nullopt;
    words.insert(words.end(), store->begin(), store->end());
    words.push_back(*wait);
  }
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_runtime_fast_gate_island_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, const ConSanMoiWorkgroupSources &workgroup_sources,
    [[maybe_unused]] uint32_t first_record_index, uint64_t island_text_offset,
    uint64_t cave_text_offset, uint64_t return_text_offset, uint64_t island_word_count,
    rj_code_arch_t arch) {
  if (!options.moi_exec_save_sgpr || !options.moi_dispatch_id_sgpr ||
      options.moi_runtime_sample_stride <= 1u)
    return std::nullopt;
  const uint16_t saved_scc = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u);
  const uint16_t quotient = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 5u);
  const uint16_t residue = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
  const uint16_t shift =
      scalar_positive_inline_u32(std::countr_zero(options.moi_runtime_sample_stride));
  const uint32_t mask = options.moi_runtime_sample_stride - 1u;
  const uint32_t selected_residue = options.moi_runtime_sample_offset & mask;

  std::vector<uint32_t> words;
  words.reserve(island_word_count);
  const auto save_scc_word = build_rdna4_s_cselect_b32(saved_scc, scalar_positive_inline_u32(1),
                                                       scalar_positive_inline_u32(0), arch);
  const auto mix_dispatch =
      build_s_sub_u32(residue, scalar_positive_inline_u32(0), *options.moi_dispatch_id_sgpr, arch);
  if (!save_scc_word || !mix_dispatch)
    return std::nullopt;
  words.push_back(*save_scc_word);
  words.push_back(*mix_dispatch);

  // Sample a dispatch/workgroup consistently at every site. A per-site sample
  // counter used to send producer and consumer sites into unrelated banks, so
  // the checker could never compare their evidence. Workgroup coordinates keep
  // the configured stride spatial without turning a selected dispatch into an
  // all-workgroups-at-once event.
  const auto append_workgroup_mix = [&](const ConSanMoiWorkgroupSource &source) -> bool {
    if (!source.scalar_src)
      return true;
    uint16_t coordinate = *source.scalar_src;
    if (source.mask_low_16) {
      words.push_back(build_s_lshl_b32(quotient, coordinate, scalar_positive_inline_u32(16), arch));
      words.push_back(build_s_lshr_b32(quotient, quotient, scalar_positive_inline_u32(16), arch));
      coordinate = quotient;
    } else if (source.shift_right_16) {
      words.push_back(build_s_lshr_b32(quotient, coordinate, scalar_positive_inline_u32(16), arch));
      coordinate = quotient;
    }
    const auto mix = build_s_sub_u32(residue, residue, coordinate, arch);
    if (!mix)
      return false;
    words.push_back(*mix);
    return true;
  };
  if (!append_workgroup_mix(workgroup_sources.x) || !append_workgroup_mix(workgroup_sources.y) ||
      !append_workgroup_mix(workgroup_sources.z))
    return std::nullopt;

  words.push_back(build_s_lshr_b32(quotient, residue, shift, arch));
  words.push_back(build_s_lshl_b32(quotient, quotient, shift, arch));
  const auto subtract = build_s_sub_u32(residue, residue, quotient, arch);
  const auto compare = build_s_cmp_eq_u32(residue, scalar_positive_inline_u32(0), arch);
  const auto restore_scc = build_rdna4_s_cmp_lg_u32(saved_scc, scalar_positive_inline_u32(0), arch);
  if (!subtract || !compare || !restore_scc)
    return std::nullopt;
  words.push_back(*subtract);
  for (uint32_t remaining = selected_residue; remaining != 0u;) {
    const uint32_t chunk = std::min<uint32_t>(remaining, 64u);
    const auto subtract_selected =
        build_s_sub_u32(residue, residue, scalar_positive_inline_u32(chunk), arch);
    if (!subtract_selected)
      return std::nullopt;
    words.push_back(*subtract_selected);
    remaining -= chunk;
  }
  words.push_back(*compare);
  const size_t selected_branch_index = words.size();
  words.push_back(build_s_nop(0, arch));
  words.push_back(*restore_scc);
  for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  const size_t return_branch_index = words.size();
  words.push_back(build_s_nop(0, arch));
  const size_t selected_index = words.size();
  const auto branch_delta = [](size_t branch, size_t target) -> std::optional<int16_t> {
    const int64_t delta = static_cast<int64_t>(target) - static_cast<int64_t>(branch) - 1;
    if (delta < std::numeric_limits<int16_t>::min() || delta > std::numeric_limits<int16_t>::max())
      return std::nullopt;
    return static_cast<int16_t>(delta);
  };
  const auto selected_offset = branch_delta(selected_branch_index, selected_index);
  const uint64_t return_branch_text_offset =
      island_text_offset + return_branch_index * sizeof(uint32_t);
  const auto return_offset =
      compute_sopp_branch_simm16(return_branch_text_offset, return_text_offset);
  if (!selected_offset || !return_offset)
    return std::nullopt;
  const auto select = build_s_cbranch_scc1(*selected_offset, arch);
  if (!select)
    return std::nullopt;
  words[selected_branch_index] = *select;
  words[return_branch_index] = build_s_branch(*return_offset, arch);
  if (!append_moi_scc_preserving_indirect_jump(words, island_text_offset, cave_text_offset,
                                               *options.moi_exec_save_sgpr, saved_scc,
                                               /*capture_scc=*/false, arch) ||
      words.size() > island_word_count)
    return std::nullopt;
  words.resize(island_word_count, build_s_nop(0, arch));
  return words;
}
