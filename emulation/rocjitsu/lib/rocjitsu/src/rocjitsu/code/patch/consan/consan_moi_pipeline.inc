void try_apply_fence_record_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                  rj_code_arch_t arch, ConSanResult &result) {
  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::RecordReplay)
    return;
  const bool has_eligible =
      std::ranges::any_of(result.moi_fence_candidates, &ConSanMoiFenceCandidate::eligible);
  if (!has_eligible)
    return;
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || !options.moi_report_buffer_address) {
    result.warnings.emplace_back(
        "ConSan MOI fence record patch requires RDNA4 and a MOI report buffer");
    return;
  }
  const ConSanMoiReportBufferLayout layout =
      resolve_record_replay_report_layout(options, options.moi_track_barriers,
                                          /*include_atomics=*/true, /*include_fences=*/true);
  const uint32_t max_candidates = std::min(options.max_patches, layout.fence_record_capacity);
  if (max_candidates == 0) {
    result.warnings.emplace_back("ConSan MOI fence record patch has no fence record capacity");
    return;
  }

  std::vector<FenceRecordCandidate> candidates;
  append_fence_record_candidates(result, candidates);
  if (!options.test_kernel_name_filter.empty()) {
    std::erase_if(candidates, [&](const FenceRecordCandidate &candidate) {
      return candidate.fence.container_name.find(options.test_kernel_name_filter) ==
             std::string::npos;
    });
  }
  std::ranges::sort(candidates, {}, [](const FenceRecordCandidate &candidate) {
    return candidate.fence.text_offset;
  });
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI fence record patch rejected all qualified communication events");
    return;
  }

  std::span<const uint8_t> active_bytes = bytes;
  if (result.modified)
    active_bytes = std::span<const uint8_t>(result.elf_bytes.data(), result.elf_bytes.size());
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI fence record patch found no .text section");
    return;
  }
  struct PlannedFenceRecord {
    FenceRecordCandidate candidate;
    ResolvedMoiScratchPlan resources;
    ConSanMoiAtomicAddressPlan address_plan;
    std::optional<VgprSpillSequence> spill;
  };
  std::vector<PlannedFenceRecord> selected;
  MoiResourcePlanningState resource_state(bytes, arch, options, result);
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    if (selected.size() == max_candidates)
      break;
    FenceRecordCandidate candidate = candidates[candidate_index];
    const ConSanMoiFenceCandidate &fence = candidate.fence;
    if (fence.text_offset > old_text.size() || fence.size > old_text.size() - fence.text_offset) {
      result.errors.emplace_back("ConSan MOI fence record patch site is outside .text");
      return;
    }
    const bool overlaps = std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
      const uint64_t patch_end = patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1);
      return fence.text_offset < patch_end && patch.anchor_offset < fence.text_offset + fence.size;
    });
    if (overlaps) {
      result.warnings.emplace_back("ConSan MOI fence record patch skipped an overlapping patch");
      continue;
    }
    const uint16_t scratch_count = inline_atomic_scratch_count(candidate.communication_site);
    const ConSanCandidateResourcePlan *existing_plan =
        resource_plan_for_site(result, ConSanResourceSiteKind::Fence, fence.text_offset);
    ConSanCandidateResourcePlan plan =
        existing_plan
            ? *existing_plan
            : plan_moi_resource_site(resource_state, options, ConSanResourceSiteKind::Fence,
                                     candidate_index, fence.text_offset,
                                     candidate.kernel_descriptor_file_offset, scratch_count);
    if (!existing_plan)
      result.resource_plans.push_back(plan);
    const auto resources = resolve_moi_scratch_plan(plan, options, scratch_count);
    if (!resources) {
      result.warnings.emplace_back("ConSan MOI fence record patch has no usable resource plan");
      continue;
    }
    // The atomic probe has already grown .text. Spill-backed fence emission
    // needs a relocated private-window owner in that intermediate image; an
    // original-image spill plan is inventory evidence only and must not be
    // guessed across the second growth pass.
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired) {
      result.warnings.emplace_back(
          "ConSan MOI fence record patch does not support spill resources across the second "
          "text-growth pass");
      continue;
    }
    const ConSanMoiAtomicAddressPlan address_plan = plan_consan_moi_atomic_address(
        candidate.communication_site, resources->base, resources->count, resources->source, arch);
    if (!address_plan.supported()) {
      result.warnings.emplace_back(
          "ConSan MOI fence record patch rejected communication address: " +
          std::string(consan_moi_atomic_address_support_name(address_plan.support)));
      continue;
    }
    if (!options.moi_owner_vgpr && !candidate.kernel_descriptor_file_offset) {
      candidate.kernel_descriptor_file_offset =
          common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      if (!candidate.kernel_descriptor_file_offset)
        continue;
    }
    note_descriptor_requirements(descriptor_requirements, *resources);
    selected.push_back({std::move(candidate), *resources, address_plan, std::nullopt});
  }
  if (selected.empty()) {
    result.warnings.emplace_back("ConSan MOI fence record patch found no patchable fences");
    return;
  }
  if (!apply_descriptor_requirements(patcher, code_object, active_bytes, result,
                                     descriptor_requirements, result.errors) ||
      !apply_spill_descriptor_requirements(patcher, code_object, active_bytes, result,
                                           private_requirements, result.errors))
    return;

  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator indirect_allocator(old_text, result, arch,
                                                kMoiLocalIndirectIslandWords);
  std::vector<ConSanPatchInfo> patches;
  uint32_t record_index = 0;
  const uint32_t record_count = static_cast<uint32_t>(selected.size());
  for (const PlannedFenceRecord &planned : selected) {
    const ConSanMoiFenceCandidate &fence = planned.candidate.fence;
    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = fence.text_offset + fence.size;
    ConSanOptions candidate_options = options;
    candidate_options.scratch_vgpr = planned.resources.base;
    auto cave_words = build_fence_record_cave_words(
        active_bytes, planned.candidate, planned.address_plan, candidate_options,
        planned.spill ? &*planned.spill : nullptr, arch, record_index, record_count,
        layout.fence_records_offset, cave_text_offset, return_text_offset, result.errors);
    if (!cave_words)
      return;
    const bool direct_entry =
        compute_sopp_branch_simm16(fence.text_offset, cave_text_offset).has_value();
    const std::optional<uint64_t> indirect_entry_island =
        direct_entry ? std::nullopt : indirect_allocator.claim_reachable(fence.text_offset);
    if (!direct_entry && !indirect_entry_island) {
      result.warnings.emplace_back(
          "ConSan MOI fence record patch found no reachable indirect entry island in " +
          fence.container_name);
      continue;
    }
    std::string placement_error;
    const auto placement = placement_planner.plan_indirect_appended(
        fence.text_offset, fence.size, cave_words->size() * sizeof(uint32_t), &placement_error);
    if (!placement) {
      result.errors.emplace_back("ConSan MOI fence record placement failed: " + placement_error);
      return;
    }
    if (placement->body_offset != cave_text_offset || new_text.size() != placement->body_offset) {
      result.errors.emplace_back("ConSan MOI fence record patch has a stale cave mapping");
      return;
    }
    const uint64_t entry_target = indirect_entry_island.value_or(placement->body_offset);
    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, entry_target);
    if (!fwd) {
      result.errors.emplace_back("ConSan MOI fence record forward branch is out of range");
      return;
    }
    std::vector<uint32_t> anchor_words(fence.size / sizeof(uint32_t), build_s_nop(0, arch));
    if (anchor_words.empty()) {
      result.errors.emplace_back("ConSan MOI fence record patch empty anchor");
      return;
    }
    anchor_words.front() = build_s_branch(*fwd, arch);
    std::memcpy(new_text.data() + fence.text_offset, anchor_words.data(),
                anchor_words.size() * sizeof(uint32_t));
    if (indirect_entry_island &&
        !emit_moi_local_indirect_entry_island(
            new_text, *indirect_entry_island, placement->body_offset, fence.text_offset, options,
            planned.resources.owner_descriptor_file_offsets, arch, patches, result.errors,
            "ConSan MOI fence record patch")) {
      return;
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineMoiFenceRecord;
    info.anchor_offset = fence.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = fence.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    info.scratch_vgpr = planned.resources.base;
    info.owner_descriptor_file_offsets = planned.resources.owner_descriptor_file_offsets;
    if (planned.spill) {
      info.spilled_vgpr_count = planned.spill->vgpr_count;
      info.required_private_segment_size = planned.spill->total_private_bytes;
    }
    patches.push_back(std::move(info));
    ++record_index;
  }
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("ConSan MOI fence record patch could not grow .text");
    return;
  }
  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), patches.begin(), patches.end());
  result.modified = true;
}

void append_moi_operational_resource_plans(std::span<const uint8_t> bytes,
                                           const ConSanOptions &options, rj_code_arch_t arch,
                                           ConSanResult &result) {
  MoiResourcePlanningState state(bytes, arch, options, result);
  if (options.moi_track_barriers) {
    std::vector<BarrierRecordCandidate> candidates;
    for (const ConSanKernelInfo &kernel : result.kernels)
      append_barrier_epoch_candidates(kernel, candidates);
    for (const ConSanFunctionInfo &function : result.functions)
      append_barrier_epoch_candidates(function, candidates);
    apply_test_container_filter(candidates, options);
    for (size_t index = 0; index < candidates.size(); ++index) {
      const BarrierRecordCandidate &candidate = candidates[index];
      result.resource_plans.push_back(plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Barrier, index, candidate.site.text_offset,
          candidate.kernel_descriptor_file_offset,
          /*scratch_count=*/options.moi_engine == ConSanMoiEngine::Sampled
              ? (options.automatic_moi_private_epoch ? 9u : 7u)
          : options.moi_engine == ConSanMoiEngine::InlineShadow ? 1u
                                                                : 6u));
    }
  }
  if (options.moi_track_atomics) {
    std::vector<AtomicRecordCandidate> candidates;
    if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
      for (const ConSanKernelInfo &kernel : result.kernels)
        append_inline_atomic_candidates(kernel, result, candidates);
      for (const ConSanFunctionInfo &function : result.functions)
        append_inline_atomic_candidates(function, result, candidates);
      candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                      [](const auto &candidate) {
                                        return classify_consan_moi_inline_atomic_support(
                                                   candidate.site, candidate.event_kind) !=
                                               ConSanMoiInlineAtomicSupport::Supported;
                                      }),
                       candidates.end());
    } else {
      for (const ConSanKernelInfo &kernel : result.kernels)
        append_atomic_record_candidates(kernel, result, candidates);
      for (const ConSanFunctionInfo &function : result.functions)
        append_atomic_record_candidates(function, result, candidates);
    }
    apply_test_container_filter(candidates, options);
    for (size_t index = 0; index < candidates.size(); ++index) {
      const AtomicRecordCandidate &candidate = candidates[index];
      const uint16_t scratch_count =
          options.moi_engine == ConSanMoiEngine::Sampled ? 10u
          : options.moi_engine == ConSanMoiEngine::InlineShadow
              ? inline_shadow_atomic_scratch_count(candidate.site, candidate.event_kind)
              : inline_atomic_scratch_count(candidate.site);
      result.resource_plans.push_back(plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Atomic, index, candidate.site.text_offset,
          candidate.kernel_descriptor_file_offset, scratch_count));
    }
  }
  if (options.moi_track_atomics && options.moi_engine == ConSanMoiEngine::RecordReplay) {
    std::vector<FenceRecordCandidate> candidates;
    append_fence_record_candidates(result, candidates);
    if (!options.test_kernel_name_filter.empty()) {
      std::erase_if(candidates, [&](const FenceRecordCandidate &candidate) {
        return candidate.fence.container_name.find(options.test_kernel_name_filter) ==
               std::string::npos;
      });
    }
    for (size_t index = 0; index < candidates.size(); ++index) {
      const FenceRecordCandidate &candidate = candidates[index];
      result.resource_plans.push_back(plan_moi_resource_site(
          state, options, ConSanResourceSiteKind::Fence, index, candidate.fence.text_offset,
          candidate.kernel_descriptor_file_offset,
          inline_atomic_scratch_count(candidate.communication_site)));
    }
  }
}

void append_moi_site_disposition(ConSanResourceSiteKind kind, std::string_view container_name,
                                 bool in_kernel, uint64_t text_offset, std::string_view mnemonic,
                                 ConSanSiteDispositionReason reason, ConSanResult &result) {
  ConSanSiteDispositionRecord record;
  record.site_kind = kind;
  record.disposition = reason == ConSanSiteDispositionReason::None
                           ? ConSanSiteDisposition::Supported
                           : ConSanSiteDisposition::Unsupported;
  record.reason = reason;
  if (record.disposition == ConSanSiteDisposition::Supported) {
    record.lowering_outcome = ConSanSiteLoweringOutcome::Pending;
    record.lowering_reason = ConSanSiteLoweringReason::None;
  } else {
    record.lowering_outcome = ConSanSiteLoweringOutcome::Unsupported;
    record.lowering_reason = ConSanSiteLoweringReason::SemanticUnsupported;
  }
  record.container_name = container_name;
  record.in_kernel = in_kernel;
  record.text_offset = text_offset;
  record.mnemonic = mnemonic;
  result.site_dispositions.push_back(std::move(record));
}

[[nodiscard]] ConSanSiteDispositionReason
inline_atomic_disposition_reason(ConSanMoiInlineAtomicSupport support) {
  switch (support) {
  case ConSanMoiInlineAtomicSupport::Supported:
    return ConSanSiteDispositionReason::None;
  case ConSanMoiInlineAtomicSupport::NonFlatAddress:
    return ConSanSiteDispositionReason::NonFlatAtomicAddress;
  case ConSanMoiInlineAtomicSupport::NonzeroOffset:
    return ConSanSiteDispositionReason::NonzeroFlatOffset;
  case ConSanMoiInlineAtomicSupport::UnsupportedScope:
    return ConSanSiteDispositionReason::UnsupportedScope;
  case ConSanMoiInlineAtomicSupport::UnsupportedWidth:
    return ConSanSiteDispositionReason::InvalidAccessWidth;
  case ConSanMoiInlineAtomicSupport::CompareExchangeOutcomeUnavailable:
    return ConSanSiteDispositionReason::CompareExchangeOutcomeUnavailable;
  case ConSanMoiInlineAtomicSupport::UnsupportedEncoding:
    return ConSanSiteDispositionReason::UnsupportedAtomicEncoding;
  case ConSanMoiInlineAtomicSupport::MissingOperands:
    return ConSanSiteDispositionReason::MissingAtomicOperands;
  case ConSanMoiInlineAtomicSupport::MissingOrderingMetadata:
    return ConSanSiteDispositionReason::MissingOrderingMetadata;
  }
  return ConSanSiteDispositionReason::UnsupportedAtomicEncoding;
}

[[nodiscard]] ConSanSiteDispositionReason
sampled_atomic_disposition_reason(std::string_view reason) {
  if (reason == "missing-shared-sync-event" || reason == "unqualified-shared-sync-sequence")
    return ConSanSiteDispositionReason::UnqualifiedSyncSequence;
  if (reason == "unsupported-qualified-memory-role")
    return ConSanSiteDispositionReason::UnsupportedMemoryRole;
  if (reason == "missing-qualified-scope")
    return ConSanSiteDispositionReason::MissingScope;
  if (reason == "unsupported-qualified-scope")
    return ConSanSiteDispositionReason::UnsupportedScope;
  if (reason == "unsupported-qualified-byte-range")
    return ConSanSiteDispositionReason::InvalidAccessWidth;
  if (reason == "compare-exchange-dynamic-outcome-unavailable")
    return ConSanSiteDispositionReason::CompareExchangeOutcomeUnavailable;
  if (reason == "unsupported-qualified-rmw-outcome")
    return ConSanSiteDispositionReason::UnsupportedAtomicOutcome;
  return ConSanSiteDispositionReason::UnsupportedMetadataEncoding;
}

[[nodiscard]] ConSanSiteDispositionReason
first_light_atomic_disposition_reason(const ConSanAtomicSite &site) {
  if (!site.mnemonic.starts_with("flat_atomic") && !site.mnemonic.starts_with("global_atomic"))
    return ConSanSiteDispositionReason::NonFlatAtomicAddress;
  if (!site.addr_vgpr || !site.data_vgpr)
    return ConSanSiteDispositionReason::MissingAtomicOperands;
  if (*site.addr_vgpr >= 255)
    return ConSanSiteDispositionReason::ReservedFlatAddressRegister;
  if (!site.raw_scope || !site.raw_th)
    return ConSanSiteDispositionReason::MissingOrderingMetadata;
  const bool compare_exchange = site.mnemonic.find("cmpswap") != std::string::npos ||
                                site.mnemonic.find("cmpxchg") != std::string::npos;
  if (compare_exchange && !site.returns_old_value.value_or(false))
    return ConSanSiteDispositionReason::CompareExchangeOutcomeUnavailable;
  return ConSanSiteDispositionReason::None;
}

template <typename Container>
void append_moi_atomic_site_dispositions(const Container &container, const ConSanOptions &options,
                                         const ConSanResult &semantic_result,
                                         ConSanResult &result) {
  if (!options.moi_track_atomics ||
      (!options.test_kernel_name_filter.empty() &&
       container.name.find(options.test_kernel_name_filter) == std::string::npos))
    return;
  const bool in_kernel = std::is_same_v<Container, ConSanKernelInfo>;
  if (in_kernel && is_rocclr_runtime_kernel_name(container.name))
    return;
  for (const ConSanAtomicSite &site : container.atomic_sites) {
    const auto event =
        std::ranges::find_if(semantic_result.sync_events, [&](const ConSanSyncEvent &candidate) {
          return candidate.kind == ConSanSyncEventKind::Atomic &&
                 candidate.container_name == container.name && candidate.in_kernel == in_kernel &&
                 candidate.text_offset == site.text_offset;
        });
    const std::optional<ConSanMoiAtomicEventKind> event_kind =
        atomic_event_kind_for_site(semantic_result, container.name, in_kernel, site);
    ConSanSiteDispositionReason reason = ConSanSiteDispositionReason::None;
    if (event == semantic_result.sync_events.end() || event->execution_owners.empty()) {
      // Runtime-library helpers that no inventoried kernel can execute are
      // decoded inventory, but not relevant workload support sites. Keep the
      // operational candidate path stable until report sizing and persistent
      // state selection no longer depend on it; the coverage ledger must not
      // call an impossible MissingOwner plan supported.
      reason = ConSanSiteDispositionReason::UnqualifiedSyncSequence;
    } else if (!event_kind) {
      reason = ConSanSiteDispositionReason::UnqualifiedSyncSequence;
    } else if (options.moi_engine == ConSanMoiEngine::InlineShadow) {
      reason = inline_atomic_disposition_reason(
          classify_consan_moi_inline_atomic_support(site, *event_kind));
    } else {
      reason = first_light_atomic_disposition_reason(site);
      if (reason == ConSanSiteDispositionReason::None &&
          options.moi_engine == ConSanMoiEngine::Sampled) {
        std::string sampled_reason;
        if (!sampled_atomic_semantics_for_site(semantic_result, container.name, in_kernel, site,
                                               sampled_reason))
          reason = sampled_atomic_disposition_reason(sampled_reason);
      }
    }
    append_moi_site_disposition(ConSanResourceSiteKind::Atomic, container.name, in_kernel,
                                site.text_offset, site.mnemonic, reason, result);
  }
}

template <typename Container>
void append_moi_barrier_site_dispositions(const Container &container, const ConSanOptions &options,
                                          const ConSanResult &semantic_result,
                                          ConSanResult &result) {
  if (!options.moi_track_barriers ||
      (!options.test_kernel_name_filter.empty() &&
       container.name.find(options.test_kernel_name_filter) == std::string::npos))
    return;
  const bool in_kernel = std::is_same_v<Container, ConSanKernelInfo>;
  if (in_kernel && is_rocclr_runtime_kernel_name(container.name))
    return;
  for (const ConSanBarrierSite &site : container.barrier_sites) {
    if (options.moi_engine == ConSanMoiEngine::InlineShadow && !in_kernel &&
        (result.moi_persistent_vgprs_automatic || result.moi_private_epoch_automatic)) {
      const ConSanCandidateResourcePlan *plan =
          resource_plan_for_site(result, ConSanResourceSiteKind::Barrier, site.text_offset);
      if (plan == nullptr || plan->owner_descriptor_file_offsets.empty())
        continue;
    }
    ConSanSiteDispositionReason reason = ConSanSiteDispositionReason::None;
    if (options.moi_engine == ConSanMoiEngine::Sampled) {
      const ConSanSyncEvent *event = nullptr;
      size_t event_count = 0;
      for (const ConSanSyncEvent &candidate : semantic_result.sync_events) {
        if (candidate.kind != ConSanSyncEventKind::Barrier ||
            candidate.container_name != container.name || candidate.in_kernel != in_kernel ||
            candidate.text_offset != site.text_offset || candidate.size != site.size)
          continue;
        ++event_count;
        event = &candidate;
      }
      const bool qualified =
          event_count == 1u &&
          std::ranges::any_of(semantic_result.sync_sequences, [&](const auto &sequence) {
            return consan_moi_sampled_qualifies_barrier_sequence(sequence) &&
                   std::ranges::find(sequence.member_event_identities, event->identity) !=
                       sequence.member_event_identities.end();
          });
      if (!qualified)
        reason = ConSanSiteDispositionReason::UnqualifiedSyncSequence;
    } else if (site.size != sizeof(uint32_t)) {
      reason = ConSanSiteDispositionReason::InvalidBarrierEncoding;
    }
    append_moi_site_disposition(ConSanResourceSiteKind::Barrier, container.name, in_kernel,
                                site.text_offset, site.mnemonic, reason, result);
  }
}

void append_moi_sync_site_dispositions(const ConSanOptions &options, ConSanResult &result) {
  // Keep the semantic inventory immutable while appending to the disposition
  // vector that is part of the same result.
  const ConSanResult &semantic_result = result;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    append_moi_atomic_site_dispositions(kernel, options, semantic_result, result);
    append_moi_barrier_site_dispositions(kernel, options, semantic_result, result);
  }
  for (const ConSanFunctionInfo &function : result.functions) {
    append_moi_atomic_site_dispositions(function, options, semantic_result, result);
    append_moi_barrier_site_dispositions(function, options, semantic_result, result);
  }

  if (!options.moi_track_atomics || options.moi_engine != ConSanMoiEngine::RecordReplay)
    return;
  std::vector<FenceRecordCandidate> admitted;
  append_fence_record_candidates(result, admitted);
  for (const ConSanMoiFenceCandidate &fence : result.moi_fence_candidates) {
    if ((!options.test_kernel_name_filter.empty() &&
         fence.container_name.find(options.test_kernel_name_filter) == std::string::npos) ||
        (fence.in_kernel && is_rocclr_runtime_kernel_name(fence.container_name)))
      continue;
    ConSanSiteDispositionReason reason = ConSanSiteDispositionReason::None;
    const auto communication = std::ranges::find(
        result.sync_events, fence.communication_event_identity, &ConSanSyncEvent::identity);
    if (!fence.eligible) {
      reason = ConSanSiteDispositionReason::IneligibleFence;
    } else if (communication == result.sync_events.end() ||
               communication->execution_owners.empty()) {
      reason = ConSanSiteDispositionReason::MissingCommunicationEvent;
    } else if (std::ranges::none_of(admitted, [&](const FenceRecordCandidate &candidate) {
                 return candidate.fence.identity == fence.identity;
               })) {
      reason = ConSanSiteDispositionReason::MissingCommunicationEvent;
    }
    append_moi_site_disposition(ConSanResourceSiteKind::Fence, fence.container_name,
                                fence.in_kernel, fence.text_offset, "fence", reason, result);
  }
}

[[nodiscard]] std::optional<ConSanResourceSiteKind>
moi_instrumentation_patch_site_kind(const ConSanPatchInfo &patch, const ConSanResult &result) {
  switch (patch.kind) {
  case ConSanPatchKind::InlineMoiAccessRecordStore:
  case ConSanPatchKind::TrampolineMoiAccessRecordStore:
  case ConSanPatchKind::InlineMoiExactShadowStore:
  case ConSanPatchKind::TrampolineMoiExactShadowStore:
  case ConSanPatchKind::InlineMoiSampledWatchpointStore:
  case ConSanPatchKind::TrampolineMoiSampledWatchpointStore:
    return ConSanResourceSiteKind::Access;
  case ConSanPatchKind::TrampolineMoiBarrierRecord:
  case ConSanPatchKind::TrampolineMoiInlineEpochBarrier:
  case ConSanPatchKind::InlineMalformedBarrierAbort:
    return ConSanResourceSiteKind::Barrier;
  case ConSanPatchKind::TrampolineMoiInlineAtomicOrdering:
  case ConSanPatchKind::TrampolineMoiAtomicRecord:
    return ConSanResourceSiteKind::Atomic;
  case ConSanPatchKind::TrampolineMoiFenceRecord:
    return ConSanResourceSiteKind::Fence;
  case ConSanPatchKind::TrampolineMoiSampledSyncMetadata:
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      if (plan.text_offset == patch.anchor_offset &&
          (plan.site_kind == ConSanResourceSiteKind::Barrier ||
           plan.site_kind == ConSanResourceSiteKind::Atomic))
        return plan.site_kind;
    }
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] bool sampled_barrier_patch_covers_site(const ConSanSiteDispositionRecord &site,
                                                     const ConSanPatchInfo &patch,
                                                     const ConSanResult &result) {
  if (site.site_kind != ConSanResourceSiteKind::Barrier ||
      patch.kind != ConSanPatchKind::TrampolineMoiSampledSyncMetadata ||
      patch.covered_sync_event_count == 0)
    return false;
  const auto event = std::ranges::find_if(result.sync_events, [&](const ConSanSyncEvent &item) {
    return item.kind == ConSanSyncEventKind::Barrier &&
           item.container_name == site.container_name && item.in_kernel == site.in_kernel &&
           item.text_offset == site.text_offset;
  });
  if (event == result.sync_events.end())
    return false;
  const ConSanSyncSequence *sequence = find_consan_sync_sequence_for_event(result, event->identity);
  if (!sequence || sequence->member_event_identities.size() != patch.covered_sync_event_count)
    return false;
  return std::ranges::any_of(result.sync_events, [&](const ConSanSyncEvent &member) {
    return member.text_offset == patch.anchor_offset &&
           std::ranges::find(sequence->member_event_identities, member.identity) !=
               sequence->member_event_identities.end();
  });
}

void finalize_moi_site_lowering_outcomes(ConSanResult &result) {
  for (ConSanSiteDispositionRecord &site : result.site_dispositions) {
    if (site.disposition == ConSanSiteDisposition::NotApplicable) {
      site.lowering_outcome = ConSanSiteLoweringOutcome::NotApplicable;
      site.lowering_reason = ConSanSiteLoweringReason::SemanticNotApplicable;
      continue;
    }
    if (site.disposition == ConSanSiteDisposition::Unsupported) {
      site.lowering_outcome = ConSanSiteLoweringOutcome::Unsupported;
      site.lowering_reason = ConSanSiteLoweringReason::SemanticUnsupported;
      continue;
    }

    const bool patched = std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
      if (patch.phase != ConSanPatchPhase::Instrumentation)
        return false;
      const auto patch_kind = moi_instrumentation_patch_site_kind(patch, result);
      return (patch_kind && *patch_kind == site.site_kind &&
              patch.anchor_offset == site.text_offset) ||
             sampled_barrier_patch_covers_site(site, patch, result);
    });
    if (patched) {
      site.lowering_outcome = ConSanSiteLoweringOutcome::Patched;
      site.lowering_reason = ConSanSiteLoweringReason::None;
      site.resource_reason = ConSanRegisterPlanReason::None;
      continue;
    }

    const auto failed_plan =
        std::ranges::find_if(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
          return plan.site_kind == site.site_kind && plan.text_offset == site.text_offset &&
                 plan.source == ConSanRegisterAllocationSource::Unsupported;
        });
    if (failed_plan != result.resource_plans.end()) {
      site.lowering_outcome = ConSanSiteLoweringOutcome::ResourceFailed;
      site.lowering_reason = ConSanSiteLoweringReason::UnsupportedResourcePlan;
      site.resource_reason = failed_plan->reason;
      continue;
    }

    site.lowering_outcome = ConSanSiteLoweringOutcome::PlacementOrLoweringFailed;
    site.lowering_reason = ConSanSiteLoweringReason::InstrumentationPatchMissing;
    site.resource_reason = ConSanRegisterPlanReason::None;
  }
}

void rebuild_moi_resource_plans(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                rj_code_arch_t arch, ConSanResult &result) {
  append_moi_resource_plans(bytes, options, arch, result);
  append_moi_operational_resource_plans(bytes, options, arch, result);
}

void summarize_moi_resource_plans(ConSanResult &result) {
  result.resource_plan_summary = {};
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    switch (plan.source) {
    case ConSanRegisterAllocationSource::Explicit:
      ++result.resource_plan_summary.explicit_plans;
      break;
    case ConSanRegisterAllocationSource::LivenessDead:
      ++result.resource_plan_summary.dead_plans;
      break;
    case ConSanRegisterAllocationSource::DescriptorGrowth:
      ++result.resource_plan_summary.descriptor_growth_plans;
      break;
    case ConSanRegisterAllocationSource::SpillRequired:
      ++result.resource_plan_summary.spill_plans;
      result.resource_plan_summary.planned_spill_slot_bytes +=
          static_cast<size_t>(plan.scratch_vgpr_count) * SpillManager::kSlotBytes;
      break;
    case ConSanRegisterAllocationSource::Unsupported:
      ++result.resource_plan_summary.unsupported_plans;
      break;
    }
  }
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.spilled_vgpr_count == 0)
      continue;
    ++result.resource_plan_summary.emitted_spill_patches;
    result.resource_plan_summary.emitted_spill_slot_bytes +=
        static_cast<size_t>(patch.spilled_vgpr_count) * SpillManager::kSlotBytes;
  }
}

} // namespace
