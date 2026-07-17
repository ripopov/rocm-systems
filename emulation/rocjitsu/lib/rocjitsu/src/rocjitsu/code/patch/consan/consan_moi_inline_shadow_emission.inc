[[nodiscard]] bool append_inline_shadow_diagnostic_words(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate, const ConSanOptions &options,
    rj_code_arch_t arch, const ConSanMoiReportBufferLayout &layout, uint16_t old_value_vgpr,
    uint16_t old_value_hi_vgpr, uint16_t current_value_vgpr, uint16_t current_field_vgpr,
    uint16_t lds_byte_offset_vgpr, uint32_t static_byte_offset, uint32_t byte_count,
    bool special_state_already_saved = false,
    std::optional<uint16_t> diagnostic_lane_mask_sgpr = std::nullopt,
    bool capture_first_diagnostic_only = false) {
  if (!options.moi_exec_save_sgpr || layout.diagnostic_capacity == 0)
    return true;

  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint64_t report_base = *options.moi_report_buffer_address;
  const uint64_t diagnostic_base = report_base + layout.diagnostic_records_offset;

  if (!special_state_already_saved && !append_save_moi_special_state(words, options, arch))
    return false;

  const uint32_t mov_zero = build_v_mov_b32_e32(tmp_vgpr, scalar_positive_inline_u32(0), arch);
  const auto prior_nonempty =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(old_value_vgpr), tmp_vgpr, arch);
  const auto narrow_nonempty =
      build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  if (!prior_nonempty || !narrow_nonempty)
    return false;
  words.push_back(mov_zero);
  words.push_back(*prior_nonempty);
  words.push_back(*narrow_nonempty);

  const uint16_t address_lo_vgpr = *options.scratch_vgpr;
  if (!append_extract_exact_shadow_generation(words, tmp_vgpr, old_value_vgpr, old_value_hi_vgpr,
                                              address_lo_vgpr, arch) ||
      !append_extract_exact_shadow_generation(words, current_field_vgpr, current_value_vgpr,
                                              static_cast<uint16_t>(current_value_vgpr + 1u),
                                              address_lo_vgpr, arch)) {
    return false;
  }
  const auto same_workgroup =
      build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
  const auto narrow_same_workgroup = build_s_and_saveexec_b64(
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u), kRdna4VccLo, arch);
  if (!same_workgroup || !narrow_same_workgroup)
    return false;
  words.push_back(*same_workgroup);
  words.push_back(*narrow_same_workgroup);

  if (!append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch) ||
      !append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch)) {
    return false;
  }
  const auto owner_ne =
      build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
  const auto narrow_conflict = build_s_and_saveexec_b64(
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u), kRdna4VccLo, arch);
  if (!owner_ne || !narrow_conflict)
    return false;
  words.push_back(*owner_ne);
  words.push_back(*narrow_conflict);

  uint16_t next_exec_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u);
  const bool has_epoch = options.moi_epoch_vgpr || options.automatic_moi_private_epoch;
  if (has_epoch) {
    if (!append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch) ||
        !append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch)) {
      return false;
    }
    const auto epoch_eq =
        build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
    const auto narrow_same_epoch = build_s_and_saveexec_b64(next_exec_save_sgpr, kRdna4VccLo, arch);
    if (!epoch_eq || !narrow_same_epoch)
      return false;
    words.push_back(*epoch_eq);
    words.push_back(*narrow_same_epoch);
    next_exec_save_sgpr = static_cast<uint16_t>(next_exec_save_sgpr + 2u);
  }

  const auto current_kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  if (current_kind != ConSanMoiShadowAccessKind::Write) {
    const auto prior_kind = build_v_and_b32_e32_literal(
        tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), old_value_vgpr,
        arch);
    const auto kind_ne = build_v_cmp_ne_u32_e32_vcc(
        scalar_positive_inline_u32(static_cast<uint32_t>(current_kind)), tmp_vgpr, arch);
    const auto narrow_kind_conflict =
        build_s_and_saveexec_b64(next_exec_save_sgpr, kRdna4VccLo, arch);
    if (!prior_kind || !kind_ne || !narrow_kind_conflict)
      return false;
    words.insert(words.end(), prior_kind->begin(), prior_kind->end());
    words.push_back(*kind_ne);
    words.push_back(*narrow_kind_conflict);
  }

  if (options.moi_track_atomics) {
    const uint16_t base = *options.scratch_vgpr;
    const uint16_t exec_base = *options.moi_exec_save_sgpr;
    const uint16_t conflict_exec = static_cast<uint16_t>(exec_base + 2u);
    const uint16_t ordered_exec = static_cast<uint16_t>(exec_base + 4u);
    const uint16_t predicate_exec = static_cast<uint16_t>(exec_base + 6u);
    const uint16_t token_version = static_cast<uint16_t>(base + 7u);
    const uint16_t token_kind = static_cast<uint16_t>(base + 8u);
    const uint16_t token_epoch = static_cast<uint16_t>(base + 9u);
    const uint16_t source_version = static_cast<uint16_t>(base + 10u);
    const uint16_t token_hash = static_cast<uint16_t>(base + 11u);
    const uint16_t source_address = static_cast<uint16_t>(base + 12u);
    const uint16_t prior_owner = static_cast<uint16_t>(base + 23u);
    const auto save_conflict = build_s_mov_b64(conflict_exec, kRdna4ExecLo, arch);
    if (!save_conflict)
      return false;
    words.push_back(*save_conflict);

    if (!append_extract_exact_shadow_generation(words, address_lo_vgpr, current_value_vgpr,
                                                static_cast<uint16_t>(current_value_vgpr + 1u),
                                                current_field_vgpr, arch) ||
        !append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                           consan_moi_exact_shadow::owner_shift,
                                           consan_moi_exact_shadow::max_owner, arch) ||
        !append_extract_exact_shadow_field(words, prior_owner, old_value_vgpr,
                                           consan_moi_exact_shadow::owner_shift,
                                           consan_moi_exact_shadow::max_owner, arch) ||
        !append_inline_acquired_token_slot_address(
            words,
            *options.moi_report_buffer_address + layout.inline_acquired_epoch_token_slots_offset,
            layout.inline_acquired_epoch_token_capacity, address_lo_vgpr, current_field_vgpr,
            prior_owner, address_lo_vgpr, token_hash, static_cast<uint16_t>(address_lo_vgpr + 1u),
            arch))
      return false;

    const auto narrow = [&]() {
      const auto instruction = build_s_and_saveexec_b64(predicate_exec, kRdna4VccLo, arch);
      if (!instruction)
        return false;
      words.push_back(*instruction);
      return true;
    };
    const auto require_equal = [&](size_t offset, uint16_t expected) {
      if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr, offset, tmp_vgpr, arch))
        return false;
      const auto equal = build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(expected), tmp_vgpr, arch);
      if (!equal)
        return false;
      words.push_back(*equal);
      return narrow();
    };
    const auto require_scalar_equal = [&](size_t offset, uint16_t expected) {
      if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr, offset, tmp_vgpr, arch))
        return false;
      const auto equal = build_v_cmp_eq_u32_e32_vcc(expected, tmp_vgpr, arch);
      if (!equal)
        return false;
      words.push_back(*equal);
      return narrow();
    };
    const auto require_zero = [&](size_t offset) {
      if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr, offset, tmp_vgpr, arch))
        return false;
      const auto zero = build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
      if (!zero)
        return false;
      words.push_back(*zero);
      return narrow();
    };

    if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version),
                                        token_version, arch))
      return false;
    const auto token_nonzero =
        build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(0), token_version, arch);
    const auto token_parity = build_v_and_b32_e32_literal(tmp_vgpr, 1u, token_version, arch);
    if (!token_nonzero || !token_parity)
      return false;
    words.push_back(*token_nonzero);
    if (!narrow())
      return false;
    words.insert(words.end(), token_parity->begin(), token_parity->end());
    const auto token_even =
        build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
    if (!token_even)
      return false;
    words.push_back(*token_even);
    if (!narrow())
      return false;

    if (!append_extract_exact_shadow_generation(words, current_field_vgpr, current_value_vgpr,
                                                static_cast<uint16_t>(current_value_vgpr + 1u),
                                                tmp_vgpr, arch) ||
        !require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key),
                       current_field_vgpr) ||
        !append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                           consan_moi_exact_shadow::owner_shift,
                                           consan_moi_exact_shadow::max_owner, arch) ||
        !require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
                       current_field_vgpr) ||
        !require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
                       prior_owner) ||
        !append_load_u32_vgpr_at_offset(
            words, address_lo_vgpr,
            offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one), token_epoch,
            arch) ||
        !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind),
                                        token_kind, arch) ||
        !require_scalar_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id),
                              *options.moi_dispatch_id_sgpr) ||
        !require_scalar_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) +
                                  sizeof(uint32_t),
                              static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u)) ||
        !append_load_u32_vgpr_at_offset(
            words, address_lo_vgpr,
            offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address), source_address,
            arch) ||
        !append_load_u32_vgpr_at_offset(
            words, address_lo_vgpr,
            offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address) +
                sizeof(uint32_t),
            static_cast<uint16_t>(source_address + 1u), arch) ||
        !append_load_u32_vgpr_at_offset(
            words, address_lo_vgpr,
            offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version), source_version,
            arch) ||
        !require_zero(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reserved)) ||
        !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version),
                                        static_cast<uint16_t>(base + 14u), arch))
      return false;
    const auto source_address_nonzero =
        build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(0), source_address, arch);
    const auto source_version_nonzero =
        build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(0), source_version, arch);
    const auto source_version_parity =
        build_v_and_b32_e32_literal(tmp_vgpr, 1u, source_version, arch);
    if (!source_address_nonzero || !source_version_nonzero || !source_version_parity)
      return false;
    words.push_back(*source_address_nonzero);
    if (!narrow())
      return false;
    words.push_back(*source_version_nonzero);
    if (!narrow())
      return false;
    words.insert(words.end(), source_version_parity->begin(), source_version_parity->end());
    const auto source_version_even =
        build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
    if (!source_version_even)
      return false;
    words.push_back(*source_version_even);
    if (!narrow())
      return false;
    const auto token_unchanged = build_v_cmp_eq_u32_e32_vcc(
        vector_source_vgpr(token_version), static_cast<uint16_t>(base + 14u), arch);
    if (!token_unchanged)
      return false;
    words.push_back(*token_unchanged);
    if (!narrow())
      return false;

    if (!append_extract_exact_shadow_field(words, current_field_vgpr, old_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch))
      return false;
    const auto max_epoch =
        build_v_mov_b32_e64_literal(tmp_vgpr, consan_moi_exact_shadow::max_epoch, arch);
    const auto below_max =
        build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
    if (!max_epoch || !below_max)
      return false;
    words.insert(words.end(), max_epoch->begin(), max_epoch->end());
    words.push_back(*below_max);
    if (!narrow())
      return false;
    const auto add_one = build_v_add_nc_u32_e32(current_field_vgpr, scalar_positive_inline_u32(1),
                                                current_field_vgpr, arch);
    if (!add_one)
      return false;
    words.push_back(*add_one);
    const auto token_insufficient =
        build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(current_field_vgpr), token_epoch, arch);
    if (!token_insufficient)
      return false;
    words.push_back(*token_insufficient);
    const auto remove_insufficient =
        build_s_and_not1_b64(kRdna4ExecLo, kRdna4ExecLo, kRdna4VccLo, arch);
    if (!remove_insufficient)
      return false;
    words.push_back(*remove_insufficient);

    // Follow both direct and inherited tokens to one stable source release and
    // its complete causal snapshot. No token authorizes ordering unless the
    // final release version brackets every source and ancestry field.
    const uint16_t valid_exec = static_cast<uint16_t>(exec_base + 14u);
    const uint16_t active_exec = static_cast<uint16_t>(exec_base + 16u);
    const uint16_t inactive_exec = static_cast<uint16_t>(exec_base + 18u);
    const uint16_t release_version = static_cast<uint16_t>(base + 16u);
    const uint16_t snapshot_address = static_cast<uint16_t>(base + 17u);
    const uint16_t snapshot_temporary = static_cast<uint16_t>(base + 19u);
    const uint16_t release_owner = static_cast<uint16_t>(base + 20u);
    const uint16_t release_epoch = static_cast<uint16_t>(base + 21u);
    const uint16_t snapshot_hash = static_cast<uint16_t>(base + 22u);
    const uint16_t snapshot_count = token_version;
    const uint16_t snapshot_flags = source_version;
    const uint16_t previous_owner = token_hash;
    const uint16_t entry_owner = static_cast<uint16_t>(base + 14u);
    const uint16_t entry_epoch = static_cast<uint16_t>(base + 15u);
    const auto save_exec = [&](uint16_t destination) {
      const auto instruction = build_s_mov_b64(destination, kRdna4ExecLo, arch);
      if (!instruction)
        return false;
      words.push_back(*instruction);
      return true;
    };
    const auto restore_exec = [&](uint16_t source) {
      const auto instruction = build_s_mov_b64(kRdna4ExecLo, source, arch);
      if (!instruction)
        return false;
      words.push_back(*instruction);
      return true;
    };
    if (!append_inline_atomic_slot_address(
            words, *options.moi_report_buffer_address + layout.inline_atomic_release_slots_offset,
            layout.inline_atomic_release_capacity, source_address, address_lo_vgpr, arch) ||
        !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                        offsetof(ConSanMoiInlineAtomicReleaseSlot, version),
                                        release_version, arch) ||
        !require_equal(offsetof(ConSanMoiInlineAtomicReleaseSlot, version), source_version) ||
        !require_equal(offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
                       source_address) ||
        !require_equal(offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) +
                           sizeof(uint32_t),
                       static_cast<uint16_t>(source_address + 1u)) ||
        !append_extract_exact_shadow_generation(words, current_field_vgpr, current_value_vgpr,
                                                static_cast<uint16_t>(current_value_vgpr + 1u),
                                                tmp_vgpr, arch) ||
        !require_equal(offsetof(ConSanMoiInlineAtomicReleaseSlot, workgroup_key),
                       current_field_vgpr) ||
        !require_scalar_equal(offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id),
                              *options.moi_dispatch_id_sgpr) ||
        !require_scalar_equal(offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id) +
                                  sizeof(uint32_t),
                              static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u)) ||
        !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                        offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                                        release_owner, arch) ||
        !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                        offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch_plus_one),
                                        release_epoch, arch))
      return false;
    const auto owner_nonzero =
        build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(0), release_owner, arch);
    const auto owner_not_consumer = build_v_cmp_ne_u32_e32_vcc(
        vector_source_vgpr(*options.moi_owner_vgpr), release_owner, arch);
    const auto epoch_nonzero =
        build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(0), release_epoch, arch);
    const auto epoch_limit = build_v_mov_b32_e64_literal(
        snapshot_temporary, consan_moi_exact_shadow::max_epoch + 1u, arch);
    const auto epoch_in_range =
        build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(snapshot_temporary), release_epoch, arch);
    if (!owner_nonzero || !owner_not_consumer || !epoch_nonzero || !epoch_limit || !epoch_in_range)
      return false;
    words.insert(words.end(), epoch_limit->begin(), epoch_limit->end());
    for (const auto instruction :
         {*owner_nonzero, *owner_not_consumer, *epoch_nonzero, *epoch_in_range}) {
      words.push_back(instruction);
      if (!narrow())
        return false;
    }

    if (!append_inline_atomic_snapshot_address(
            words, *options.moi_report_buffer_address + layout.inline_causal_snapshots_offset,
            layout.inline_causal_snapshot_capacity, source_address, snapshot_address, snapshot_hash,
            snapshot_temporary, arch) ||
        !append_load_u32_vgpr_at_offset(words, snapshot_address,
                                        offsetof(ConSanMoiInlineCausalSnapshot, entry_count),
                                        snapshot_count, arch) ||
        !append_load_u32_vgpr_at_offset(words, snapshot_address,
                                        offsetof(ConSanMoiInlineCausalSnapshot, flags),
                                        snapshot_flags, arch))
      return false;
    const auto flags_clear =
        build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), snapshot_flags, arch);
    const auto count_in_range = build_v_cmp_gt_u32_e32_vcc(
        scalar_positive_inline_u32(kConSanMoiInlineCausalSnapshotEntryCapacity + 1u),
        snapshot_count, arch);
    if (!flags_clear || !count_in_range)
      return false;
    words.push_back(*flags_clear);
    if (!narrow())
      return false;
    words.push_back(*count_in_range);
    if (!narrow() || !save_exec(valid_exec))
      return false;
    const auto clear_ordered = build_s_mov_b64(ordered_exec, scalar_positive_inline_u32(0), arch);
    const auto clear_previous_owner = build_v_mov_b32_e64_literal(previous_owner, 0u, arch);
    if (!clear_ordered || !clear_previous_owner)
      return false;
    words.push_back(*clear_ordered);
    words.insert(words.end(), clear_previous_owner->begin(), clear_previous_owner->end());

    for (uint16_t i = 0; i < kConSanMoiInlineCausalSnapshotEntryCapacity; ++i) {
      const size_t entry_offset = offsetof(ConSanMoiInlineCausalSnapshot, entries) +
                                  i * sizeof(ConSanMoiInlineCausalSnapshotEntry);
      if (!restore_exec(valid_exec) ||
          !append_load_u32_vgpr_at_offset(words, snapshot_address, entry_offset, entry_owner,
                                          arch) ||
          !append_load_u32_vgpr_at_offset(words, snapshot_address, entry_offset + sizeof(uint32_t),
                                          entry_epoch, arch))
        return false;

      const auto active = build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(snapshot_count),
                                                     scalar_positive_inline_u32(i), arch);
      if (!active)
        return false;
      words.push_back(*active);
      if (!narrow())
        return false;
      const auto entry_owner_nonzero =
          build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(0), entry_owner, arch);
      const auto entry_not_releaser =
          build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(release_owner), entry_owner, arch);
      const auto entry_not_consumer = build_v_cmp_ne_u32_e32_vcc(
          vector_source_vgpr(*options.moi_owner_vgpr), entry_owner, arch);
      const auto entry_epoch_nonzero =
          build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(0), entry_epoch, arch);
      const auto entry_epoch_limit = build_v_mov_b32_e64_literal(
          snapshot_temporary, consan_moi_exact_shadow::max_epoch + 1u, arch);
      const auto entry_epoch_in_range =
          build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(snapshot_temporary), entry_epoch, arch);
      if (!entry_owner_nonzero || !entry_not_releaser || !entry_not_consumer ||
          !entry_epoch_nonzero || !entry_epoch_limit || !entry_epoch_in_range)
        return false;
      words.insert(words.end(), entry_epoch_limit->begin(), entry_epoch_limit->end());
      for (const auto instruction : {*entry_owner_nonzero, *entry_not_releaser, *entry_not_consumer,
                                     *entry_epoch_nonzero, *entry_epoch_in_range}) {
        words.push_back(instruction);
        if (!narrow())
          return false;
      }
      if (i != 0u) {
        const auto canonical =
            build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(entry_owner), previous_owner, arch);
        if (!canonical)
          return false;
        words.push_back(*canonical);
        if (!narrow())
          return false;
      }
      if (!save_exec(active_exec))
        return false;

      const auto inherited_kind =
          build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(1), token_kind, arch);
      const auto ancestor_owner =
          build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(prior_owner), entry_owner, arch);
      if (!inherited_kind || !ancestor_owner)
        return false;
      words.push_back(*inherited_kind);
      if (!narrow())
        return false;
      words.push_back(*ancestor_owner);
      if (!narrow())
        return false;
      const auto ancestor_epoch_insufficient =
          build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(token_epoch), entry_epoch, arch);
      if (!ancestor_epoch_insufficient)
        return false;
      words.push_back(*ancestor_epoch_insufficient);
      const auto remove_ancestor_epoch_insufficient =
          build_s_and_not1_b64(kRdna4ExecLo, kRdna4ExecLo, kRdna4VccLo, arch);
      const auto accumulate_inherited =
          build_s_xor_b64(ordered_exec, ordered_exec, kRdna4ExecLo, arch);
      if (!remove_ancestor_epoch_insufficient || !accumulate_inherited)
        return false;
      words.push_back(*remove_ancestor_epoch_insufficient);
      words.push_back(*accumulate_inherited);

      if (!restore_exec(active_exec))
        return false;
      words.push_back(build_v_mov_b32_e32(previous_owner, vector_source_vgpr(entry_owner), arch));

      if (!restore_exec(valid_exec))
        return false;
      const auto inactive =
          build_v_cmp_gt_u32_e32_vcc(scalar_positive_inline_u32(i + 1u), snapshot_count, arch);
      if (!inactive)
        return false;
      words.push_back(*inactive);
      if (!narrow())
        return false;
      const auto tail_owner_zero =
          build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), entry_owner, arch);
      const auto tail_epoch_zero =
          build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), entry_epoch, arch);
      if (!tail_owner_zero || !tail_epoch_zero)
        return false;
      words.push_back(*tail_owner_zero);
      if (!narrow())
        return false;
      words.push_back(*tail_epoch_zero);
      if (!narrow() || !save_exec(inactive_exec))
        return false;
      const auto union_valid = build_s_xor_b64(valid_exec, active_exec, inactive_exec, arch);
      if (!union_valid)
        return false;
      words.push_back(*union_valid);
    }

    if (!restore_exec(valid_exec) ||
        !append_inline_atomic_slot_address(
            words, *options.moi_report_buffer_address + layout.inline_atomic_release_slots_offset,
            layout.inline_atomic_release_capacity, source_address, address_lo_vgpr, arch) ||
        !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                        offsetof(ConSanMoiInlineAtomicReleaseSlot, version),
                                        snapshot_temporary, arch))
      return false;
    const auto release_unchanged =
        build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(release_version), snapshot_temporary, arch);
    if (!release_unchanged)
      return false;
    words.push_back(*release_unchanged);
    if (!narrow() || !save_exec(valid_exec))
      return false;

    // An inherited match collected during the scan is authoritative only for
    // lanes whose complete snapshot and final release version remained valid.
    const auto inherited_invalid =
        build_s_and_not1_b64(inactive_exec, ordered_exec, valid_exec, arch);
    const auto retain_valid_inherited =
        build_s_and_not1_b64(ordered_exec, ordered_exec, inactive_exec, arch);
    if (!inherited_invalid || !retain_valid_inherited)
      return false;
    words.push_back(*inherited_invalid);
    words.push_back(*retain_valid_inherited);

    if (!restore_exec(valid_exec))
      return false;
    const auto direct_kind =
        build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), token_kind, arch);
    const auto direct_owner =
        build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(prior_owner), release_owner, arch);
    if (!direct_kind || !direct_owner)
      return false;
    words.push_back(*direct_kind);
    if (!narrow())
      return false;
    words.push_back(*direct_owner);
    if (!narrow())
      return false;
    const auto direct_epoch_insufficient =
        build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(token_epoch), release_epoch, arch);
    if (!direct_epoch_insufficient)
      return false;
    words.push_back(*direct_epoch_insufficient);
    const auto remove_direct_epoch_insufficient =
        build_s_and_not1_b64(kRdna4ExecLo, kRdna4ExecLo, kRdna4VccLo, arch);
    if (!remove_direct_epoch_insufficient)
      return false;
    words.push_back(*remove_direct_epoch_insufficient);
    if (!save_exec(active_exec))
      return false;
    const auto combine_ordered = build_s_xor_b64(ordered_exec, ordered_exec, active_exec, arch);
    const auto remove_ordered =
        build_s_and_not1_b64(kRdna4ExecLo, conflict_exec, ordered_exec, arch);
    if (!combine_ordered || !remove_ordered)
      return false;
    words.push_back(*combine_ordered);
    words.push_back(*remove_ordered);
  }

  const uint16_t slot_vgpr = current_field_vgpr;
  const auto save_conflict_exec =
      build_s_mov_b64(static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u), kRdna4ExecLo, arch);
  const auto mbcnt_lo =
      build_v_mbcnt_lo_u32_b32(tmp_vgpr, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u),
                               scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi =
      build_v_mbcnt_hi_u32_b32(tmp_vgpr, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 3u),
                               vector_source_vgpr(tmp_vgpr), arch);
  const auto first_active_lane =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
  const auto narrow_representative = build_s_and_saveexec_b64(
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u), kRdna4VccLo, arch);
  if (!save_conflict_exec || !mbcnt_lo || !mbcnt_hi || !first_active_lane || !narrow_representative)
    return false;
  words.push_back(*save_conflict_exec);
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*narrow_representative);

  if (capture_first_diagnostic_only) {
    // A concurrency fault can make the same static conflict execute in every
    // affected workgroup. Preserve one complete, actionable first-fault
    // record instead of turning those repetitions into report overflow. The
    // returned old count is also the slot index: only the 0 -> 1 winner keeps
    // EXEC and writes slot zero.
    const uint64_t count_address = report_base + offsetof(ConSanMoiReportHeader, diagnostic_count);
    const auto mov_address_lo = build_v_mov_b32_e64_literal(
        *options.scratch_vgpr, static_cast<uint32_t>(count_address), arch);
    const auto mov_address_hi =
        build_v_mov_b32_e64_literal(static_cast<uint16_t>(*options.scratch_vgpr + 1u),
                                    static_cast<uint32_t>(count_address >> 32u), arch);
    const uint32_t mov_claimed =
        build_v_mov_b32_e32(slot_vgpr, scalar_positive_inline_u32(1), arch);
    const uint32_t mov_unclaimed =
        build_v_mov_b32_e32(tmp_vgpr, scalar_positive_inline_u32(0), arch);
    const auto claim = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
        *options.scratch_vgpr, slot_vgpr, slot_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
        arch);
    const auto claim_won =
        build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), slot_vgpr, arch);
    const auto narrow_winner = build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u), kRdna4VccLo, arch);
    if (!mov_address_lo || !mov_address_hi || !claim || !claim_won || !narrow_winner)
      return false;
    words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
    words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
    words.push_back(mov_claimed);
    words.push_back(mov_unclaimed);
    words.insert(words.end(), claim->begin(), claim->end());
    words.push_back(kWaitLoadcnt0);
    words.push_back(*claim_won);
    words.push_back(*narrow_winner);
  } else {
    if (!append_atomic_fetch_add_one_u32(
            words, report_base + offsetof(ConSanMoiReportHeader, diagnostic_count), slot_vgpr,
            *options.scratch_vgpr, arch))
      return false;
    const auto mov_capacity =
        build_v_mov_b32_e64_literal(tmp_vgpr, layout.diagnostic_capacity, arch);
    const auto slot_in_capacity =
        build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(tmp_vgpr), slot_vgpr, arch);
    const auto narrow_capacity = build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u), kRdna4VccLo, arch);
    if (!mov_capacity || !slot_in_capacity || !narrow_capacity)
      return false;
    words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
    words.push_back(*slot_in_capacity);
    words.push_back(*narrow_capacity);
  }

  const bool compact_local_diagnostic =
      options.moi_inline_workgroup_shadow && !options.moi_track_atomics;
  const uint16_t compact_address_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 11u);
  if (compact_local_diagnostic &&
      !append_dynamic_diagnostic_record_address(words, diagnostic_base, slot_vgpr,
                                                compact_address_vgpr, arch)) {
    return false;
  }

  const auto store_literal = [&](size_t offset, uint32_t value) {
    if (compact_local_diagnostic) {
      // This path claims the freshly zeroed slot exactly once. Avoid writing
      // fields whose ABI value is already zero, and use the one-word inline
      // encoding for small constants. The full diagnostic record remains
      // byte-for-byte compatible while hot exact probes stay compact enough
      // for large code objects.
      if (capture_first_diagnostic_only && value == 0u)
        return true;
      const auto store = build_flat_store_b32_vaddr_vsrc(compact_address_vgpr, tmp_vgpr, arch,
                                                         static_cast<uint32_t>(offset));
      if (!store)
        return false;
      if (value <= 64u) {
        words.push_back(build_v_mov_b32_e32(tmp_vgpr, scalar_positive_inline_u32(value), arch));
      } else {
        const auto mov_value = build_v_mov_b32_e64_literal(tmp_vgpr, value, arch);
        if (!mov_value)
          return false;
        words.insert(words.end(), mov_value->begin(), mov_value->end());
      }
      words.insert(words.end(), store->begin(), store->end());
      return true;
    }
    return append_dynamic_diagnostic_store_u32_literal(words, diagnostic_base + offset, value,
                                                       slot_vgpr, *options.scratch_vgpr, arch);
  };
  const auto store_vgpr = [&](size_t offset, uint16_t value_vgpr) {
    if (compact_local_diagnostic) {
      const auto store = build_flat_store_b32_vaddr_vsrc(compact_address_vgpr, value_vgpr, arch,
                                                         static_cast<uint32_t>(offset));
      if (!store)
        return false;
      words.insert(words.end(), store->begin(), store->end());
      return true;
    }
    return append_dynamic_diagnostic_store_u32_vgpr(words, diagnostic_base + offset, value_vgpr,
                                                    slot_vgpr, *options.scratch_vgpr, arch);
  };
  const auto store_scalar = [&](size_t offset, uint16_t scalar_src) {
    if (compact_local_diagnostic) {
      words.push_back(build_v_mov_b32_e32(tmp_vgpr, scalar_src, arch));
      const auto store = build_flat_store_b32_vaddr_vsrc(compact_address_vgpr, tmp_vgpr, arch,
                                                         static_cast<uint32_t>(offset));
      if (!store)
        return false;
      words.insert(words.end(), store->begin(), store->end());
      return true;
    }
    return append_dynamic_diagnostic_store_u32_scalar_src(
        words, diagnostic_base + offset, scalar_src, slot_vgpr, *options.scratch_vgpr, arch);
  };
  if (!store_literal(offsetof(ConSanMoiDiagnosticRecord, kind),
                     static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, backend),
                     static_cast<uint32_t>(ConSanMoiEngine::InlineShadow)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, generation),
                     static_cast<uint32_t>(options.moi_report_generation)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, generation) + sizeof(uint32_t),
                     static_cast<uint32_t>(options.moi_report_generation >> 32u)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, second_instruction_offset),
                     static_cast<uint32_t>(candidate.text_offset)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, second_access_kind),
                     static_cast<uint32_t>(current_kind)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lane_mask), 0u) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lane_mask) + sizeof(uint32_t), 0u) ||
      !store_scalar(offsetof(ConSanMoiDiagnosticRecord, second_lane_mask),
                    diagnostic_lane_mask_sgpr.value_or(
                        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u))) ||
      !store_scalar(offsetof(ConSanMoiDiagnosticRecord, second_lane_mask) + sizeof(uint32_t),
                    static_cast<uint16_t>(
                        diagnostic_lane_mask_sgpr.value_or(*options.moi_exec_save_sgpr + 2u) + 1u)))
    return false;

  if (!append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch) ||
      !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_owner_id), tmp_vgpr) ||
      !append_extract_exact_shadow_field(words, tmp_vgpr, current_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch) ||
      !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, second_owner_id), tmp_vgpr)) {
    return false;
  }

  const auto prior_inst_shift = build_v_lshrrev_b32_e32(
      tmp_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::instruction_offset_shift - 32u),
      old_value_hi_vgpr, arch);
  const auto prior_kind = build_v_and_b32_e32_literal(
      tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), old_value_vgpr,
      arch);
  if (!prior_inst_shift || !prior_kind)
    return false;
  words.push_back(*prior_inst_shift);
  if (!store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_instruction_offset), tmp_vgpr)) {
    return false;
  }
  words.insert(words.end(), prior_kind->begin(), prior_kind->end());
  if (!store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_access_kind), tmp_vgpr)) {
    return false;
  }
  if (has_epoch) {
    if (!append_extract_exact_shadow_field(words, tmp_vgpr, current_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch) ||
        !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, epoch), tmp_vgpr)) {
      return false;
    }
  }

  uint16_t diagnostic_offset_vgpr = lds_byte_offset_vgpr;
  if (static_byte_offset != 0) {
    if (!append_compute_effective_lds_byte_offset(words, current_value_vgpr, lds_byte_offset_vgpr,
                                                  static_byte_offset, arch))
      return false;
    diagnostic_offset_vgpr = current_value_vgpr;
  }
  if (!store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_offset),
                  diagnostic_offset_vgpr) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_count), byte_count) ||
      !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_offset),
                  diagnostic_offset_vgpr) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_count), byte_count))
    return false;

  const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!restore_exec)
    return false;
  words.push_back(*restore_exec);
  return append_restore_moi_special_state(words, options, arch);
}

// Publish one exact slot through an odd-version reservation. EXEC on entry is
// the set of lane publishers selected by the caller. EXEC on return contains
// only successfully committed publishers whose prior slot belonged to the
// same full hardware dispatch; those are the only lanes allowed to diagnose.
// A uniform address/metadata group retries bounded reservation contention;
// malformed nonuniform groups retain one-shot fail-closed accounting. Every
// entering lane that still cannot commit increments undercoverage exactly once.
[[nodiscard]] bool append_versioned_exact_shadow_transaction(
    std::vector<uint32_t> &words, const ConSanOptions &options, rj_code_arch_t arch,
    uint16_t address_lo_vgpr, uint16_t current_low_vgpr, uint16_t old_value_vgpr,
    uint16_t publisher_exec_sgpr, uint16_t committed_exec_sgpr, bool retry_uniform_contention) {
  if (!options.scratch_vgpr || !options.moi_exec_save_sgpr || !options.moi_dispatch_id_sgpr ||
      !options.moi_report_buffer_address)
    return false;

  const uint16_t exec_base = *options.moi_exec_save_sgpr;
  const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(exec_base + 2u);
  const uint16_t empty_exec_sgpr = exec_base;
  const uint16_t ready_exec_sgpr = temporary_exec_sgpr;
  const uint16_t retry_count_sgpr = static_cast<uint16_t>(exec_base + 8u);
  const uint16_t low_dispatch_exec_sgpr = static_cast<uint16_t>(exec_base + 18u);
  const uint16_t saved_address_lo_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t saved_address_hi_vgpr = static_cast<uint16_t>(old_value_vgpr + 3u);
  const uint16_t saved_current_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 4u);
  const uint16_t saved_current_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 5u);
  const uint16_t prior_dispatch_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 6u);
  const uint16_t prior_dispatch_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 7u);
  const uint16_t ready_version_vgpr = static_cast<uint16_t>(old_value_vgpr + 8u);
  const uint16_t cas_new_vgpr = static_cast<uint16_t>(old_value_vgpr + 9u);
  const uint16_t cas_expected_vgpr = static_cast<uint16_t>(old_value_vgpr + 10u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);

  const auto narrow_vcc = [&]() -> bool {
    const auto narrow = build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
    if (!narrow)
      return false;
    words.push_back(*narrow);
    return true;
  };
  const auto require_equal_literal = [&](uint16_t value_vgpr, uint32_t literal) -> bool {
    const auto compare =
        build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(literal), value_vgpr, arch);
    if (!compare)
      return false;
    words.push_back(*compare);
    return narrow_vcc();
  };
  const auto require_not_equal_literal = [&](uint16_t value_vgpr, uint32_t literal) -> bool {
    const auto compare =
        build_v_cmp_ne_u32_e32_vcc(scalar_positive_inline_u32(literal), value_vgpr, arch);
    if (!compare)
      return false;
    words.push_back(*compare);
    return narrow_vcc();
  };
  const auto restore_slot_address = [&]() {
    words.push_back(
        build_v_mov_b32_e32(address_lo_vgpr, vector_source_vgpr(saved_address_lo_vgpr), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(address_lo_vgpr + 1u),
                                        vector_source_vgpr(saved_address_hi_vgpr), arch));
  };

  constexpr uint32_t kUniformContentionRetryLimit = 2048u;
  if (retry_uniform_contention) {
    constexpr uint16_t kScalarLiteralSource = 255u;
    words.push_back(build_s_mov_b32(retry_count_sgpr, kScalarLiteralSource, arch));
    words.push_back(kUniformContentionRetryLimit);
  }
  const size_t retry_begin = words.size();

  if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, version),
                                      ready_version_vgpr, arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, packed_access),
                                      old_value_vgpr, arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, packed_access) +
                                          sizeof(uint32_t),
                                      static_cast<uint16_t>(old_value_vgpr + 1u), arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id),
                                      prior_dispatch_low_vgpr, arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) +
                                          sizeof(uint32_t),
                                      prior_dispatch_high_vgpr, arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, reserved), tmp_vgpr,
                                      arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, version),
                                      cas_expected_vgpr, arch)) {
    return false;
  }

  // A changed read, odd predecessor, or terminal version is unusable. Keep
  // narrowing instead of branching so a mixed wave can still publish valid
  // independent slots.
  const auto same_version =
      build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(ready_version_vgpr), cas_expected_vgpr, arch);
  const auto parity = build_v_and_b32_e32_literal(cas_new_vgpr, 1u, ready_version_vgpr, arch);
  const auto max_ready =
      build_v_mov_b32_e64_literal(cas_new_vgpr, kConSanMoiInlineExactMaxReadyVersion, arch);
  const auto below_terminal =
      build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(cas_new_vgpr), ready_version_vgpr, arch);
  if (!same_version || !parity || !max_ready || !below_terminal)
    return false;
  words.push_back(*same_version);
  if (!narrow_vcc())
    return false;
  words.insert(words.end(), parity->begin(), parity->end());
  if (!require_equal_literal(cas_new_vgpr, 0))
    return false;
  words.insert(words.end(), max_ready->begin(), max_ready->end());
  words.push_back(*below_terminal);
  if (!narrow_vcc())
    return false;
  const auto save_stable_exec = build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
  if (!save_stable_exec)
    return false;
  words.push_back(*save_stable_exec);

  // Version zero is empty only when every payload word is zero.
  if (!require_equal_literal(ready_version_vgpr, 0) || !require_equal_literal(old_value_vgpr, 0) ||
      !require_equal_literal(static_cast<uint16_t>(old_value_vgpr + 1u), 0) ||
      !require_equal_literal(prior_dispatch_low_vgpr, 0) ||
      !require_equal_literal(prior_dispatch_high_vgpr, 0) || !require_equal_literal(tmp_vgpr, 0)) {
    return false;
  }
  const auto save_empty_exec = build_s_mov_b64(empty_exec_sgpr, kRdna4ExecLo, arch);
  const auto restore_stable_exec = build_s_mov_b64(kRdna4ExecLo, committed_exec_sgpr, arch);
  if (!save_empty_exec || !restore_stable_exec)
    return false;
  words.push_back(*save_empty_exec);
  words.push_back(*restore_stable_exec);

  // A nonempty ready predecessor must itself be structurally valid. Owner and
  // epoch zero are legal packed values; access kind, workgroup key, dispatch,
  // and reserved state are the discriminants.
  if (!require_not_equal_literal(ready_version_vgpr, 0) || !require_equal_literal(tmp_vgpr, 0)) {
    return false;
  }
  const auto kind = build_v_and_b32_e32_literal(
      tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), old_value_vgpr,
      arch);
  if (!kind)
    return false;
  words.insert(words.end(), kind->begin(), kind->end());
  if (!require_not_equal_literal(tmp_vgpr, 0))
    return false;
  const auto kind_in_range = build_v_cmp_gt_u32_e32_vcc(
      scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Atomic) + 1u),
      tmp_vgpr, arch);
  if (!kind_in_range)
    return false;
  words.push_back(*kind_in_range);
  if (!narrow_vcc() ||
      !append_extract_exact_shadow_generation(words, tmp_vgpr, old_value_vgpr,
                                              static_cast<uint16_t>(old_value_vgpr + 1u),
                                              cas_new_vgpr, arch) ||
      !require_not_equal_literal(tmp_vgpr, 0)) {
    return false;
  }
  const auto save_ready_fields_exec = build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
  if (!save_ready_fields_exec)
    return false;
  words.push_back(*save_ready_fields_exec);

  // Form the nonzero 64-bit dispatch predicate without truncating it: the low
  // nonzero and low-zero/high-nonzero masks are disjoint, so XOR is their union.
  if (!require_not_equal_literal(prior_dispatch_low_vgpr, 0))
    return false;
  const auto save_low_dispatch_exec = build_s_mov_b64(low_dispatch_exec_sgpr, kRdna4ExecLo, arch);
  const auto restore_ready_fields_exec = build_s_mov_b64(kRdna4ExecLo, committed_exec_sgpr, arch);
  if (!save_low_dispatch_exec || !restore_ready_fields_exec)
    return false;
  words.push_back(*save_low_dispatch_exec);
  words.push_back(*restore_ready_fields_exec);
  if (!require_equal_literal(prior_dispatch_low_vgpr, 0) ||
      !require_not_equal_literal(prior_dispatch_high_vgpr, 0)) {
    return false;
  }
  const auto save_high_dispatch_exec = build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
  const auto union_ready_dispatch =
      build_s_xor_b64(ready_exec_sgpr, low_dispatch_exec_sgpr, committed_exec_sgpr, arch);
  const auto union_valid = build_s_xor_b64(kRdna4ExecLo, empty_exec_sgpr, ready_exec_sgpr, arch);
  if (!save_high_dispatch_exec || !union_ready_dispatch || !union_valid)
    return false;
  words.push_back(*save_high_dispatch_exec);
  words.push_back(*union_ready_dispatch);
  words.push_back(*union_valid);

  const auto claim_new =
      build_v_add_nc_u32_e32(cas_new_vgpr, scalar_positive_inline_u32(1), ready_version_vgpr, arch);
  const uint32_t claim_expected =
      build_v_mov_b32_e32(cas_expected_vgpr, vector_source_vgpr(ready_version_vgpr), arch);
  if (!claim_new || !append_add_literal_field(words, address_lo_vgpr,
                                              offsetof(ConSanMoiInlineExactShadowSlot, version),
                                              tmp_vgpr, arch)) {
    return false;
  }
  words.push_back(*claim_new);
  words.push_back(claim_expected);
  const auto claim = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      address_lo_vgpr, cas_new_vgpr, cas_new_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
      arch);
  if (!claim)
    return false;
  words.insert(words.end(), claim->begin(), claim->end());
  words.push_back(kWaitLoadcnt0);
  const auto claim_won =
      build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(cas_expected_vgpr), cas_new_vgpr, arch);
  if (!claim_won)
    return false;
  words.push_back(*claim_won);
  if (!narrow_vcc())
    return false;
  if (retry_uniform_contention) {
    const auto save_claimed = build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
    const auto select_failed =
        build_s_and_not1_b64(kRdna4ExecLo, publisher_exec_sgpr, committed_exec_sgpr, arch);
    const auto decrement =
        build_s_sub_u32(retry_count_sgpr, retry_count_sgpr, scalar_positive_inline_u32(1), arch);
    const auto retries_remain =
        build_rdna4_s_cmp_lg_u32(retry_count_sgpr, scalar_positive_inline_u32(0), arch);
    const auto restore_claimed = build_s_mov_b64(kRdna4ExecLo, committed_exec_sgpr, arch);
    if (!save_claimed || !select_failed || !decrement || !retries_remain || !restore_claimed) {
      return false;
    }
    words.push_back(*save_claimed);
    words.push_back(*select_failed);
    // Refresh the version only on contending lanes. The ordinary two-read
    // snapshot is substantially cheaper on the successful path, while this
    // device-scope zero-add prevents a loser from retaining a stale odd cache
    // line across every bounded retry.
    restore_slot_address();
    if (!append_add_literal_field(words, address_lo_vgpr,
                                  offsetof(ConSanMoiInlineExactShadowSlot, version), tmp_vgpr,
                                  arch) ||
        !append_atomic_load_u32(words, address_lo_vgpr, cas_expected_vgpr, arch)) {
      return false;
    }
    restore_slot_address();
    words.push_back(build_s_sleep(/*delay=*/1, arch));
    words.push_back(*decrement);
    words.push_back(*retries_remain);
    restore_slot_address();
    words.push_back(
        build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(current_low_vgpr + 1u),
                                        vector_source_vgpr(saved_current_high_vgpr), arch));
    const auto retries_exhausted = build_s_cbranch_scc0(/*simm16=*/1, arch);
    if (!retries_exhausted)
      return false;
    words.push_back(*retries_exhausted);
    const size_t retry_branch = words.size();
    const int64_t retry_delta =
        static_cast<int64_t>(retry_begin) - static_cast<int64_t>(retry_branch + 1u);
    if (retry_delta < std::numeric_limits<int16_t>::min() ||
        retry_delta > std::numeric_limits<int16_t>::max()) {
      return false;
    }
    const auto retry = build_s_cbranch_execnz(static_cast<int16_t>(retry_delta), arch);
    if (!retry)
      return false;
    words.push_back(*retry);
    words.push_back(*restore_claimed);
  }
  restore_slot_address();

  if (!append_store_u32_vgpr_at_offset(words, address_lo_vgpr,
                                       offsetof(ConSanMoiInlineExactShadowSlot, packed_access),
                                       saved_current_low_vgpr, arch) ||
      !append_store_u32_vgpr_at_offset(words, address_lo_vgpr,
                                       offsetof(ConSanMoiInlineExactShadowSlot, packed_access) +
                                           sizeof(uint32_t),
                                       saved_current_high_vgpr, arch)) {
    return false;
  }
  words.push_back(build_v_mov_b32_e32(current_low_vgpr, *options.moi_dispatch_id_sgpr, arch));
  words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(current_low_vgpr + 1u),
                                      static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u),
                                      arch));
  if (!append_store_u32_vgpr_at_offset(words, address_lo_vgpr,
                                       offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id),
                                       current_low_vgpr, arch) ||
      !append_store_u32_vgpr_at_offset(words, address_lo_vgpr,
                                       offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) +
                                           sizeof(uint32_t),
                                       static_cast<uint16_t>(current_low_vgpr + 1u), arch)) {
    return false;
  }
  const auto zero = build_v_mov_b32_e64_literal(tmp_vgpr, 0, arch);
  const auto store_drain = build_s_wait_storecnt0(arch);
  if (!zero || !store_drain)
    return false;
  words.insert(words.end(), zero->begin(), zero->end());
  if (!append_store_u32_vgpr_at_offset(words, address_lo_vgpr,
                                       offsetof(ConSanMoiInlineExactShadowSlot, reserved), tmp_vgpr,
                                       arch)) {
    return false;
  }
  words.push_back(*store_drain);

  const auto commit_new =
      build_v_add_nc_u32_e32(cas_new_vgpr, scalar_positive_inline_u32(2), ready_version_vgpr, arch);
  const auto commit_expected = build_v_add_nc_u32_e32(
      cas_expected_vgpr, scalar_positive_inline_u32(1), ready_version_vgpr, arch);
  if (!commit_new || !commit_expected ||
      !append_add_literal_field(words, address_lo_vgpr,
                                offsetof(ConSanMoiInlineExactShadowSlot, version), tmp_vgpr,
                                arch)) {
    return false;
  }
  words.push_back(*commit_new);
  words.push_back(*commit_expected);
  const auto commit = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      address_lo_vgpr, cas_new_vgpr, cas_new_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
      arch);
  if (!commit)
    return false;
  words.insert(words.end(), commit->begin(), commit->end());
  words.push_back(kWaitLoadcnt0);
  const auto commit_won =
      build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(cas_expected_vgpr), cas_new_vgpr, arch);
  if (!commit_won)
    return false;
  words.push_back(*commit_won);
  if (!narrow_vcc())
    return false;
  const auto save_committed = build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
  if (!save_committed)
    return false;
  words.push_back(*save_committed);

  // Count successful publications before excluding cross-dispatch priors from
  // diagnosis. The event counter therefore remains a successful-commit count.
  if (!append_atomic_fetch_add_one_u32(words,
                                       *options.moi_report_buffer_address +
                                           offsetof(ConSanMoiReportHeader, event_counter),
                                       tmp_vgpr, address_lo_vgpr, arch)) {
    return false;
  }
  restore_slot_address();

  const auto restore_publishers = build_s_mov_b64(kRdna4ExecLo, publisher_exec_sgpr, arch);
  const auto select_failed =
      build_s_and_not1_b64(kRdna4ExecLo, publisher_exec_sgpr, committed_exec_sgpr, arch);
  const auto restore_committed = build_s_mov_b64(kRdna4ExecLo, committed_exec_sgpr, arch);
  if (!restore_publishers || !select_failed || !restore_committed)
    return false;
  words.push_back(*restore_publishers);
  words.push_back(*select_failed);
  if (!append_atomic_fetch_add_one_u32(
          words,
          *options.moi_report_buffer_address +
              offsetof(ConSanMoiReportHeader, inline_undercoverage_count),
          tmp_vgpr, address_lo_vgpr, arch)) {
    return false;
  }
  words.push_back(*restore_committed);
  restore_slot_address();
  words.push_back(
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch));
  words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(current_low_vgpr + 1u),
                                      vector_source_vgpr(saved_current_high_vgpr), arch));

  const auto same_dispatch_low =
      build_v_cmp_eq_u32_e32_vcc(*options.moi_dispatch_id_sgpr, prior_dispatch_low_vgpr, arch);
  const auto same_dispatch_high = build_v_cmp_eq_u32_e32_vcc(
      static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u), prior_dispatch_high_vgpr, arch);
  if (!same_dispatch_low || !same_dispatch_high)
    return false;
  words.push_back(*same_dispatch_low);
  if (!narrow_vcc())
    return false;
  words.push_back(*same_dispatch_high);
  return narrow_vcc();
}

// Partition the incoming wave by effective exact-shadow address. Each address
// group with uniform metadata publishes through one representative lane. A
// group whose owner/epoch/kind metadata is not uniform submits every lane to
// the versioned transaction rather than selecting arbitrary metadata; one CAS
// winner may publish and every loser poisons coverage. The loop removes at
// least one lane on every iteration, so runtime work is bounded by the hardware
// wave size while generated code size remains constant.
[[nodiscard]] bool append_wave_coalesced_exact_shadow_swap(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate, const ConSanOptions &options,
    rj_code_arch_t arch, const ConSanMoiReportBufferLayout &layout, uint16_t address_lo_vgpr,
    uint16_t current_low_vgpr, uint16_t current_high_vgpr, uint16_t old_value_vgpr,
    uint16_t lds_byte_offset_vgpr, uint32_t static_byte_offset, uint32_t byte_count) {
  if (!options.moi_exec_save_sgpr)
    return false;

  const uint16_t base = *options.moi_exec_save_sgpr;
  const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(base + 2u);
  const uint16_t publisher_exec_sgpr = static_cast<uint16_t>(base + 4u);
  const uint16_t committed_exec_sgpr = static_cast<uint16_t>(base + 6u);
  const uint16_t incoming_exec_sgpr = static_cast<uint16_t>(base + 12u);
  const uint16_t pending_exec_sgpr = static_cast<uint16_t>(base + 14u);
  const uint16_t group_exec_sgpr = static_cast<uint16_t>(base + 16u);
  const uint16_t address_key_sgpr = static_cast<uint16_t>(base + 18u);
  const uint16_t value_key_sgpr = static_cast<uint16_t>(base + 19u);
  const uint16_t lane_rank_vgpr = static_cast<uint16_t>(old_value_vgpr + 1u);
  const uint16_t saved_address_lo_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t saved_address_hi_vgpr = static_cast<uint16_t>(old_value_vgpr + 3u);
  const uint16_t saved_current_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 4u);
  const uint16_t saved_current_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 5u);

  const auto save_incoming_exec = build_s_mov_b64(incoming_exec_sgpr, kRdna4ExecLo, arch);
  const auto initialize_pending = build_s_mov_b64(pending_exec_sgpr, kRdna4ExecLo, arch);
  const auto select_pending = build_s_mov_b64(kRdna4ExecLo, pending_exec_sgpr, arch);
  const uint32_t save_address_lo =
      build_v_mov_b32_e32(saved_address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), arch);
  const uint32_t save_address_hi = build_v_mov_b32_e32(
      saved_address_hi_vgpr, vector_source_vgpr(static_cast<uint16_t>(address_lo_vgpr + 1u)), arch);
  const uint32_t save_current_low =
      build_v_mov_b32_e32(saved_current_low_vgpr, vector_source_vgpr(current_low_vgpr), arch);
  const uint32_t save_current_high =
      build_v_mov_b32_e32(saved_current_high_vgpr, vector_source_vgpr(current_high_vgpr), arch);
  const uint32_t restore_address_lo =
      build_v_mov_b32_e32(address_lo_vgpr, vector_source_vgpr(saved_address_lo_vgpr), arch);
  const uint32_t restore_address_hi = build_v_mov_b32_e32(
      static_cast<uint16_t>(address_lo_vgpr + 1u), vector_source_vgpr(saved_address_hi_vgpr), arch);
  const uint32_t restore_current_low =
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch);
  const uint32_t restore_current_high =
      build_v_mov_b32_e32(current_high_vgpr, vector_source_vgpr(saved_current_high_vgpr), arch);
  const auto read_address = build_v_readfirstlane_b32(address_key_sgpr, address_lo_vgpr, arch);
  const auto read_address_high =
      build_v_readfirstlane_b32(value_key_sgpr, static_cast<uint16_t>(address_lo_vgpr + 1u), arch);
  const auto read_value = build_v_readfirstlane_b32(value_key_sgpr, current_low_vgpr, arch);
  const auto address_equal = build_v_cmp_eq_u32_e32_vcc(address_key_sgpr, address_lo_vgpr, arch);
  const auto narrow_address = build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  const auto save_group = build_s_mov_b64(group_exec_sgpr, kRdna4ExecLo, arch);
  const auto value_equal = build_v_cmp_eq_u32_e32_vcc(value_key_sgpr, current_low_vgpr, arch);
  const auto narrow_value = build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  const auto low_mask_equal = build_s_cmp_eq_u32(kRdna4ExecLo, group_exec_sgpr, arch);
  const auto high_mask_equal = build_s_cmp_eq_u32(
      static_cast<uint16_t>(kRdna4ExecLo + 1u), static_cast<uint16_t>(group_exec_sgpr + 1u), arch);
  if (!save_incoming_exec || !initialize_pending || !select_pending || !read_address ||
      !read_address_high || !read_value || !address_equal || !narrow_address || !save_group ||
      !value_equal || !narrow_value || !low_mask_equal || !high_mask_equal)
    return false;

  words.push_back(*save_incoming_exec);
  words.push_back(*initialize_pending);
  words.push_back(save_address_lo);
  words.push_back(save_address_hi);
  words.push_back(save_current_low);
  words.push_back(save_current_high);
  if (options.moi_partition_mask_debug) {
    // Capture the first incoming lane's address before loop temporaries can
    // reuse these SGPRs. InlineShadow leaves all three record counts at zero,
    // so the debug reader can use count fields without corrupting the layout
    // capacities that locate the exact-shadow array.
    const uint64_t report_base = *options.moi_report_buffer_address;
    words.push_back(*read_address);
    words.push_back(*read_address_high);
    if (!append_store_u32_sgpr(words,
                               report_base + offsetof(ConSanMoiReportHeader, access_record_count),
                               address_key_sgpr, current_low_vgpr, address_lo_vgpr, arch) ||
        !append_store_u32_sgpr(words,
                               report_base + offsetof(ConSanMoiReportHeader, barrier_record_count),
                               value_key_sgpr, current_low_vgpr, address_lo_vgpr, arch))
      return false;
  }
  const size_t loop = words.size();
  words.push_back(*select_pending);
  words.push_back(restore_address_lo);
  words.push_back(restore_address_hi);
  words.push_back(restore_current_low);
  words.push_back(restore_current_high);
  words.push_back(*read_address);
  words.push_back(*read_value);
  words.push_back(*address_equal);
  words.push_back(*narrow_address);
  words.push_back(*save_group);
  if (options.moi_partition_mask_debug) {
    // Capture the current group's exchange address before diagnostics can
    // reuse working VGPRs. Each group overwrites this pair, matching the
    // retained final_group mask. The old-value pair has not been populated by
    // the shadow exchange yet, so it is the only safe address scratch here.
    // In particular, do not use saved_address_lo_vgpr: doing so corrupts the
    // per-lane loop invariant before the first exchange and makes the debug
    // path observe different behavior from the production path.
    const uint64_t report_base = *options.moi_report_buffer_address;
    if (!append_store_u32_vgpr(words,
                               report_base + offsetof(ConSanMoiReportHeader, atomic_record_count),
                               address_lo_vgpr, old_value_vgpr, arch) ||
        !append_store_u32_vgpr(words, report_base + offsetof(ConSanMoiReportHeader, flags),
                               static_cast<uint16_t>(address_lo_vgpr + 1u), old_value_vgpr, arch))
      return false;
  }
  words.push_back(*value_equal);
  words.push_back(*narrow_value);
  words.push_back(*low_mask_equal);
  const size_t low_fallback_branch = words.size();
  words.push_back(0);
  words.push_back(*high_mask_equal);
  const size_t high_fallback_branch = words.size();
  words.push_back(0);

  const auto mbcnt_lo = build_v_mbcnt_lo_u32_b32(lane_rank_vgpr, group_exec_sgpr,
                                                 scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi =
      build_v_mbcnt_hi_u32_b32(lane_rank_vgpr, static_cast<uint16_t>(group_exec_sgpr + 1u),
                               vector_source_vgpr(lane_rank_vgpr), arch);
  const auto first_lane =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
  const auto narrow_first = build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  const auto restore_group_exec = build_s_mov_b64(kRdna4ExecLo, group_exec_sgpr, arch);
  const auto save_publishers = build_s_mov_b64(publisher_exec_sgpr, kRdna4ExecLo, arch);
  const auto use_group_diagnostic_mask =
      build_s_mov_b64(publisher_exec_sgpr, group_exec_sgpr, arch);
  const auto use_committed_diagnostic_mask =
      build_s_mov_b64(publisher_exec_sgpr, kRdna4ExecLo, arch);
  if (!mbcnt_lo || !mbcnt_hi || !first_lane || !narrow_first || !restore_group_exec ||
      !save_publishers || !use_group_diagnostic_mask || !use_committed_diagnostic_mask)
    return false;
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_lane);
  words.push_back(*narrow_first);
  words.push_back(*save_publishers);
  if (!append_versioned_exact_shadow_transaction(words, options, arch, address_lo_vgpr,
                                                 current_low_vgpr, old_value_vgpr,
                                                 publisher_exec_sgpr, committed_exec_sgpr,
                                                 /*retry_uniform_contention=*/true)) {
    return false;
  }
  // A uniform representative stands for the complete metadata-identical
  // address group. Retain that exact group as the diagnostic lane mask.
  words.push_back(*use_group_diagnostic_mask);
  const size_t common_branch = words.size();
  words.push_back(0);

  const size_t fallback = words.size();
  words.push_back(*restore_group_exec);
  words.push_back(*save_publishers);
  if (!append_versioned_exact_shadow_transaction(words, options, arch, address_lo_vgpr,
                                                 current_low_vgpr, old_value_vgpr,
                                                 publisher_exec_sgpr, committed_exec_sgpr,
                                                 /*retry_uniform_contention=*/false)) {
    return false;
  }
  // Nonuniform metadata cannot be attributed to the whole group. EXEC now
  // contains only committed, same-dispatch publishers admitted to diagnose.
  words.push_back(*use_committed_diagnostic_mask);
  const size_t diagnostic = words.size();

  // Counter updates and version-address CAS operations use the working address
  // pair. Restore it from the loop invariants before diagnostics and the next
  // partition iteration.
  words.push_back(restore_address_lo);
  words.push_back(restore_address_hi);

  // The partition operations changed VCC/SCC. Restore the incoming values so
  // the existing diagnostic helper can snapshot and restore the true program
  // state while evaluating every lane in this address group.
  if (!append_restore_moi_special_state(words, options, arch) ||
      !append_inline_shadow_diagnostic_words(
          words, candidate, options, arch, layout, old_value_vgpr,
          static_cast<uint16_t>(old_value_vgpr + 1u), current_low_vgpr, current_high_vgpr,
          lds_byte_offset_vgpr, static_byte_offset, byte_count,
          /*special_state_already_saved=*/true,
          /*diagnostic_lane_mask_sgpr=*/publisher_exec_sgpr))
    return false;

  if (options.moi_partition_mask_debug) {
    // Debug-only side channel for gfx1201 qualification. The report fields are
    // otherwise unused by InlineShadow. Capture before the subtraction while
    // group EXEC is still active; the matching hook log labels the run as
    // non-acceptance telemetry.
    const uint64_t report_base = *options.moi_report_buffer_address;
    if (!append_store_u32_sgpr(words, report_base + offsetof(ConSanMoiReportHeader, dispatch_id),
                               group_exec_sgpr, current_low_vgpr, address_lo_vgpr, arch) ||
        !append_store_u32_sgpr(
            words, report_base + offsetof(ConSanMoiReportHeader, dispatch_id) + sizeof(uint32_t),
            static_cast<uint16_t>(group_exec_sgpr + 1u), current_low_vgpr, address_lo_vgpr, arch))
      return false;
  }

  // The selected group is a subset of pending EXEC, so XOR removes exactly
  // that group. Keep this independent of the gfx12 AND-NOT operand convention;
  // live qualification is the authority for this generated control flow.
  const auto remove_group =
      build_s_xor_b64(pending_exec_sgpr, pending_exec_sgpr, group_exec_sgpr, arch);
  const auto select_remaining = build_s_mov_b64(kRdna4ExecLo, pending_exec_sgpr, arch);
  const auto restore_incoming_exec = build_s_mov_b64(kRdna4ExecLo, incoming_exec_sgpr, arch);
  if (!remove_group || !select_remaining || !restore_incoming_exec)
    return false;
  words.push_back(*remove_group);
  // Install the remaining mask into EXEC and branch on it. Live gfx1201 debug
  // telemetry proved this traversal reaches the final divergent lane when
  // there is scalar distance after the pending-mask XOR. Keep that dependency
  // explicit: without it, both an immediate EXEC move and an immediate scalar
  // compare observed only the first address group in acceptance runs.
  words.push_back(build_s_nop(0, arch));
  words.push_back(build_s_nop(0, arch));
  words.push_back(*select_remaining);
  const size_t loop_branch = words.size();
  words.push_back(0);
  words.push_back(*restore_incoming_exec);
  if (!append_restore_moi_special_state(words, options, arch))
    return false;

  const auto branch_offset = [](size_t from, size_t to) -> std::optional<int16_t> {
    const int64_t delta = static_cast<int64_t>(to) - static_cast<int64_t>(from + 1u);
    if (delta < std::numeric_limits<int16_t>::min() || delta > std::numeric_limits<int16_t>::max())
      return std::nullopt;
    return static_cast<int16_t>(delta);
  };
  const auto low_delta = branch_offset(low_fallback_branch, fallback);
  const auto high_delta = branch_offset(high_fallback_branch, fallback);
  const auto diagnostic_delta = branch_offset(common_branch, diagnostic);
  const auto loop_delta = branch_offset(loop_branch, loop);
  if (!low_delta || !high_delta || !diagnostic_delta || !loop_delta)
    return false;
  const auto low_branch = build_s_cbranch_scc0(*low_delta, arch);
  const auto high_branch = build_s_cbranch_scc0(*high_delta, arch);
  const auto continue_loop = build_s_cbranch_execnz(*loop_delta, arch);
  if (!low_branch || !high_branch || !continue_loop)
    return false;
  words[low_fallback_branch] = *low_branch;
  words[high_fallback_branch] = *high_branch;
  words[common_branch] = build_s_branch(*diagnostic_delta, arch);
  words[loop_branch] = *continue_loop;
  return true;
}

// Publish one exact metadata entry into the bounded shadow interval reserved
// in the owning kernel's LDS allocation. Unlike the global versioned table,
// a DS exchange is naturally scoped to one workgroup and atomically returns
// the complete prior 64-bit entry, so it needs neither dispatch qualification
// nor a multi-instruction reservation protocol.
[[nodiscard]] bool append_workgroup_local_exact_shadow_swap(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate, const ConSanOptions &options,
    rj_code_arch_t arch, const ConSanMoiReportBufferLayout &report_layout,
    const ConSanMoiWorkgroupShadowLayout &workgroup_shadow, uint16_t address_vgpr,
    uint16_t current_low_vgpr, uint16_t current_high_vgpr, uint16_t old_value_vgpr,
    uint16_t lds_byte_offset_vgpr, uint32_t static_byte_offset, uint32_t cell_index,
    uint32_t byte_count, uint16_t diagnostic_lds_byte_offset_vgpr,
    uint32_t diagnostic_static_byte_offset) {
  if (!options.scratch_vgpr || !options.moi_exec_save_sgpr)
    return false;

  const uint16_t temporary_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t saved_current_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 4u);
  const uint16_t saved_current_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 5u);
  const uint16_t saved_cell_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t exec_base = *options.moi_exec_save_sgpr;
  const uint16_t incoming_exec_sgpr = static_cast<uint16_t>(exec_base + 12u);
  const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(exec_base + 14u);
  const uint16_t bounded_exec_sgpr = static_cast<uint16_t>(exec_base + 16u);
  const uint16_t valid_workgroup_exec_sgpr = static_cast<uint16_t>(exec_base + 22u);
  uint16_t effective_offset_vgpr = lds_byte_offset_vgpr;
  if (static_byte_offset != 0u) {
    if (!append_compute_effective_lds_byte_offset(words, temporary_vgpr, lds_byte_offset_vgpr,
                                                  static_byte_offset, arch))
      return false;
    effective_offset_vgpr = temporary_vgpr;
  }

  const auto cell = build_v_lshrrev_b32_e32(
      temporary_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
      effective_offset_vgpr, arch);
  const uint32_t shadow_cell_capacity = workgroup_shadow.size / sizeof(uint64_t);
  const uint32_t cell_capacity =
      cell_index < shadow_cell_capacity ? shadow_cell_capacity - cell_index : 0u;
  const auto capacity = build_v_mov_b32_e64_literal(address_vgpr, cell_capacity, arch);
  const auto in_bounds =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(address_vgpr), temporary_vgpr, arch);
  const auto save_incoming_exec = build_s_mov_b64(incoming_exec_sgpr, kRdna4ExecLo, arch);
  const auto narrow_bounded = build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  const auto save_bounded_exec = build_s_mov_b64(bounded_exec_sgpr, kRdna4ExecLo, arch);
  const auto select_out_of_bounds =
      build_s_and_not1_b64(kRdna4ExecLo, incoming_exec_sgpr, bounded_exec_sgpr, arch);
  const auto restore_bounded_exec = build_s_mov_b64(kRdna4ExecLo, bounded_exec_sgpr, arch);
  const auto shadow_byte_offset =
      build_v_lshlrev_b32_e32(temporary_vgpr, scalar_positive_inline_u32(3), temporary_vgpr, arch);
  const uint32_t local_base = workgroup_shadow.base + cell_index * sizeof(uint64_t);
  const auto local_address =
      build_v_add_nc_u32_e32_literal(address_vgpr, local_base, temporary_vgpr, arch);
  const auto exchange =
      build_ds_storexchg_rtn_b64(old_value_vgpr, address_vgpr, current_low_vgpr, 0u, arch);
  const auto restore_valid_workgroup =
      build_s_mov_b64(kRdna4ExecLo, valid_workgroup_exec_sgpr, arch);
  if (!cell || !capacity || !in_bounds || !save_incoming_exec || !narrow_bounded ||
      !save_bounded_exec || !select_out_of_bounds || !restore_bounded_exec || !shadow_byte_offset ||
      !local_address || !exchange || !restore_valid_workgroup)
    return false;

  words.push_back(
      build_v_mov_b32_e32(saved_current_low_vgpr, vector_source_vgpr(current_low_vgpr), arch));
  words.push_back(
      build_v_mov_b32_e32(saved_current_high_vgpr, vector_source_vgpr(current_high_vgpr), arch));
  words.push_back(*cell);
  words.push_back(build_v_mov_b32_e32(saved_cell_vgpr, vector_source_vgpr(temporary_vgpr), arch));
  words.insert(words.end(), capacity->begin(), capacity->end());
  words.push_back(*in_bounds);
  words.push_back(*save_incoming_exec);
  words.push_back(*narrow_bounded);
  words.push_back(*save_bounded_exec);
  words.push_back(*select_out_of_bounds);
  if (!append_atomic_fetch_add_one_u32(
          words,
          *options.moi_report_buffer_address +
              offsetof(ConSanMoiReportHeader, inline_undercoverage_count),
          temporary_vgpr, address_vgpr, arch))
    return false;
  words.push_back(*restore_bounded_exec);
  words.push_back(build_v_mov_b32_e32(temporary_vgpr, vector_source_vgpr(saved_cell_vgpr), arch));
  words.push_back(*shadow_byte_offset);
  words.insert(words.end(), local_address->begin(), local_address->end());
  words.insert(words.end(), exchange->begin(), exchange->end());
  words.push_back(kWaitDscnt0);

  // Address formation and conflict filtering use condition state. Reinstall
  // the displaced program's state before the diagnostic helper, whose final
  // restoration makes the local path observationally transparent.
  if (!append_restore_moi_special_state(words, options, arch) ||
      !append_inline_shadow_diagnostic_words(
          words, candidate, options, arch, report_layout, old_value_vgpr,
          static_cast<uint16_t>(old_value_vgpr + 1u), current_low_vgpr, current_high_vgpr,
          diagnostic_lds_byte_offset_vgpr, diagnostic_static_byte_offset, byte_count,
          /*special_state_already_saved=*/true,
          /*diagnostic_lane_mask_sgpr=*/std::nullopt,
          /*capture_first_diagnostic_only=*/true))
    return false;

  words.push_back(
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch));
  words.push_back(
      build_v_mov_b32_e32(current_high_vgpr, vector_source_vgpr(saved_current_high_vgpr), arch));
  words.push_back(*restore_valid_workgroup);
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_shadow_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, const ConSanMoiReportBufferLayout &layout,
    const ConSanMoiWorkgroupSources &workgroup_sources,
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow,
    std::optional<uint32_t> private_epoch_offset, std::optional<uint32_t> private_owner_offset,
    std::optional<uint16_t> private_owner_shift, std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (!options.moi_exec_save_sgpr || !options.moi_dispatch_id_sgpr) {
    errors.emplace_back(
        "ConSan MOI versioned exact-shadow probe requires EXEC-save and dispatch-ID state");
    return std::nullopt;
  }
  if (options.moi_partition_mask_debug &&
      (!options.moi_exec_save_sgpr || layout.diagnostic_capacity == 0)) {
    errors.emplace_back("ConSan MOI partition-mask debug requires wave-coalesced InlineShadow");
    return std::nullopt;
  }
  const uint16_t scratch_count = inline_shadow_scratch_count(options, candidate);
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back("ConSan MOI inline-shadow probe scratch VGPR window exceeds the limit");
    return std::nullopt;
  }
  if (!validate_inline_shadow_exec_save_sgpr(options, errors))
    return std::nullopt;
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (reject_candidate_scratch_range_overlap(candidate, *options.scratch_vgpr, scratch_count,
                                             errors))
    return std::nullopt;
  if (!options.moi_owner_vgpr && !options.automatic_moi_private_epoch) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires persistent owner state");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI epoch", errors) ||
      reject_optional_scratch_range_overlap(options.moi_workgroup_key_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI workgroup key", errors))
    return std::nullopt;

  const std::optional<std::vector<ConSanMoiAccessRange>> access_ranges =
      candidate_access_ranges(bytes, candidate);
  if (!access_ranges || access_ranges->empty()) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires a supported LDS access range");
    return std::nullopt;
  }

  const uint16_t address_lo_vgpr = *options.scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 1u);
  const uint16_t low_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t high_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t old_value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  const std::optional<uint16_t> saved_lds_byte_offset_vgpr =
      moi_load_clobbers_address(candidate)
          ? std::optional<uint16_t>(
                static_cast<uint16_t>(*options.scratch_vgpr + inline_shadow_scratch_count(options) +
                                      inline_shadow_loop_scratch_count(options, candidate)))
          : std::nullopt;

  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  const uint32_t low_literal = static_cast<uint32_t>(kind);
  const uint32_t high_literal = static_cast<uint32_t>(
      (candidate.text_offset & consan_moi_exact_shadow::max_instruction_offset)
      << (consan_moi_exact_shadow::instruction_offset_shift - 32u));
  const uint64_t exact_shadow_base =
      *options.moi_report_buffer_address + layout.exact_shadow_entries_offset;

  std::vector<uint32_t> words;
  words.reserve(candidate.size / sizeof(uint32_t) + 64u + (options.moi_exec_save_sgpr ? 120u : 0u) +
                (options.moi_epoch_vgpr ? 2u : 0u));
  // A DS load may use the same VGPR for its address and destination. Preserve
  // the effective address before executing the displaced guest instruction;
  // otherwise the loaded payload becomes the exact-shadow cell index.
  if (saved_lds_byte_offset_vgpr) {
    words.push_back(build_v_mov_b32_e32(*saved_lds_byte_offset_vgpr,
                                        vector_source_vgpr(*lds_byte_offset_vgpr), arch));
    lds_byte_offset_vgpr = saved_lds_byte_offset_vgpr;
  }
  for (uint64_t offset = 0; offset < candidate.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  // Preserve application condition state before metadata construction. Owner,
  // epoch, address, and partition setup all use VCC/SCC-producing
  // instructions; snapshotting inside the exchange loop would retain those
  // generated values rather than the state at the displaced LDS access.
  const bool preserve_special_state = options.moi_exec_save_sgpr.has_value();
  if (preserve_special_state && !append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not preserve application VCC/SCC");
    return std::nullopt;
  }
  words.push_back(kWaitDscnt0);

  const uint16_t workgroup_key_vgpr = old_value_vgpr;
  const uint16_t original_exec_save_offset = 20u;
  if (options.moi_exec_save_sgpr &&
      !append_inline_workgroup_key(words, workgroup_sources, options, workgroup_key_vgpr, tmp_vgpr,
                                   address_lo_vgpr, original_exec_save_offset, arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not derive workgroup identity");
    return std::nullopt;
  }
  if (options.moi_exec_save_sgpr) {
    // The packed key is injective only inside its documented dimensional
    // bounds. Keep the guest running, but make every excluded lane-site
    // encounter observable in the Inline-only header flags/count field so a
    // clean acceptance run cannot silently pass with detector undercoverage.
    const uint16_t exec_base = *options.moi_exec_save_sgpr;
    const auto save_valid_exec = build_s_mov_b64(exec_base, kRdna4ExecLo, arch);
    const auto select_invalid_exec = build_s_and_not1_b64(
        kRdna4ExecLo, static_cast<uint16_t>(exec_base + original_exec_save_offset), exec_base,
        arch);
    const auto restore_valid_exec = build_s_mov_b64(kRdna4ExecLo, exec_base, arch);
    std::vector<uint32_t> accounting_words;
    if (!save_valid_exec || !select_invalid_exec || !restore_valid_exec ||
        !append_atomic_fetch_add_one_u32(
            accounting_words,
            *options.moi_report_buffer_address +
                offsetof(ConSanMoiReportHeader, inline_unsupported_count),
            tmp_vgpr, address_lo_vgpr, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not account unsupported workgroup identity");
      return std::nullopt;
    }
    words.push_back(*save_valid_exec);
    words.push_back(*select_invalid_exec);
    words.insert(words.end(), accounting_words.begin(), accounting_words.end());
    words.push_back(*restore_valid_exec);
  }
  if (workgroup_shadow) {
    const auto retain_valid_workgroup = build_s_mov_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 22u), kRdna4ExecLo, arch);
    if (!retain_valid_workgroup) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not retain validated workgroup lanes");
      return std::nullopt;
    }
    words.push_back(*retain_valid_workgroup);
  }

  const auto mov_low = build_v_mov_b32_e64_literal(low_vgpr, low_literal, arch);
  const auto mov_high = build_v_mov_b32_e64_literal(high_vgpr, high_literal, arch);
  if (!mov_low || !mov_high) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode exact-shadow literals");
    return std::nullopt;
  }
  words.insert(words.end(), mov_low->begin(), mov_low->end());
  if (!append_inline_shadow_owner_field(words, options, low_vgpr, tmp_vgpr, arch,
                                        private_owner_offset, private_owner_shift, errors)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode owner field");
    return std::nullopt;
  }
  if (!append_inline_shadow_epoch_field(words, options, private_epoch_offset, low_vgpr, tmp_vgpr,
                                        arch, errors)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode epoch field");
    return std::nullopt;
  }
  words.insert(words.end(), mov_high->begin(), mov_high->end());
  if (options.moi_exec_save_sgpr &&
      !append_add_exact_shadow_generation(words, low_vgpr, high_vgpr, workgroup_key_vgpr, tmp_vgpr,
                                          arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode workgroup identity");
    return std::nullopt;
  }

  for (const ConSanMoiAccessRange &range : *access_ranges) {
    const ConSanMoiLdsCellRange cell_range =
        consan_moi_lds_cell_range_for_bytes(range.static_byte_offset, range.byte_count);
    for (uint32_t cell_index = 0; cell_index < cell_range.cell_count; ++cell_index) {
      if (workgroup_shadow) {
        if (cell_range.cell_count > 1u) {
          if (cell_index != 0u)
            continue;
          const uint16_t loop_counter_vgpr =
              static_cast<uint16_t>(*options.scratch_vgpr + inline_shadow_scratch_count(options));
          const uint16_t loop_offset_vgpr = static_cast<uint16_t>(loop_counter_vgpr + 1u);
          if (range.static_byte_offset == 0u) {
            words.push_back(build_v_mov_b32_e32(loop_offset_vgpr,
                                                vector_source_vgpr(*lds_byte_offset_vgpr), arch));
          } else if (!append_compute_effective_lds_byte_offset(words, loop_offset_vgpr,
                                                               *lds_byte_offset_vgpr,
                                                               range.static_byte_offset, arch)) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not initialize its local cell loop");
            return std::nullopt;
          }
          words.push_back(
              build_v_mov_b32_e32(loop_counter_vgpr, scalar_positive_inline_u32(0), arch));
          const size_t loop_begin = words.size();
          if (!append_workgroup_local_exact_shadow_swap(
                  words, candidate, options, arch, layout, *workgroup_shadow, address_lo_vgpr,
                  low_vgpr, high_vgpr, old_value_vgpr, loop_offset_vgpr,
                  /*static_byte_offset=*/0u, /*cell_index=*/0u, range.byte_count,
                  *lds_byte_offset_vgpr, range.static_byte_offset)) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode looped local LDS shadow publish");
            return std::nullopt;
          }
          const auto advance_offset = build_v_add_nc_u32_e32(
              loop_offset_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_bytes),
              loop_offset_vgpr, arch);
          const auto advance_counter = build_v_add_nc_u32_e32(
              loop_counter_vgpr, scalar_positive_inline_u32(1), loop_counter_vgpr, arch);
          const auto more_cells = build_v_cmp_gt_u32_e32_vcc(
              scalar_positive_inline_u32(cell_range.cell_count), loop_counter_vgpr, arch);
          const uint16_t loop_exec_save_sgpr =
              static_cast<uint16_t>(*options.moi_exec_save_sgpr + 14u);
          const uint16_t valid_workgroup_exec_sgpr =
              static_cast<uint16_t>(*options.moi_exec_save_sgpr + 22u);
          const auto narrow_more_cells =
              build_s_and_saveexec_b64(loop_exec_save_sgpr, kRdna4VccLo, arch);
          const auto restore_valid_workgroup =
              build_s_mov_b64(kRdna4ExecLo, valid_workgroup_exec_sgpr, arch);
          if (!advance_offset || !advance_counter || !more_cells || !narrow_more_cells ||
              !restore_valid_workgroup) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its local cell loop control");
            return std::nullopt;
          }
          words.push_back(*advance_offset);
          words.push_back(*advance_counter);
          words.push_back(*more_cells);
          // Vector compares only define VCC for active lanes. Branching on raw
          // VCC can therefore observe stale guest bits from inactive lanes and
          // loop forever. Narrow EXEC to the active lanes that still need a
          // cell, branch on that mask, then restore the validated workgroup
          // mask on loop exit.
          words.push_back(*narrow_more_cells);
          const size_t loop_branch = words.size();
          const int64_t loop_delta =
              static_cast<int64_t>(loop_begin) - static_cast<int64_t>(loop_branch + 1u);
          if (loop_delta < std::numeric_limits<int16_t>::min() ||
              loop_delta > std::numeric_limits<int16_t>::max()) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its local cell loop branch");
            return std::nullopt;
          }
          const auto loop = build_s_cbranch_execnz(static_cast<int16_t>(loop_delta), arch);
          if (!loop) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its local cell loop branch");
            return std::nullopt;
          }
          words.push_back(*loop);
          words.push_back(*restore_valid_workgroup);
        } else if (!append_workgroup_local_exact_shadow_swap(
                       words, candidate, options, arch, layout, *workgroup_shadow, address_lo_vgpr,
                       low_vgpr, high_vgpr, old_value_vgpr, *lds_byte_offset_vgpr,
                       range.static_byte_offset, cell_index, range.byte_count,
                       *lds_byte_offset_vgpr, range.static_byte_offset)) {
          errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode local LDS shadow publish");
          return std::nullopt;
        }
        continue;
      }
      const auto mov_address_lo = build_v_mov_b32_e64_literal(
          address_lo_vgpr, static_cast<uint32_t>(exact_shadow_base), arch);
      const auto mov_address_hi = build_v_mov_b32_e64_literal(
          address_hi_vgpr, static_cast<uint32_t>(exact_shadow_base >> 32u), arch);
      if (!mov_address_lo || !mov_address_hi) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow address");
        return std::nullopt;
      }
      words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
      words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());

      const uint32_t entries_per_dispatch_bank =
          layout.exact_shadow_entry_capacity / kConSanMoiInlineExactDispatchBankCount;
      const uint32_t dispatch_bank_stride =
          entries_per_dispatch_bank * sizeof(ConSanMoiInlineExactShadowSlot);
      const uint32_t copy_dispatch_id =
          build_v_mov_b32_e32(tmp_vgpr, *options.moi_dispatch_id_sgpr, arch);
      const auto select_dispatch_bank = build_v_and_b32_e32_literal(
          tmp_vgpr, kConSanMoiInlineExactDispatchBankCount - 1u, tmp_vgpr, arch);
      const auto scale_dispatch_bank =
          build_v_mul_lo_u32_vop3_literal(tmp_vgpr, dispatch_bank_stride, tmp_vgpr, arch);
      const auto add_dispatch_bank = build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch);
      if (entries_per_dispatch_bank == 0u || dispatch_bank_stride == 0u || !select_dispatch_bank ||
          !scale_dispatch_bank || !add_dispatch_bank) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode dispatch bank");
        return std::nullopt;
      }
      words.push_back(copy_dispatch_id);
      words.insert(words.end(), select_dispatch_bank->begin(), select_dispatch_bank->end());
      words.insert(words.end(), scale_dispatch_bank->begin(), scale_dispatch_bank->end());
      words.insert(words.end(), add_dispatch_bank->begin(), add_dispatch_bank->end());

      uint16_t effective_lds_byte_offset_vgpr = *lds_byte_offset_vgpr;
      if (range.static_byte_offset != 0) {
        if (!append_compute_effective_lds_byte_offset(words, tmp_vgpr, *lds_byte_offset_vgpr,
                                                      range.static_byte_offset, arch)) {
          errors.emplace_back("ConSan MOI inline-shadow probe could not encode LDS byte offset");
          return std::nullopt;
        }
        effective_lds_byte_offset_vgpr = tmp_vgpr;
      }

      const auto start_cell_shift = build_v_lshrrev_b32_e32(
          tmp_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
          effective_lds_byte_offset_vgpr, arch);
      const auto byte_index_shift =
          build_v_lshlrev_b32_e32(tmp_vgpr, scalar_positive_inline_u32(3), tmp_vgpr, arch);
      const auto address_with_low_cell =
          build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch);
      const auto high_cell_shift =
          build_v_lshlrev_b32_e32(tmp_vgpr, scalar_positive_inline_u32(1), tmp_vgpr, arch);
      const auto address_with_high_cell =
          build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch);
      if (!start_cell_shift || !byte_index_shift || !address_with_low_cell || !high_cell_shift ||
          !address_with_high_cell) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow publish");
        return std::nullopt;
      }
      words.push_back(*start_cell_shift);
      words.push_back(*byte_index_shift);
      words.insert(words.end(), address_with_low_cell->begin(), address_with_low_cell->end());
      words.push_back(*high_cell_shift);
      words.insert(words.end(), address_with_high_cell->begin(), address_with_high_cell->end());
      if (cell_index != 0) {
        const auto add_cell = build_v_add_u64_signed_i24(
            address_lo_vgpr,
            static_cast<int32_t>(cell_index * sizeof(ConSanMoiInlineExactShadowSlot)), arch);
        if (!add_cell) {
          errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow cell offset");
          return std::nullopt;
        }
        words.insert(words.end(), add_cell->begin(), add_cell->end());
      }
      if (!append_wave_coalesced_exact_shadow_swap(
              words, candidate, options, arch, layout, address_lo_vgpr, low_vgpr, high_vgpr,
              old_value_vgpr, *lds_byte_offset_vgpr, range.static_byte_offset, range.byte_count)) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not encode versioned shadow publish");
        return std::nullopt;
      }
    }
  }
  if (workgroup_shadow) {
    const uint16_t exec_base = *options.moi_exec_save_sgpr;
    const uint16_t active_exec_sgpr = static_cast<uint16_t>(exec_base + 12u);
    const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(exec_base + 14u);
    const auto save_active_exec = build_s_mov_b64(active_exec_sgpr, kRdna4ExecLo, arch);
    const auto mbcnt_lo =
        build_v_mbcnt_lo_u32_b32(tmp_vgpr, active_exec_sgpr, scalar_positive_inline_u32(0), arch);
    const auto mbcnt_hi = build_v_mbcnt_hi_u32_b32(
        tmp_vgpr, static_cast<uint16_t>(active_exec_sgpr + 1u), vector_source_vgpr(tmp_vgpr), arch);
    const auto first_active =
        build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
    const auto narrow_first = build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
    if (!save_active_exec || !mbcnt_lo || !mbcnt_hi || !first_active || !narrow_first) {
      errors.emplace_back("ConSan MOI inline-shadow probe could not elect visible evidence");
      return std::nullopt;
    }
    words.push_back(*save_active_exec);
    words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
    words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
    words.push_back(*first_active);
    words.push_back(*narrow_first);
    if (!append_publish_visible_evidence_if_zero(
            words,
            *options.moi_report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter),
            tmp_vgpr, address_lo_vgpr, temporary_exec_sgpr, arch)) {
      errors.emplace_back("ConSan MOI inline-shadow probe could not publish visible evidence");
      return std::nullopt;
    }
  }
  if (options.moi_exec_save_sgpr) {
    const auto restore_original_exec = build_s_mov_b64(
        kRdna4ExecLo,
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + original_exec_save_offset), arch);
    if (!restore_original_exec) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not restore workgroup-filtered state");
      return std::nullopt;
    }
    words.push_back(*restore_original_exec);
    if (preserve_special_state && !append_restore_moi_special_state(words, options, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not restore workgroup-filtered state");
      return std::nullopt;
    }
  }
  return words;
}

[[nodiscard]] bool append_moi_scc_preserving_indirect_jump(
    std::vector<uint32_t> &words, uint64_t words_text_offset, uint64_t target_text_offset,
    uint16_t pc_sgpr, uint16_t saved_scc_sgpr, bool capture_scc, rj_code_arch_t arch);
