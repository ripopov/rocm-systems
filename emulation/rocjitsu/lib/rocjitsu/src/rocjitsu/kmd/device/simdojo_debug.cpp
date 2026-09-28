// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/simdojo_device_backend.h"
#include "rocjitsu/kmd/linux/cwsr.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/amdgpu/iod.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"
#include "util/log.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "uapi/linux/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <cerrno>
#include <unistd.h>
#include <unordered_set>

namespace rocjitsu {

namespace {

uint32_t gfx1250_status(uint32_t status) {
  // SCC, priorities and HALT moved to STATE_PRIV. The remaining fields used by
  // dbgapi retain their gfx9 positions.
  return status & ~((0x1Fu << 0) | (1u << 7) | (1u << 13) | (1u << 19));
}

uint32_t internal_status_from_gfx1250(const kmd::CwsrWaveState &state) {
  uint32_t status = amdgpu::update_status_from_gfx12_state_priv(state.status, state.state_priv);
  status &= ~amdgpu::Wavefront::kStatusHaltMask;
  if (state.saved_status_halt)
    status |= amdgpu::Wavefront::kStatusHaltMask;
  return status;
}

uint32_t internal_trapsts_from_gfx1250(const kmd::CwsrWaveState &state) {
  uint32_t trapsts = amdgpu::update_trapsts_from_gfx12_excp_flag_priv(0, state.excp_flag_priv);
  return amdgpu::update_trapsts_from_gfx12_excp_flag_user(trapsts, state.excp_flag_user);
}

kmd::CwsrWaveState build_cwsr_wave_state(amdgpu::Wavefront &wf, rj_code_arch_t arch,
                                         bool runtime_save = false) {
  kmd::CwsrWaveState state;
  const bool gfx12_5 = kmd::cwsr_layout_kind(arch) == kmd::CwsrLayoutKind::Gfx12_5;
  const uint32_t raw_status = wf.status_raw();
  state.pc = wf.pc;
  if (runtime_save && wf.in_trap_handler() && wf.fatal_exception_pending()) {
    const uint32_t pc_hi_mask = gfx12_5 ? 0x01FFFFFFu : 0x0000FFFFu;
    state.pc = wf.ttmp(0) | (static_cast<uint64_t>(wf.ttmp(1) & pc_hi_mask) << 32);
  }
  // A wave frozen inside the trap handler has the HANDLER's EXEC live, not the
  // application's: the handler runs with its own mask and parks 0x80000000 (and
  // then a doorbell id) in EXEC_LO around MSG_GET_DOORBELL. Publishing that
  // makes the debugger report every lane active, which inverts `lane apply
  // -active/-inactive` (gdb.rocm/lane-info.exp). trap_saved_exec_ holds the
  // interrupted value for the whole handler, so un-shadow it here, as the
  // STATUS field below does with trap_saved_status().
  //
  // in_trap_handler() is the predicate for both, and it opens at handler entry
  // rather than at MSG_INTERRUPT because the doorbell park happens on the way
  // there. It closes at s_rfe, which puts the interrupted values back, so from
  // then on the live registers are the application's again.
  //
  // The record is not write-only -- apply_cwsr_to_wave() feeds it back on
  // resume -- so un-shadowing here is only safe because that path shadows in
  // the other direction under the same predicate: a debugger edit to EXEC is
  // written to trap_saved_exec_, which the handler's s_rfe installs, instead of
  // overwriting the mask a mid-flight handler is still running under.
  state.exec = wf.in_trap_handler() ? wf.trap_saved_exec() : wf.exec();
  state.vcc = wf.vcc();
  state.flat_scratch = wf.scratch_base();
  state.m0 = wf.m0();
  state.mode = wf.mode_raw();
  state.trapsts = wf.trapsts();
  // Whether the application was already halted when it trapped outlives the
  // handler: trap_saved_status_ still describes the interrupted wave after
  // s_rfe, and trap_interrupt_sent() is the flag that says a handler produced
  // this stop. in_trap_handler() has to be part of the predicate too, or the
  // two halves of the published STATUS come from different registers between
  // trap entry and MSG_INTERRUPT -- the body from trap_saved_status_ and the
  // HALT bit from the handler's live one. apply_cwsr_to_wave() recombines them
  // into a word that never existed on the wave, so an identity round-trip in
  // that window rewrites the application's saved HALT with the handler's.
  const bool handler_context = wf.in_trap_handler() || wf.trap_interrupt_sent();
  state.saved_status_halt = ((handler_context ? wf.trap_saved_status() : raw_status) >> 13) & 1u;
  state.wave_stopped = wf.debug_halted() || (runtime_save && wf.fatal_exception_pending());
  // The published STATUS shadows on the narrower predicate, the same one EXEC
  // uses: only while the handler is actually running is the live register the
  // handler's rather than the application's. Once s_rfe has run, the live value
  // is the application's again and is what the debugger should see.
  const uint32_t application_status = wf.in_trap_handler() ? wf.trap_saved_status() : raw_status;
  state.status =
      state.wave_stopped ? application_status | (1u << 13) : application_status & ~(1u << 13);
  if (gfx12_5) {
    uint32_t debug_status = application_status;
    if (state.wave_stopped)
      debug_status |= amdgpu::Wavefront::kStatusHaltMask;
    else
      debug_status &= ~amdgpu::Wavefront::kStatusHaltMask;
    state.state_priv = amdgpu::gfx12_state_priv_from_status(debug_status, state.flat_scratch != 0);
    state.status = gfx1250_status(application_status);
    state.excp_flag_priv = amdgpu::gfx12_excp_flag_priv_from_trapsts(state.trapsts);
    state.excp_flag_user = (state.trapsts & 0x7Fu) | wf.gfx12_excp_flag_user_extra_raw();
    state.trap_ctrl = wf.gfx12_trap_ctrl_raw();
    state.xnack_state_priv = wf.gfx1250_xnack_state_priv_raw();
    state.xnack_mask = wf.gfx1250_xnack_mask_raw();
  }
  state.trap_id = wf.trap_id();
  state.wave_id = wf.debug_wave_id();
  state.group_ids = wf.wg_coord();
  state.wave_in_group = wf.wave_in_group();
  state.queue_packet_id = wf.queue_packet_id() & 0x1FFFFFFu;
  state.scratch_scoreboard_id = wf.scratch_scoreboard_id();
  state.shader_engine_id = wf.shader_engine_id();
  state.spi_ttmps_setup = true;
  state.num_sgprs = wf.num_sgprs();
  state.num_vgprs = wf.num_vgprs();

  state.sgprs.resize(state.num_sgprs);
  for (uint32_t s = 0; s < state.num_sgprs; ++s)
    state.sgprs[s] = wf.debug_read_sgpr(s);
  // The CWSR record's VGPR stride is wave64-shaped (cwsr.cpp kVgprLaneBytes)
  // whatever the wave size is, but the physical register only has wf_size()
  // lanes -- a wave32 VectorReg<32> is exactly 32 wide, so reading lanes 32-63
  // runs off it into the neighbouring register. Read the live lanes and leave
  // the rest of the wave64-shaped slot at the zero resize() already wrote; the
  // restore path is bounded by wf_size() too and never looks at them.
  state.vgprs.resize(static_cast<size_t>(state.num_vgprs) * 64);
  const uint32_t live_lanes = wf.wf_size();
  for (uint32_t r = 0; r < state.num_vgprs; ++r)
    for (uint32_t lane = 0; lane < live_lanes; ++lane)
      state.vgprs[static_cast<size_t>(r) * 64 + lane] = wf.debug_read_vgpr(r, lane);
  state.lds.resize(wf.lds_size());
  if (!state.lds.empty())
    static_cast<const amdgpu::Lds &>(wf.lds()).read(wf.lds_base(), state.lds.data(),
                                                    static_cast<uint32_t>(state.lds.size()));
  return state;
}

void prepare_cwsr_wave_states(std::vector<kmd::CwsrWaveState> &waves) {
  std::ranges::sort(waves, [](const auto &lhs, const auto &rhs) {
    if (lhs.queue_packet_id != rhs.queue_packet_id)
      return lhs.queue_packet_id < rhs.queue_packet_id;
    if (lhs.group_ids != rhs.group_ids)
      return lhs.group_ids < rhs.group_ids;
    return lhs.wave_in_group < rhs.wave_in_group;
  });
  auto same_group = [](const auto &lhs, const auto &rhs) {
    return lhs.queue_packet_id == rhs.queue_packet_id && lhs.group_ids == rhs.group_ids;
  };
  for (size_t index = 0; index < waves.size(); ++index) {
    waves[index].is_first_in_group = index == 0 || !same_group(waves[index], waves[index - 1]);
    waves[index].is_last_in_group =
        index + 1 == waves.size() || !same_group(waves[index], waves[index + 1]);
    if (!waves[index].is_first_in_group)
      waves[index].lds.clear();
  }
}

void apply_cwsr_to_wave(amdgpu::Wavefront &wave, const kmd::CwsrWaveState &state,
                        rj_code_arch_t arch) {
  constexpr uint32_t kModeDebugEnMask = 1u << 11;
  constexpr uint32_t kStatusHaltMask = amdgpu::Wavefront::kStatusHaltMask;
  const bool gfx12_5 = kmd::cwsr_layout_kind(arch) == kmd::CwsrLayoutKind::Gfx12_5;
  wave.pc = state.pc;
  // Mirror of the un-shadowing in build_cwsr_wave_state(): while a handler is
  // mid-flight the published EXEC is the interrupted application mask, so the
  // value coming back belongs to the shadow, not to the live register the
  // handler is executing under. s_rfe installs it when the handler returns.
  if (wave.in_trap_handler())
    wave.set_trap_saved_exec(state.exec);
  else
    wave.set_exec(state.exec);
  wave.set_vcc_raw(state.vcc);
  wave.set_m0(state.m0);
  // STATUS shadows the same way EXEC does, and the cost of getting it wrong is
  // higher: the ROCr handler raises STATUS.HALT and only then returns, so a
  // record applied in that window used to clear the HALT the handler had just
  // set. s_rfe then saw a running wave, resumed it, and the breakpoint was
  // silently lost -- one inferior in gdb.rocm/multi-inferior-stress.exp running
  // its kernel to completion and exiting without ever stopping. Route the
  // record's HALT to the interrupted state the handler restores and leave the
  // live register to the handler.
  const uint32_t restored_status = gfx12_5 ? internal_status_from_gfx1250(state)
                                           : (state.status & ~kStatusHaltMask) |
                                                 (state.saved_status_halt ? kStatusHaltMask : 0u);
  if (wave.in_trap_handler())
    wave.set_trap_saved_status(restored_status);
  else
    wave.set_status_raw(restored_status);
  // The live STATUS.HALT stays as it is for a handler that raised it with
  // s_setreg: there the bit is the handler's own request to keep the wave
  // stopped, and clearing it on a resume loses the breakpoint --
  // DbgTrapCwsrShadowsTrapHandlerRegistersAndRoutesDebuggerEdits pins that.
  // A wave halted at s_sendmsghalt sits in the same window and wants the
  // opposite: it has already reported and is waiting to be let go, so leaving
  // the bit set strands it at the handler return. The record cannot tell the
  // two apart, which is why the wave records who raised the bit.
  if (!state.wave_stopped && wave.self_halted()) {
    wave.set_status_halt(false);
    wave.set_self_halted(false);
  }
  wave.set_mode_raw(state.mode);
  wave.set_trapsts(gfx12_5 ? internal_trapsts_from_gfx1250(state) : state.trapsts);
  if (gfx12_5) {
    wave.set_gfx12_excp_flag_user_extra_raw(state.excp_flag_user & (0x3u << 30));
    wave.set_gfx12_trap_ctrl_raw(state.trap_ctrl);
    wave.set_gfx1250_xnack_state_priv_raw(state.xnack_state_priv);
    wave.set_gfx1250_xnack_mask_raw(state.xnack_mask);
  }
  wave.set_debug_wave_id(state.wave_id);
  wave.set_queue_packet_id(state.queue_packet_id);
  const uint32_t stack_pointer = wave.debug_read_sgpr(32);
  const uint32_t stack_frame = wave.debug_read_sgpr(33);
  // rocm-dbgapi may submit a lightweight control-state update with no register
  // payload after displaced stepping. Preserve the SGPR/VGPR values produced
  // by the displaced instruction in that case rather than restoring zeros.
  //
  // Skip exactly the slots the codec treats as aliases, not everything above
  // the architected count: s104/s105 sit between the FLAT_SCRATCH and VCC
  // aliases and are ordinary registers the debugger may edit. Sharing
  // cwsr_sgpr_slot_is_aliased() with the codec is what keeps the two ends of
  // the round trip from disagreeing about which slots carry register state.
  if (!state.sgprs.empty())
    for (uint32_t s = 0; s < state.num_sgprs && s < state.sgprs.size(); ++s)
      if (s != 32 && s != 33 && !kmd::cwsr_sgpr_slot_is_aliased(s, arch))
        wave.debug_write_sgpr(s, state.sgprs[s]);
  if (!state.vgprs.empty())
    for (uint32_t r = 0; r < state.num_vgprs; ++r)
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
        const size_t index = static_cast<size_t>(r) * 64 + lane;
        if (index < state.vgprs.size())
          wave.debug_write_vgpr(r, lane, state.vgprs[index]);
      }
  // FLAT_SCRATCH travels in its own field because it aliases two SGPR slots the
  // loop above skips. This used to save and re-install wave.scratch_base()
  // around the loop, which the loop cannot reach anyway (scratch_base_ is a
  // standalone member), so the only effect was discarding the debugger's edit.
  wave.set_scratch_base(state.flat_scratch);
  wave.debug_write_sgpr(32, stack_pointer);
  wave.debug_write_sgpr(33, stack_frame);
  const bool single_step = !state.wave_stopped && (gfx12_5 ? (state.trap_ctrl & (1u << 9)) != 0
                                                           : (state.mode & kModeDebugEnMask) != 0);
  wave.set_debug_single_step(single_step);
  wave.set_debug_halted(state.wave_stopped);
  // debug_suspended is deliberately left set here. Clearing it per wave made
  // each wave runnable the moment its own record was applied, so the engine
  // could execute it while its queue-mates -- and their LDS images -- were
  // still being restored. resume_debug_queues() commits the bit for every wave
  // in the queue in one pass once the whole restore has succeeded.
}

} // namespace

bool kmd::SimdojoDeviceBackend::save_queue(const DebugQueueContext &context) {
  const uint32_t process_id = context.process_id, queue_id = context.queue_id;
  const uint64_t ctx_base = context.base;
  const uint32_t ctx_size = context.size;
  const bool runtime_save = context.runtime_save;
  if (ctx_base == 0)
    return false;
  // Never publish a record shaped for the wrong architecture. Handlers ask
  // debug_stop_publishable() before claiming a stop, so reaching here on an
  // unmodelled part means a caller skipped the gate; refuse rather than hand
  // rocm-dbgapi an image it would decode against its own layout.
  if (!kmd::cwsr_layout_modelled(soc_.arch()))
    return false;

  const uint32_t area_count = kmd::cwsr_area_count(soc_.arch(), soc_.num_xcds());
  std::vector<std::vector<kmd::CwsrWaveState>> waves_by_area(area_count);
  bool memory_pending = false;
  auto collect = [&](amdgpu::CommandProcessor *cp, uint32_t area) {
    for (auto *cu : cp->compute_units()) {
      cu->with_wave_state_locked([&] {
        for (uint32_t i = 0; i < cu->num_wf_slots(); ++i) {
          auto *wave = cu->wf(i);
          if (wave && (wave->debug_stopped() || (runtime_save && wave->runtime_suspended())) &&
              wave->process_id() == process_id && wave->queue_id() == queue_id) {
            // Pausing instruction issue does not cancel retained memory work.
            // Counters remain held through deferred architectural writeback;
            // refuse a snapshot until every paused wave has settled.
            if (!wave->wait_counters().empty()) {
              memory_pending = true;
              continue;
            }
            waves_by_area[area].push_back(build_cwsr_wave_state(*wave, soc_.arch(), runtime_save));
          }
        }
      });
    }
  };
  for (uint32_t xcc_id = 0; xcc_id < area_count; ++xcc_id)
    collect(soc_.xcd(xcc_id)->command_processor(), xcc_id);
  if (memory_pending)
    return false;
  if (runtime_save) {
    soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units())
        cu->with_wave_state_locked([&] { cu->flush_l1(process_id); });
    });
    for (uint32_t xcc = 0; xcc < soc_.num_xcds(); ++xcc)
      if (soc_.xcd(xcc)->l2_cache()->flush_all(process_id) != amdgpu::VmAccessOutcome::Complete)
        return false;
    for (uint32_t iod = 0; iod < soc_.num_iods(); ++iod)
      if (soc_.iod(iod)->msc()->flush_all() != amdgpu::VmAccessOutcome::Complete)
        return false;
  }
  size_t wave_count = 0;
  for (const auto &waves : waves_by_area)
    wave_count += waves.size();
  if (wave_count == 0 && !runtime_save)
    return false;

  const uint32_t control_stack_capacity = context.control_stack_capacity;
  const std::optional<amdgpu::GpuVmAccess> vm_access = soc_.gpu_vm().snapshot_vmid(process_id);
  if (!vm_access)
    return false;
  bool publish_ok = true;
  auto write_block = [&](uint64_t address, std::span<const uint8_t> bytes) {
    if (context.target_mem_fd < 0) {
      // A refused write publishes nothing, so reporting success would tell the
      // debugger a context is available at an address that holds none of it.
      // The pwrite path below fails publication on a short write for the same
      // reason; this is that failure arriving through the memory model.
      if (vm_access->write(address, std::as_bytes(bytes)) != amdgpu::VmAccessOutcome::Complete) {
        publish_ok = false;
        util::Logger::warn("CWSR target write faulted: addr=0x", std::hex, address, std::dec,
                           " pid=", context.client_pid);
      }
      return;
    }
    const ssize_t written =
        pwrite(context.target_mem_fd, bytes.data(), bytes.size(), static_cast<off_t>(address));
    if (written != static_cast<ssize_t>(bytes.size())) {
      publish_ok = false;
      util::Logger::warn("CWSR target write failed: addr=0x", std::hex, address, " pid=", std::dec,
                         context.client_pid, " rc=", written, " errno=", errno);
    }
  };
  bool layouts_ok = true;
  // Wave-state publication owns only the first four header words. libhsakmt
  // owns DebugOffset/DebugSize, ErrorReason/ErrorEventId and the reserved tail.
  const std::array<uint8_t, 4 * sizeof(uint32_t)> empty_header{};
  for (uint32_t area = 0; area < area_count; ++area) {
    const uint64_t area_base = ctx_base + static_cast<uint64_t>(area) * ctx_size;
    auto &waves = waves_by_area[area];
    if (waves.empty()) {
      write_block(area_base, empty_header);
      continue;
    }
    std::array<uint32_t, 2> debug_region{};
    const uint64_t debug_fields_address = area_base + 4 * sizeof(uint32_t);
    if (context.target_mem_fd >= 0) {
      if (pread(context.target_mem_fd, debug_region.data(), sizeof(debug_region),
                static_cast<off_t>(debug_fields_address)) != sizeof(debug_region))
        return false;
    } else if (vm_access->read(debug_fields_address,
                               std::as_writable_bytes(std::span(debug_region))) !=
               amdgpu::VmAccessOutcome::Complete) {
      return false;
    }
    prepare_cwsr_wave_states(waves);
    layouts_ok &= kmd::serialize_queue_cwsr_bulk(
                      area_base, ctx_size, waves, write_block, soc_.arch(), control_stack_capacity,
                      kmd::CwsrDebugRegion{debug_region[0], debug_region[1]})
                      .ok;
  }
  if (!layouts_ok || !publish_ok)
    return false;

  return true;
}

int kmd::SimdojoDeviceBackend::wave_state(const DebugQueueContext &context, uint64_t destination,
                                          uint32_t &control_size, uint32_t &wave_size) {
  if (context.runtime_save) {
    // Pausing instruction issue leaves no runnable wave to drive retained
    // memory on a single-queue CU. Settle only the queue owned by this ioctl.
    soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units())
        cu->retry_paused_queue_memory(context.process_id, context.queue_id);
    });
    if (!save_queue(context))
      return -EIO;
  }
  const auto access = soc_.gpu_vm().snapshot_vmid(context.process_id);
  if (!access)
    return -EFAULT;
  const uint64_t total_bytes = uint64_t{context.size} * soc_.num_xcds();
  if (destination > std::numeric_limits<uint64_t>::max() - total_bytes)
    return -EFAULT;
  uint32_t first_control_size = 0, first_wave_size = 0;
  for (uint32_t xcc = 0; xcc < soc_.num_xcds(); ++xcc) {
    const uint64_t offset = uint64_t{xcc} * context.size;
    const uint64_t source = context.base + offset;
    // KFD copies only wave-state fields; the caller owns the remaining header.
    std::array<uint32_t, 4> header{};
    if (access->read(source, std::as_writable_bytes(std::span(header))) !=
        amdgpu::VmAccessOutcome::Complete)
      return -EFAULT;
    if (header[0] > context.size || header[1] > context.size - header[0] ||
        header[2] > context.size || header[3] > header[2])
      return -EIO;
    if (xcc == 0) {
      // KFD returns XCC0 sizes; consumers inspect each area's own header.
      first_control_size = header[1];
      first_wave_size = header[3];
    }
    if (destination != context.base) {
      const uint64_t area_destination = destination + offset;
      std::vector<std::byte> stack(header[1]);
      auto write_user = [&](uint64_t address, std::span<const std::byte> bytes) {
        if (bytes.empty())
          return true;
        if (context.target_mem_fd >= 0)
          return pwrite(context.target_mem_fd, bytes.data(), bytes.size(),
                        static_cast<off_t>(address)) == static_cast<ssize_t>(bytes.size());
        iovec local{const_cast<std::byte *>(bytes.data()), bytes.size()};
        iovec remote{reinterpret_cast<void *>(address), bytes.size()};
        return process_vm_writev(context.client_pid, &local, 1, &remote, 1, 0) ==
               static_cast<ssize_t>(bytes.size());
      };
      if (access->read(source + header[0], stack) != amdgpu::VmAccessOutcome::Complete ||
          !write_user(area_destination, std::as_bytes(std::span(header))) ||
          !write_user(area_destination + header[0], stack))
        return -EFAULT;
    }
  }
  control_size = first_control_size;
  wave_size = first_wave_size;
  return 0;
}

bool kmd::SimdojoDeviceBackend::restore_queue(const DebugQueueContext &context) {
  std::vector<amdgpu::Wavefront *> stopped;
  std::vector<kmd::CwsrWaveState> states;
  std::vector<amdgpu::ComputeUnitCore *> owners;
  const bool per_xcc_areas = kmd::cwsr_layout_modelled(soc_.arch());
  const uint32_t area_count = kmd::cwsr_area_count(soc_.arch(), soc_.num_xcds());
  std::vector<std::vector<kmd::CwsrWaveState>> states_by_area(area_count);
  auto collect = [&](amdgpu::CommandProcessor *cp, uint32_t area) {
    for (auto *cu : cp->compute_units()) {
      cu->with_wave_state_locked([&] {
        for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
          auto *wave = cu->wf(slot);
          if (wave && wave->debug_stopped() && wave->process_id() == context.process_id &&
              wave->queue_id() == context.queue_id) {
            stopped.push_back(wave);
            states_by_area[area].push_back(build_cwsr_wave_state(*wave, soc_.arch()));
            owners.push_back(cu);
          }
        }
      });
    }
  };
  for (uint32_t xcc_id = 0; xcc_id < soc_.num_xcds(); ++xcc_id)
    collect(soc_.xcd(xcc_id)->command_processor(), per_xcc_areas ? xcc_id : 0u);
  bool restored = stopped.empty();
  if (!stopped.empty() && context.base != 0) {
    const std::optional<amdgpu::GpuVmAccess> vm_access =
        soc_.gpu_vm().snapshot_vmid(context.process_id);
    if (!vm_access)
      return false;
    bool read_ok = true;
    auto read_block = [&](uint64_t address, std::span<uint8_t> bytes) {
      if (context.target_mem_fd < 0) {
        // A refused read leaves the span zero-filled, which deserializes as a
        // valid-looking context: waves would resume with cleared registers and
        // a zero program counter rather than staying stopped. The pread path
        // below already treats a short read that way, and this is the same
        // failure arriving through the memory model instead of the kernel.
        if (vm_access->read(address, std::as_writable_bytes(bytes)) !=
            amdgpu::VmAccessOutcome::Complete) {
          read_ok = false;
          util::Logger::warn("CWSR target read faulted: addr=0x", std::hex, address, std::dec,
                             " pid=", context.client_pid);
        }
        return;
      }
      const ssize_t bytes_read =
          pread(context.target_mem_fd, bytes.data(), bytes.size(), static_cast<off_t>(address));
      if (bytes_read != static_cast<ssize_t>(bytes.size())) {
        read_ok = false;
        std::ranges::fill(bytes, 0);
        util::Logger::warn("CWSR target read failed: addr=0x", std::hex, address, " pid=", std::dec,
                           context.client_pid, " rc=", bytes_read, " errno=", errno);
      }
    };
    restored = true;
    states.reserve(stopped.size());
    for (uint32_t area = 0; area < area_count; ++area) {
      auto &area_states = states_by_area[area];
      if (area_states.empty())
        continue;
      prepare_cwsr_wave_states(area_states);
      const uint64_t area_base = context.base + static_cast<uint64_t>(area) * context.size;
      restored &= kmd::deserialize_queue_cwsr_bulk(area_base, context.size, area_states, read_block,
                                                   soc_.arch(), context.control_stack_capacity);
      states.insert(states.end(), std::make_move_iterator(area_states.begin()),
                    std::make_move_iterator(area_states.end()));
    }
    restored = restored && read_ok;
  }
  std::unordered_set<amdgpu::ComputeUnitCore *> wake;
  if (restored) {
    std::vector<size_t> matches;
    matches.reserve(states.size());
    std::vector<bool> matched(stopped.size());
    for (const auto &state : states) {
      auto find_match = [&](bool by_wave_id) {
        for (size_t index = 0; index < stopped.size(); ++index) {
          if (matched[index])
            continue;
          const auto *candidate = stopped[index];
          if (by_wave_id ? state.wave_id != 0 && candidate->debug_wave_id() == state.wave_id
                         : candidate->debug_wave_id() == 0 &&
                               candidate->queue_packet_id() == state.queue_packet_id &&
                               candidate->wg_coord() == state.group_ids &&
                               candidate->wave_in_group() == state.wave_in_group)
            return index;
        }
        return stopped.size();
      };
      size_t index = find_match(true);
      if (index == stopped.size())
        index = find_match(false);
      if (index == stopped.size()) {
        restored = false;
        break;
      }
      matched[index] = true;
      matches.push_back(index);
    }
    restored = restored && matches.size() == stopped.size();
    if (restored) {
      for (size_t state_index = 0; state_index < states.size(); ++state_index) {
        const size_t wave_index = matches[state_index];
        if (!states[state_index].lds.empty())
          stopped[wave_index]->lds().write(stopped[wave_index]->lds_base(),
                                           states[state_index].lds.data(),
                                           static_cast<uint32_t>(states[state_index].lds.size()));
        owners[wave_index]->with_wave_state_locked(
            [&] { apply_cwsr_to_wave(*stopped[wave_index], states[state_index], soc_.arch()); });
      }
    }
  }
  // This is the single commit point for debug_suspended: apply_cwsr_to_wave()
  // leaves every wave suspended, so none of them runs until the whole queue's
  // CWSR state and LDS have been restored above.
  //
  // Accumulated inside the locked loop below rather than read afterwards.
  // The loop clears debug_suspended, which makes these waves runnable, so
  // from that point the engine thread can be writing debug_halted_ -- a plain
  // bool -- concurrently. This value decides whether the queue gate reopens,
  // so reading it unlocked would race the answer as well as the byte.
  bool keep_dispatch_suspended = false;
  bool saw_runtime_exception = false;
  bool unresolved_runtime_exception = false;
  for (size_t index = 0; index < stopped.size(); ++index) {
    owners[index]->with_wave_state_locked([&] {
      const bool fatal_exception_pending = stopped[index]->fatal_exception_pending();
      const bool resolve_fatal_exception =
          restored && fatal_exception_pending && stopped[index]->fatal_exception_cwsr_valid();
      saw_runtime_exception |= fatal_exception_pending;
      unresolved_runtime_exception |= fatal_exception_pending && !resolve_fatal_exception;
      if (fatal_exception_pending)
        stopped[index]->set_fatal_exception_cwsr_valid(false);
      if (resolve_fatal_exception)
        stopped[index]->set_fatal_exception_pending(false);
      // A malformed or stale CWSR image must not strand a temporarily
      // suspended wave after the queue gate is released. Architecturally
      // halted waves remain halted until their CWSR record says otherwise.
      stopped[index]->set_debug_suspended(fatal_exception_pending && !resolve_fatal_exception);
      const bool halted = stopped[index]->debug_halted();
      keep_dispatch_suspended |= halted;
      if (!halted && !stopped[index]->is_halted()) {
        if (!fatal_exception_pending || resolve_fatal_exception)
          wake.insert(owners[index]);
      }
    });
  }
  if (saw_runtime_exception && !unresolved_runtime_exception && context.resolve_runtime_exceptions)
    context.resolve_runtime_exceptions();
  // Keep the queue-level launch gate closed until every resident wave has
  // consumed its CWSR state and become runnable. Reopen it before scheduling
  // those waves so CP completion processing cannot observe a queue as still
  // suspended if a resumed wave completes immediately.
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_queue_debug_suspended(context.queue_id, context.process_id, keep_dispatch_suspended,
                                  saw_runtime_exception && !unresolved_runtime_exception);
  });
  if (context.release_debug_image)
    context.release_debug_image();
  for (auto *cu : wake)
    cu->schedule_work_async();
  return restored;
}

bool kmd::SimdojoDeviceBackend::suspend_queue(uint32_t process_id, uint32_t queue_id,
                                              const std::function<bool()> &publish) {
  std::vector<std::pair<amdgpu::ComputeUnitCore *, amdgpu::Wavefront *>> newly_suspended;
  std::vector<std::pair<amdgpu::ComputeUnitCore *, amdgpu::Wavefront *>> runtime_frozen;
  bool needs_serialization = false;
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_queue_debug_suspended(queue_id, process_id, true);
    for (auto *cu : cp->compute_units()) {
      cu->with_wave_state_locked([&] {
        for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
          auto *wave = cu->wf(slot);
          if (!wave || wave->is_halted() || wave->process_id() != process_id ||
              wave->queue_id() != queue_id)
            continue;
          // Runtime exception routing freezes the wave with debug_suspended
          // so it cannot run, but it does not publish CWSR. Serialize that
          // retained fatal stop once; repeated SUSPEND_QUEUES calls must
          // preserve debugger edits in the already-published image.
          if (wave->fatal_exception_pending()) {
            runtime_frozen.emplace_back(cu, wave);
            needs_serialization |= !wave->fatal_exception_cwsr_valid();
          }
          if (wave->debug_suspended())
            continue;
          wave->set_debug_suspended(true);
          newly_suspended.emplace_back(cu, wave);
          needs_serialization = true;
        }
      });
    }
  });
  if (needs_serialization) {
    soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units())
        cu->retry_paused_queue_memory(process_id, queue_id, /*replay_loads=*/true);
    });
  }
  bool serialized = !needs_serialization || publish();
  if (serialized) {
    if (needs_serialization)
      for (auto &[cu, wave] : runtime_frozen)
        cu->with_wave_state_locked([wave] { wave->set_fatal_exception_cwsr_valid(true); });
    return true;
  }
  // No record was published, so the debugger has nothing to resume from.
  // Undo the suspension rather than leave the queue wedged: reporting the
  // error and stranding the waves is strictly worse than reporting it alone.
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_queue_debug_suspended(queue_id, process_id, false);
  });
  amdgpu::ComputeUnitCore *woken = nullptr;
  for (auto &[cu, wave] : newly_suspended) {
    cu->with_wave_state_locked([w = wave] { w->set_debug_suspended(false); });
    // Entries are appended compute unit by compute unit, so comparing against
    // the last one woken collapses a CU's waves into a single wake.
    if (cu != woken) {
      cu->schedule_work_async();
      woken = cu;
    }
  }
  return false;
}

void kmd::SimdojoDeviceBackend::clear_completed_queue(const DebugQueueContext &context) {
  const auto process_id = context.process_id;
  const auto queue_id = context.queue_id;
  bool has_stopped_wave = false;
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    for (auto *cu : cp->compute_units()) {
      cu->with_wave_state_locked([&] {
        for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
          const auto *wave = cu->wf(slot);
          if (wave && wave->debug_stopped() && wave->process_id() == process_id &&
              wave->queue_id() == queue_id)
            has_stopped_wave = true;
        }
      });
    }
  });
  if (!has_stopped_wave) {
    const auto access = soc_.gpu_vm().snapshot_vmid(process_id);
    const uint32_t area_count = kmd::cwsr_area_count(soc_.arch(), soc_.num_xcds());
    for (uint32_t area = 0; area < area_count; ++area) {
      const uint64_t area_base = context.base + static_cast<uint64_t>(area) * context.size;
      for (uint64_t offset = 0; offset < 4 * sizeof(uint32_t); offset += sizeof(uint32_t))
        if (access) {
          const uint32_t zero = 0;
          (void)access->write(area_base + offset, std::as_bytes(std::span(&zero, 1)));
        }
    }
  }
}

void kmd::SimdojoDeviceBackend::configure_debug_callbacks(const DeviceCallbacks &) {
  if (!kfd_)
    return;
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    for (auto *cu : cp->compute_units()) {
      cu->set_trap_completion_handler(
          [this](amdgpu::Wavefront &w) { kfd_->on_wave_trap_complete(w); });
      cu->set_sendmsg_handler(
          [this](amdgpu::Wavefront &w, uint32_t m) { return kfd_->on_wave_sendmsg(w, m); });
      cu->set_queue_exception_handler([this](uint32_t queue, uint32_t process, uint64_t status,
                                             bool retain,
                                             amdgpu::QueueExceptionDeliveryPolicy policy) {
        const bool delivered = kfd_->signal_runtime_queue_exception(
            kfd_->gpus_[ordinal_].gpu_id, queue, process, status,
            policy == amdgpu::QueueExceptionDeliveryPolicy::WaitForAcknowledgment);
        if (kfd_->runtime_exception_result_hook_for_testing_)
          kfd_->runtime_exception_result_hook_for_testing_(delivered);
        kfd_->complete_runtime_queue_exception(process, queue, status, delivered, retain);
        return delivered;
      });
      cu->set_single_step_handler(
          [this](amdgpu::Wavefront &w) { return kfd_->on_wave_single_step_complete(w); });
      cu->set_watchpoint_handler(
          [this](amdgpu::Wavefront &w, uint64_t a, uint32_t b, bool wr, bool at) {
            return kfd_->on_wave_watchpoint(w, a, b, wr, at);
          });
      cu->set_illegal_inst_handler(
          [this](amdgpu::Wavefront &w) { return kfd_->on_wave_illegal_instruction(w); });
      cu->set_execution_failure_handler([this](amdgpu::Wavefront &w, bool retain) {
        return kfd_->on_wave_execution_failure(w, retain);
      });
      cu->set_alu_exception_handler(
          [this](amdgpu::Wavefront &w) { return kfd_->on_wave_alu_exception(w); });
      cu->set_memory_violation_handler(
          [this](amdgpu::Wavefront &w, uint64_t a, bool wr, uint64_t pc) {
            return kfd_->on_wave_memory_violation(w, a, wr, pc);
          });
    }
  });
}

} // namespace rocjitsu
