// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/simdojo_device_backend.h"
#include "rocjitsu/kmd/linux/cwsr.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/soc.h"
#include "util/log.h"

#include <cerrno>

namespace rocjitsu {
namespace {

std::vector<SimulatedKfd::BackendDevice> make_devices(const std::vector<SoC *> &socs,
                                                      const std::vector<uint32_t> &gpu_ids) {
  std::vector<SimulatedKfd::BackendDevice> devices;
  for (size_t i = 0; i < socs.size(); ++i)
    devices.push_back({i < gpu_ids.size() ? gpu_ids[i] : socs[i]->gpu_id(),
                       std::make_unique<kmd::SimdojoDeviceBackend>(*socs[i], i)});
  return devices;
}

} // namespace

SimulatedKfd::SimulatedKfd(SoC &soc, bool daemon_mode,
                           DebugIdentityValidationHook debug_identity_validation_hook)
    : SimulatedKfd(make_devices({&soc}, {0}), daemon_mode,
                   std::move(debug_identity_validation_hook)) {
  for (auto &gpu : gpus_)
    static_cast<kmd::SimdojoDeviceBackend *>(gpu.backend.get())->bind_kfd(*this);
}

SimulatedKfd::SimulatedKfd(std::vector<SoC *> socs, std::vector<uint32_t> gpu_ids, bool daemon_mode,
                           DebugIdentityValidationHook debug_identity_validation_hook)
    : SimulatedKfd(make_devices(socs, gpu_ids), daemon_mode,
                   std::move(debug_identity_validation_hook)) {
  for (auto &gpu : gpus_)
    static_cast<kmd::SimdojoDeviceBackend *>(gpu.backend.get())->bind_kfd(*this);
}

int SimulatedKfd::submit_pm4(uint32_t render_minor, uint64_t queue_key,
                             amdgpu::Pm4Submission submission) {
  const auto process = find_process(local_process_id_);
  if (!process)
    return -ENODEV;
  std::lock_guard op_lock(process->op_mutex_);
  if (process->event_state_.is_closing())
    return -ENODEV;
  const auto ordinal = gpu_ordinal_for_render_minor(render_minor);
  if (!ordinal)
    return -ENODEV;
  auto *backend = dynamic_cast<kmd::SimdojoDeviceBackend *>(gpus_[*ordinal].backend.get());
  return backend ? backend->submit_pm4(local_process_id_, queue_key, std::move(submission))
                 : -ENOTSUP;
}

void SimulatedKfd::retire_pm4_queue(uint64_t queue_key) {
  for (auto &gpu : gpus_)
    if (auto *backend = dynamic_cast<kmd::SimdojoDeviceBackend *>(gpu.backend.get()))
      backend->retire_pm4_queue(queue_key);
}

void SimulatedKfd::on_wave_trap_complete(amdgpu::Wavefront &wave) {
  const uint32_t process_id = wave.process_id();
  const uint32_t queue_id = wave.queue_id();
  auto proc = find_process(process_id);
  if (!proc)
    return;
  auto fall_back_to_runtime = [&](bool runtime_already_reserved = false) {
    const uint64_t unreported =
        wave.trap_queue_exception_status() & ~wave.trap_runtime_exception_status();
    if (unreported != 0) {
      // Keep the retained debugger stop until ROCr actually accepts the
      // deferred exception. Its completion clears HALT only after successful
      // publication; failure leaves a future debugger able to claim the stop.
      defer_wave_exception_to_runtime(wave, unreported, /*suspend_while_pending=*/false,
                                      /*clear_debug_stop_on_success=*/true,
                                      runtime_already_reserved);
      return;
    }
    wave.set_debug_halted(false);
    wave.set_status_halt(false);
  };

  const uint64_t queue_exception_status = wave.trap_queue_exception_status();
  uint64_t debugger_status = 0;
  bool runtime_reserved = false;
  if ((queue_exception_status & wave.trap_runtime_exception_status()) == 0) {
    debugger_status = debugger_queue_exception_mask(proc, queue_id, queue_exception_status,
                                                    /*reserve_runtime_if_unclaimed=*/true);
    runtime_reserved = queue_exception_status != 0 && debugger_status == 0;
  }
  if (queue_exception_status != 0 && debugger_status == 0) {
    // The handler observed an attached debugger and requested HALT, but this
    // exception belongs to ROCr. Do not strand the wave as a debugger stop;
    // the deferred runtime route will apply the queue-wide fatal suspension.
    fall_back_to_runtime(runtime_reserved);
    return;
  }

  uint64_t ctx_base = 0;
  uint32_t gpu_id = 0;
  bool queue_found = false;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue != proc->queue_snapshot_map_.end()) {
      queue_found = true;
      ctx_base = queue->second.ctx_save_restore_address;
      gpu_id = queue->second.gpu_id;
    }
  }
  if (!queue_found) {
    fall_back_to_runtime();
    return;
  }
  if (ctx_base == 0) {
    fall_back_to_runtime();
    return;
  }
  // The wave is already halted here by the handler's own STATUS.HALT, so there
  // is no stop to decline -- but waking a debugger that can never be given a
  // record only strands it. resolve_trap_handler() also withholds the debug
  // flag on such a part, so a cooperating handler never gets this far; one that
  // raises STATUS.HALT regardless still would.
  if (!debug_stop_publishable(gpu_id)) {
    fall_back_to_runtime();
    return;
  }

  apply_debug_event_publication_hook_for_testing(proc);

  // A trap interrupt is wave-local: hardware reports it without waiting for
  // every peer in the queue to stop. The debugger's ensuing SUSPEND_QUEUES
  // request publishes the authoritative full-queue CWSR snapshot.
  const bool notified = notify_debug_event(
      proc, queue_id,
      queue_exception_status != 0 ? debugger_status : KFD_EC_MASK(EC_QUEUE_WAVE_TRAP),
      /*retain_on_rejection=*/queue_exception_status != 0,
      /*reserve_runtime_on_rejection=*/queue_exception_status != 0);
  if (!notified)
    fall_back_to_runtime(/*runtime_already_reserved=*/queue_exception_status != 0);
}

bool SimulatedKfd::on_wave_sendmsg(amdgpu::Wavefront &wave, uint32_t message) {
  constexpr uint32_t kMessageIdMask = 0xFu;
  constexpr uint32_t kMessageInterrupt = 1;
  constexpr uint32_t kMessageGetDoorbell = 10;
  constexpr uint32_t kDoorbellIdBits = 10;
  constexpr uint32_t kRuntimeQueueExceptionMask = 0x3Fu;
  const uint32_t message_id = message & kMessageIdMask;

  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;

  if (message_id == kMessageGetDoorbell) {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto doorbell = proc->queue_doorbell_map_.find(wave.queue_id());
    if (doorbell == proc->queue_doorbell_map_.end())
      return false;
    const uint32_t doorbell_id = (doorbell->second.doorbell_offset / sizeof(uint64_t)) & 0x3FFu;
    // The handler sets EXEC_LO[31] before issuing MSG_GET_DOORBELL and polls
    // until hardware clears it. Return the 10-bit ID with the pending bit clear.
    wave.set_exec((wave.exec() & 0xFFFFFFFF00000000ULL) | doorbell_id);
    return true;
  }

  if (message_id != kMessageInterrupt || !wave.in_trap_handler())
    return false;

  const auto arch = wave.cu().arch();
  const auto interrupt_abi = kmd::detail::trap_interrupt_abi(arch);
  bool profiling_interrupt = false;
  if (interrupt_abi == kmd::detail::TrapInterruptAbi::PreGfx12) {
    // ROCr's pre-GFX12 handler has two MSG_INTERRUPT sites. At the profiling
    // site it loads the event id from TTMP7; at the queue-exception site it
    // loads the packed exception from TTMP3. Both insert an s_nop immediately
    // before the message. Classify the actual control-flow site instead of
    // inspecting the TTMP values: a signal pointer can equal saved M0, and a
    // concurrent trap can legitimately rewrite those registers before the
    // second interrupt.
    if (wave.pc >= 2 * sizeof(uint32_t)) {
      const uint32_t payload_move =
          wave.cu().fetch_instruction_word(wave.pc - 2 * sizeof(uint32_t), wave.process_id());
      const uint32_t delay =
          wave.cu().fetch_instruction_word(wave.pc - sizeof(uint32_t), wave.process_id());
      const auto site =
          kmd::detail::classify_pre_gfx12_trap_interrupt_site(arch, payload_move, delay);
      if (site == kmd::detail::TrapInterruptSite::Profiling)
        profiling_interrupt = true;
      else if (site != kmd::detail::TrapInterruptSite::QueueException)
        return false;
    } else {
      return false;
    }
  } else if (interrupt_abi == kmd::detail::TrapInterruptAbi::Gfx12) {
    // GFX12 exposes the host/performance causes in common TRAPSTS bits 22/26.
    profiling_interrupt = (wave.trapsts() & ((1u << 22) | (1u << 26))) != 0;
  } else
    return false;
  // Profiling completion uses MSG_INTERRUPT too, but puts an event id in M0.
  // Do not reinterpret that id as the queue-exception layout below.
  if (profiling_interrupt)
    return true;

  // The ROCr trap-handler ABI packs the 10-bit doorbell id below six KFD queue
  // exception bits in M0. A subscribed debugger owns the complete event and its
  // CWSR publication is deferred until s_rfe applies the handler's STATUS.HALT;
  // otherwise ROCr owns it. The CU defers the CP call until its wave-state lock
  // is released.
  const uint64_t exception_status = (wave.m0() >> kDoorbellIdBits) & kRuntimeQueueExceptionMask;
  if (exception_status != 0) {
    wave.add_trap_queue_exception_status(exception_status);
    const uint64_t unreported = exception_status & ~wave.trap_runtime_exception_status();
    if (unreported != 0) {
      const uint64_t debugger_status = debugger_queue_exception_mask(
          proc, wave.queue_id(), unreported, /*reserve_runtime_if_unclaimed=*/true);
      if (debugger_status == 0)
        defer_wave_exception_to_runtime(wave, unreported, /*suspend_while_pending=*/false,
                                        /*clear_debug_stop_on_success=*/false,
                                        /*runtime_already_reserved=*/true);
    }
  }

  return true;
}

bool SimulatedKfd::defer_wave_exception_to_runtime(
    amdgpu::Wavefront &wave, uint64_t exception_mask, bool suspend_while_pending,
    bool clear_debug_stop_on_success, bool runtime_already_reserved,
    bool runtime_failure_debuggable, amdgpu::QueueExceptionDeliveryPolicy delivery_policy,
    bool retain_wave) {
  // An unpublished stop cannot be claimed by a debugger after runtime
  // delivery fails. ROCr acknowledges only its first fatal report per queue,
  // so all wave-stop rejection paths transfer ownership on publication.
  if (!runtime_failure_debuggable &&
      delivery_policy == amdgpu::QueueExceptionDeliveryPolicy::WaitForAcknowledgment)
    delivery_policy = amdgpu::QueueExceptionDeliveryPolicy::PublishOrFail;
  auto proc = find_process(wave.process_id());
  if (proc && !runtime_already_reserved)
    reserve_runtime_queue_exception(proc, wave.queue_id(), exception_mask);
  wave.set_fatal_exception_pending(true);
  if (suspend_while_pending)
    wave.set_debug_suspended(true);
  const bool queued = wave.cu().signal_queue_exception(
      wave.queue_id(), wave.process_id(), exception_mask, clear_debug_stop_on_success,
      runtime_failure_debuggable, delivery_policy, retain_wave);
  if (!queued && proc) {
    if (runtime_exception_result_hook_for_testing_)
      runtime_exception_result_hook_for_testing_(false);
    complete_runtime_queue_exception(wave.process_id(), wave.queue_id(), exception_mask,
                                     /*delivered=*/false, runtime_failure_debuggable);
  }
  if (!queued && delivery_policy == amdgpu::QueueExceptionDeliveryPolicy::PublishOrFail) {
    if (auto *sim_engine = wave.cu().engine())
      sim_engine->request_exit("Failed to queue an execution-failure notification", /*code=*/1);
  }
  // The frontend owns the notification even if it cannot be queued. Retained
  // waves stay held until delivery; callers passing retain_wave=false abort the
  // dispatch and must not leave a borrowed wave pointer in the notification.
  return true;
}

bool SimulatedKfd::on_wave_single_step_complete(amdgpu::Wavefront &wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  uint64_t ctx_base = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  if (!debug_stop_publishable(gpu_id)) {
    // Clear single-step even while declining. The compute unit discards this
    // callback's result and re-enters it for as long as the flag is set, so
    // returning false without clearing it spins on every instruction.
    wave.set_debug_single_step(false);
    return false;
  }
  amdgpu::Wavefront::DebugStopState saved = wave.debug_stop_state();
  // The completed step is consumed even when publication loses its debugger
  // claim. Restoring single_step would make ComputeUnitCore call us after every
  // subsequent instruction because it intentionally ignores our return value.
  saved.single_step = false;
  wave.set_debug_single_step(false);
  wave.debug_trap(0);
  // gfx9.4 reports completed single-step through TRAPSTS.TRAP_AFTER_INST. This
  // is the public stop-reason bit rocm-dbgapi consumes before the next resume.
  constexpr uint32_t kTrapAfterInstMask = 1u << 25;
  wave.set_trapsts(wave.trapsts() | kTrapAfterInstMask);
  // Single-step completion is wave-specific. Report it even if another wave
  // in the queue is running; rocm-dbgapi will suspend the queue before reading
  // its CWSR state. Deferring until every peer stops can lose the event forever
  // when a peer runs to normal completion without entering a debug callback.
  // Like hardware's trap interrupt, notification does not itself save the
  // queue. The debugger's ensuing SUSPEND_QUEUES request publishes one stable,
  // authoritative CWSR snapshot instead of redundantly serializing every
  // resident wave here first.
  apply_debug_event_publication_hook_for_testing(proc);
  if (!notify_debug_event(proc, wave.queue_id(), KFD_EC_MASK(EC_QUEUE_WAVE_TRAP),
                          /*retain_on_rejection=*/false)) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
  return true;
}

bool SimulatedKfd::on_wave_watchpoint(amdgpu::Wavefront &wave, uint64_t address, uint32_t bytes,
                                      bool is_write, bool is_atomic) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  uint32_t matched_slots = 0;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    if (session == debug_sessions_.end() || !session->second.enabled)
      return false;
    // The watch modes are not disjoint, so an access can satisfy more than one
    // of them: NONREAD is documented as "write or atomic operations only"
    // (kfd_ioctl.h, and rocdbgapi's os_driver.h maps the STORE_AND_RMW
    // watchpoint kind a plain `(gdb) watch` requests onto it), and ALL matches
    // everything. Hand the set of modes this access triggers to the session
    // rather than a single mode, so a write watch still fires on an atomic
    // read-modify-write. KfdProcess deliberately does not include kfd_ioctl.h,
    // so the mode semantics stay here, on the side of the boundary that owns
    // the KFD ABI.
    const uint32_t matching_modes =
        (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL) |
        (is_atomic ? (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC) |
                         (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD)
         : is_write ? (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD)
                    : (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ));
    matched_slots = session->second.matching_address_watch_slots(address, bytes, matching_modes);
  }
  if (matched_slots == 0)
    return false;

  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  if (!debug_stop_publishable(gpu_id))
    return false;

  static constexpr uint32_t kTrapstsBits[] = {1u << 7, 1u << 12, 1u << 13, 1u << 14};
  const auto saved = wave.debug_stop_state();
  uint32_t trapsts = wave.trapsts();
  for (uint32_t slot = 0; slot < KfdProcess::DebugSession::kMaxAddressWatches; ++slot)
    if ((matched_slots & (uint32_t{1} << slot)) != 0)
      trapsts |= kTrapstsBits[slot];
  wave.set_trapsts(trapsts);
  if (wave.uses_separate_trap_ctrl()) {
    constexpr uint32_t kTrapCtrlAddrWatch = 1u << 7;
    wave.set_gfx12_trap_ctrl_raw(wave.gfx12_trap_ctrl_raw() | kTrapCtrlAddrWatch);
  } else {
    constexpr uint32_t kModeExcpEnAddrWatch = 1u << 19;
    wave.set_mode_raw(wave.mode_raw() | kModeExcpEnAddrWatch);
  }
  wave.debug_trap(0);
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size,
                           KFD_EC_MASK(EC_QUEUE_WAVE_TRAP),
                           /*retain_on_rejection=*/false)) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
  return true;
}

bool SimulatedKfd::on_wave_illegal_instruction(amdgpu::Wavefront &wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  constexpr uint64_t kException = KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION);
  bool debugger_enabled = false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    debugger_enabled = session != debug_sessions_.end() && session->second.enabled;
  }
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  bool has_runtime_signal = false;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
    has_runtime_signal = queue->second.has_runtime_exception_signal;
  }
  if (!debugger_enabled || ctx_base == 0 || !debug_stop_publishable(gpu_id)) {
    if (!has_runtime_signal)
      return false;
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/false,
                                           /*runtime_failure_debuggable=*/false,
                                           amdgpu::QueueExceptionDeliveryPolicy::PublishOrFail);
  }
  constexpr uint32_t kTrapstsIllegalInst = 1u << 11;
  const auto saved = wave.debug_stop_state();
  wave.set_trapsts(wave.trapsts() | kTrapstsIllegalInst);
  wave.set_fatal_exception_pending(true);
  wave.debug_trap(0);
  bool runtime_failure_debuggable = false;
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size, kException,
                           /*retain_on_rejection=*/true, &runtime_failure_debuggable)) {
    wave.restore_debug_stop_state(saved);
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/true,
                                           runtime_failure_debuggable);
  }
  return true;
}

bool SimulatedKfd::on_wave_memory_violation(amdgpu::Wavefront &wave, uint64_t, bool,
                                            uint64_t issue_pc) {
  constexpr uint32_t kTrapstsXnackError = 1u << 28;
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  bool debugger_attached;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    debugger_attached = session != debug_sessions_.end() && session->second.enabled;
  }
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  bool has_runtime_signal = false;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
    has_runtime_signal = queue->second.has_runtime_exception_signal;
  }
  constexpr uint64_t kException = KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION);
  if (!debugger_attached || ctx_base == 0 || !debug_stop_publishable(gpu_id)) {
    if (!has_runtime_signal)
      return false;
    if (proc->xnack_enabled_.load(std::memory_order_relaxed))
      wave.pc = issue_pc;
    wave.set_trapsts(wave.trapsts() | kTrapstsXnackError);
    // ROCr removes its async handler after acknowledging the first fault on
    // a queue. Later waves transfer ownership on publication, without waiting
    // for another acknowledgment or retaining a stop for an absent debugger.
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/false,
                                           /*runtime_failure_debuggable=*/false,
                                           amdgpu::QueueExceptionDeliveryPolicy::PublishOrFail);
  }
  const auto saved = wave.debug_stop_state();
  if (proc->xnack_enabled_.load(std::memory_order_relaxed))
    wave.pc = issue_pc;
  wave.set_trapsts(wave.trapsts() | kTrapstsXnackError);
  wave.debug_trap(0);
  bool runtime_failure_debuggable = false;
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size, kException,
                           /*retain_on_rejection=*/true, &runtime_failure_debuggable)) {
    wave.restore_debug_stop_state(saved);
    // Roll back the debugger stop without losing the runtime-owned fault's reason.
    if (has_runtime_signal)
      wave.set_trapsts(wave.trapsts() | kTrapstsXnackError);
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/true,
                                           runtime_failure_debuggable);
  }
  return true;
}

bool SimulatedKfd::on_wave_alu_exception(amdgpu::Wavefront &wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    if (session == debug_sessions_.end() || !session->second.enabled)
      return false;
  }
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  if (!debug_stop_publishable(gpu_id))
    return false;
  const auto saved = wave.debug_stop_state();
  wave.set_fatal_exception_pending(true);
  wave.debug_trap(0);
  constexpr uint64_t kException = KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR);
  bool runtime_failure_debuggable = false;
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size, kException,
                           /*retain_on_rejection=*/true, &runtime_failure_debuggable)) {
    wave.restore_debug_stop_state(saved);
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/true,
                                           runtime_failure_debuggable);
  }
  return true;
}

bool SimulatedKfd::on_wave_execution_failure(amdgpu::Wavefront &wave, bool retain_wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  {
    std::lock_guard<std::mutex> lock(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end() || !queue->second.has_runtime_exception_signal)
      return false;
  }
  // A simulator capability failure or GWS timeout has no architectural wave-stop reason.
  // dbgapi cannot surface or forward a queue abort without such a wave stop;
  // giving it ownership would strand the queue. Deliver directly to ROCr even
  // with a debugger attached, preserving TRAPSTS and the debugger stop state.
  // ROCr acknowledges only once per queue and may run an error callback or
  // core dump first. Successful publication transfers ownership without waiting.
  return defer_wave_exception_to_runtime(
      wave, KFD_EC_MASK(EC_QUEUE_WAVE_ABORT),
      /*suspend_while_pending=*/retain_wave,
      /*clear_debug_stop_on_success=*/false,
      /*runtime_already_reserved=*/false,
      /*runtime_failure_debuggable=*/false,
      retain_wave ? amdgpu::QueueExceptionDeliveryPolicy::PublishOrFail
                  : amdgpu::QueueExceptionDeliveryPolicy::PublishAndCancel,
      retain_wave);
}

} // namespace rocjitsu
