// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/kmd/device/simdojo_device_backend.h"
#include "rocjitsu/kmd/device/trap_interrupt.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "uapi/linux/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <iterator>

namespace rocjitsu::kmd {

void SimdojoDeviceBackend::on_wave_trap_complete(amdgpu::Wavefront &wave) {
  const uint32_t process_id = wave.process_id();
  const uint32_t queue_id = wave.queue_id();
  const auto context = events_->event_context(process_id, queue_id);
  if (!context)
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
    debugger_status = events_->claim_debug_exception(process_id, queue_id, queue_exception_status,
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

  if (!context->debug_stop_publishable) {
    fall_back_to_runtime();
    return;
  }

  // A trap interrupt is wave-local: hardware reports it without waiting for
  // every peer in the queue to stop. The debugger's ensuing SUSPEND_QUEUES
  // request publishes the authoritative full-queue CWSR snapshot.
  const bool notified =
      events_
          ->publish_debug_stop(
              {*context,
               queue_exception_status != 0 ? debugger_status : KFD_EC_MASK(EC_QUEUE_WAVE_TRAP),
               /*save_queue=*/false, /*retain_on_rejection=*/queue_exception_status != 0})
          .delivered;
  if (!notified)
    fall_back_to_runtime(/*runtime_already_reserved=*/queue_exception_status != 0);
}

bool SimdojoDeviceBackend::on_wave_sendmsg(amdgpu::Wavefront &wave, uint32_t message) {
  constexpr uint32_t kMessageIdMask = 0xFu;
  constexpr uint32_t kMessageInterrupt = 1;
  constexpr uint32_t kMessageGetDoorbell = 10;
  constexpr uint32_t kDoorbellIdBits = 10;
  constexpr uint32_t kRuntimeQueueExceptionMask = 0x3Fu;
  const uint32_t message_id = message & kMessageIdMask;

  const auto context = events_->event_context(wave.process_id(), wave.queue_id());
  if (!context)
    return false;

  if (message_id == kMessageGetDoorbell) {
    if (!context->doorbell_id)
      return false;
    const uint32_t doorbell_id = *context->doorbell_id;
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
      const uint64_t debugger_status = events_->claim_debug_exception(
          wave.process_id(), wave.queue_id(), unreported, /*reserve_runtime_if_unclaimed=*/true);
      if (debugger_status == 0)
        defer_wave_exception_to_runtime(wave, unreported, /*suspend_while_pending=*/false,
                                        /*clear_debug_stop_on_success=*/false,
                                        /*runtime_already_reserved=*/true);
    }
  }

  return true;
}

bool SimdojoDeviceBackend::defer_wave_exception_to_runtime(
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
  const auto context = events_->event_context(wave.process_id(), wave.queue_id());
  if (context && !runtime_already_reserved)
    events_->reserve_runtime_exception(wave.process_id(), wave.queue_id(), exception_mask);
  wave.set_fatal_exception_pending(true);
  if (suspend_while_pending)
    wave.set_debug_suspended(true);
  const bool queued = wave.cu().signal_queue_exception(
      wave.queue_id(), wave.process_id(), exception_mask, clear_debug_stop_on_success,
      runtime_failure_debuggable, delivery_policy, retain_wave);
  if (!queued && context)
    events_->finish_runtime_exception(wave.process_id(), wave.queue_id(), exception_mask,
                                      /*delivered=*/false, runtime_failure_debuggable);
  if (!queued && delivery_policy == amdgpu::QueueExceptionDeliveryPolicy::PublishOrFail) {
    if (auto *sim_engine = wave.cu().engine())
      sim_engine->request_exit("Failed to queue an execution-failure notification", /*code=*/1);
  }
  // The frontend owns the notification even if it cannot be queued. Retained
  // waves stay held until delivery; callers passing retain_wave=false abort the
  // dispatch and must not leave a borrowed wave pointer in the notification.
  return true;
}

bool SimdojoDeviceBackend::on_wave_single_step_complete(amdgpu::Wavefront &wave) {
  const auto context = events_->event_context(wave.process_id(), wave.queue_id());
  if (!context)
    return false;
  if (context->context_address == 0)
    return false;
  if (!context->debug_stop_publishable) {
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
  if (!events_
           ->publish_debug_stop({*context, KFD_EC_MASK(EC_QUEUE_WAVE_TRAP),
                                 /*save_queue=*/false, /*retain_on_rejection=*/false})
           .delivered) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
  return true;
}

bool SimdojoDeviceBackend::on_wave_watchpoint(amdgpu::Wavefront &wave, uint64_t address,
                                              uint32_t bytes, bool is_write, bool is_atomic) {
  const uint32_t matched_slots =
      events_->matching_watchpoints(wave.process_id(), address, bytes, is_write, is_atomic);
  if (matched_slots == 0)
    return false;
  const auto context = events_->event_context(wave.process_id(), wave.queue_id());
  if (!context)
    return false;

  if (!context->debug_stop_publishable)
    return false;

  static constexpr uint32_t kTrapstsBits[] = {1u << 7, 1u << 12, 1u << 13, 1u << 14};
  const auto saved = wave.debug_stop_state();
  uint32_t trapsts = wave.trapsts();
  for (uint32_t slot = 0; slot < std::size(kTrapstsBits); ++slot)
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
  if (!events_
           ->publish_debug_stop({*context, KFD_EC_MASK(EC_QUEUE_WAVE_TRAP),
                                 /*save_queue=*/true, /*retain_on_rejection=*/false})
           .delivered) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
  return true;
}

bool SimdojoDeviceBackend::on_wave_illegal_instruction(amdgpu::Wavefront &wave) {
  const auto context = events_->event_context(wave.process_id(), wave.queue_id());
  if (!context)
    return false;
  constexpr uint64_t kException = KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION);
  if (!context->debugger_attached || !context->debug_stop_publishable) {
    if (!context->has_runtime_signal)
      return false;
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/false,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/false,
                                           /*runtime_failure_debuggable=*/false,
                                           amdgpu::QueueExceptionDeliveryPolicy::PublishAndCancel,
                                           /*retain_wave=*/false);
  }
  constexpr uint32_t kTrapstsIllegalInst = 1u << 11;
  const auto saved = wave.debug_stop_state();
  wave.set_trapsts(wave.trapsts() | kTrapstsIllegalInst);
  wave.set_fatal_exception_pending(true);
  wave.debug_trap(0);
  const DeviceDebugEventResult publication =
      events_->publish_debug_stop({*context, kException,
                                   /*save_queue=*/true, /*retain_on_rejection=*/true});
  const bool runtime_failure_debuggable = publication.runtime_failure_debuggable;
  if (!publication.delivered) {
    wave.restore_debug_stop_state(saved);
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/false,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/true,
                                           runtime_failure_debuggable,
                                           amdgpu::QueueExceptionDeliveryPolicy::PublishAndCancel,
                                           /*retain_wave=*/false);
  }
  return true;
}

bool SimdojoDeviceBackend::on_wave_memory_violation(amdgpu::Wavefront &wave, uint64_t, bool,
                                                    uint64_t issue_pc) {
  constexpr uint32_t kTrapstsXnackError = 1u << 28;
  const auto context = events_->event_context(wave.process_id(), wave.queue_id());
  if (!context)
    return false;
  constexpr uint64_t kException = KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION);
  if (!context->debugger_attached || !context->debug_stop_publishable) {
    if (!context->has_runtime_signal)
      return false;
    if (context->xnack_enabled)
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
  if (context->xnack_enabled)
    wave.pc = issue_pc;
  wave.set_trapsts(wave.trapsts() | kTrapstsXnackError);
  wave.debug_trap(0);
  const DeviceDebugEventResult publication =
      events_->publish_debug_stop({*context, kException,
                                   /*save_queue=*/true, /*retain_on_rejection=*/true});
  const bool runtime_failure_debuggable = publication.runtime_failure_debuggable;
  if (!publication.delivered) {
    wave.restore_debug_stop_state(saved);
    // Roll back the debugger stop without losing the runtime-owned fault's reason.
    if (context->has_runtime_signal)
      wave.set_trapsts(wave.trapsts() | kTrapstsXnackError);
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/true,
                                           runtime_failure_debuggable);
  }
  return true;
}

bool SimdojoDeviceBackend::on_wave_alu_exception(amdgpu::Wavefront &wave) {
  const auto context = events_->event_context(wave.process_id(), wave.queue_id());
  if (!context)
    return false;
  if (!context->debugger_attached || !context->debug_stop_publishable)
    return false;
  const auto saved = wave.debug_stop_state();
  wave.set_fatal_exception_pending(true);
  wave.debug_trap(0);
  constexpr uint64_t kException = KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR);
  const DeviceDebugEventResult publication =
      events_->publish_debug_stop({*context, kException,
                                   /*save_queue=*/true, /*retain_on_rejection=*/true});
  const bool runtime_failure_debuggable = publication.runtime_failure_debuggable;
  if (!publication.delivered) {
    wave.restore_debug_stop_state(saved);
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/true,
                                           runtime_failure_debuggable);
  }
  return true;
}

bool SimdojoDeviceBackend::on_wave_execution_failure(amdgpu::Wavefront &wave) {
  const auto context = events_->event_context(wave.process_id(), wave.queue_id());
  if (!context)
    return false;
  if (!context->has_runtime_signal)
    return false;
  // A simulator capability failure or GWS timeout has no architectural wave-stop reason.
  // dbgapi cannot surface or forward a queue abort without such a wave stop;
  // giving it ownership would strand the queue. Deliver directly to ROCr even
  // with a debugger attached, preserving TRAPSTS and the debugger stop state.
  // ROCr acknowledges only once per queue and may run an error callback or
  // core dump first. Successful publication transfers ownership without waiting.
  return defer_wave_exception_to_runtime(wave, KFD_EC_MASK(EC_QUEUE_WAVE_ABORT),
                                         /*suspend_while_pending=*/false,
                                         /*clear_debug_stop_on_success=*/false,
                                         /*runtime_already_reserved=*/false,
                                         /*runtime_failure_debuggable=*/false,
                                         amdgpu::QueueExceptionDeliveryPolicy::PublishAndCancel,
                                         /*retain_wave=*/false);
}

} // namespace rocjitsu::kmd
