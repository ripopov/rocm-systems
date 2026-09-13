// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/simdojo_device_backend.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"

#include <cerrno>

namespace rocjitsu::kmd {

void SimdojoDeviceBackend::set_active(bool active) {
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    for (auto *cu : cp->compute_units())
      cu->set_debug_active(active);
  });
}

void SimdojoDeviceBackend::release_queue(uint32_t process, uint32_t queue) {
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_queue_debug_suspended(queue, process, false);
    for (auto *cu : cp->compute_units()) {
      bool wake = false;
      cu->with_wave_state_locked([&] {
        for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
          auto *wave = cu->wf(slot);
          if (wave->is_halted() || wave->process_id() != process || wave->queue_id() != queue ||
              wave->fatal_exception_pending())
            continue;
          wave->set_debug_single_step(false);
          wave->set_debug_halted(false);
          wave->set_debug_suspended(false);
          // s_sendmsghalt also raises architectural STATUS.HALT. Clear it so
          // returning from the trap cannot re-halt after the debugger detaches.
          wave->set_status_halt(false);
          wave->set_self_halted(false);
          wake = true;
        }
      });
      if (wake)
        cu->schedule_work_async();
    }
  });
}

bool SimdojoDeviceBackend::signal_queue_exception(uint32_t process, uint32_t queue, uint64_t mask) {
  bool delivered = false;
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    delivered |= cp->signal_queue_exception(queue, process, mask);
  });
  return delivered;
}

rj_code_arch_t SimdojoDeviceBackend::architecture() const { return soc_.arch(); }

void SimdojoDeviceBackend::configure(const DeviceApertures &apertures,
                                     uint32_t scratch_wave_divisor,
                                     const DeviceCallbacks &callbacks) {
  const uint32_t xcc_count = soc_.num_xcds();
  for (uint32_t xcc = 0; xcc < xcc_count; ++xcc) {
    auto *cp = soc_.xcd(xcc)->command_processor();
    if (!cp)
      continue;
    cp->set_scratch_wave_divisor(scratch_wave_divisor);
    cp->set_scratch_xcc_layout(xcc, xcc_count);
  }
  soc_.set_apertures(apertures.lds_base, apertures.lds_limit, apertures.scratch_base,
                     apertures.scratch_limit);
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_interrupt_callback(callbacks.interrupt);
    cp->set_scratch_backing_resolver(callbacks.scratch_backing);
    cp->set_scratch_backing_allocator(callbacks.allocate_scratch);
    for (auto *cu : cp->compute_units()) {
      cu->set_trap_handler_resolver(
          [resolve = callbacks.trap_handler](const amdgpu::Wavefront &wave)
              -> std::optional<amdgpu::ComputeUnitCore::TrapHandlerConfig> {
            if (!resolve)
              return std::nullopt;
            const auto handler = resolve(wave.process_id());
            if (!handler)
              return std::nullopt;
            return amdgpu::ComputeUnitCore::TrapHandlerConfig{handler->tba, handler->tma,
                                                              handler->debug_enabled};
          });
      cu->set_trap_completion_handler(
          [complete = callbacks.trap_complete](amdgpu::Wavefront &wave) {
            if (complete)
              complete(wave.process_id(), wave.queue_id());
          });
      cu->set_sendmsg_handler([doorbell = callbacks.doorbell_id, exists = callbacks.process_exists](
                                  amdgpu::Wavefront &wave, uint32_t message) {
        constexpr uint32_t kMessageIdMask = 0xFu;
        constexpr uint32_t kMessageInterrupt = 1;
        constexpr uint32_t kMessageGetDoorbell = 10;
        if (!exists || !exists(wave.process_id()))
          return false;
        const uint32_t message_id = message & kMessageIdMask;
        if (message_id == kMessageGetDoorbell) {
          const auto id = doorbell ? doorbell(wave.process_id(), wave.queue_id()) : std::nullopt;
          if (!id)
            return false;
          // Clear EXEC_LO[31], which the handler polls while waiting for
          // MSG_GET_DOORBELL, and preserve EXEC_HI.
          wave.set_exec((wave.exec() & 0xFFFFFFFF00000000ULL) | *id);
          return true;
        }
        // CWSR publication waits for s_rfe to apply STATUS.HALT.
        return message_id == kMessageInterrupt && wave.in_trap_handler();
      });
    }
  });
  configure_debug_callbacks(callbacks);
}

void SimdojoDeviceBackend::set_memory_fault_reporter(MemoryFaultReporter *reporter) {
  if (auto *memory = soc_.memory())
    memory->set_memory_fault_reporter(reporter);
}

int SimdojoDeviceBackend::register_process(KfdProcess &process, bool passthrough) {
  if (auto *memory = soc_.memory()) {
    memory->register_process(process.process_id(), &process.page_table_, &process.page_table_mutex_,
                             process.page_table_generation(), process.page_table_request_mutex());
    if (passthrough)
      memory->set_passthrough(true);
  }
  return 0;
}

void SimdojoDeviceBackend::unregister_process(uint32_t process) {
  if (auto *memory = soc_.memory())
    memory->unregister_process(process);
}

void SimdojoDeviceBackend::set_process_client_pid(uint32_t process, int32_t client_pid) {
  if (auto *memory = soc_.memory())
    memory->set_process_client_pid(process, client_pid);
}

void SimdojoDeviceBackend::set_process_mem_fd(uint32_t process, int fd) {
  if (auto *memory = soc_.memory())
    memory->set_process_mem_fd(process, fd);
}

std::optional<uint32_t> SimdojoDeviceBackend::queue_xcc(uint32_t ordinal) const {
  if (!soc_.assign_queue_owner_cp(ordinal))
    return std::nullopt;
  return soc_.arch() == ROCJITSU_CODE_ARCH_CDNA5 ? soc_.queue_xcd_id(ordinal) : 0;
}

int SimdojoDeviceBackend::create_queue(const QueueDescriptor &queue) {
  auto *cp = soc_.assign_queue_owner_cp(queue.ordinal);
  if (!cp)
    return -EINVAL;
  amdgpu::HwQueue hw{};
  hw.process_id = queue.process_id;
  hw.queue_id = queue.queue_id;
  hw.ring_base_va = queue.ring_base_va;
  hw.ring_size = queue.ring_size;
  hw.read_ptr_va = queue.read_ptr_va;
  hw.write_ptr_va = queue.write_ptr_va;
  hw.doorbell_offset = queue.doorbell_offset;
  hw.doorbell_base = queue.doorbell_base;
  hw.last_doorbell = ~uint64_t(0);
  hw.host_accessible = queue.host_accessible;
  hw.is_sdma = queue.is_sdma;
  hw.xcd_fanout = queue.xcd_fanout;
  hw.queue_desc_va = queue.queue_desc_va;
  hw.exception_status_va = queue.exception_status_va;
  hw.exception_event_id = queue.exception_event_id;
  cp->register_queue(std::move(hw));
  return 0;
}

int SimdojoDeviceBackend::update_queue(uint32_t process, uint32_t queue, uint64_t ring,
                                        uint32_t size, uint32_t percentage) {
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->update_queue(queue, process, ring, size, percentage);
  });
  return 0;
}

void SimdojoDeviceBackend::destroy_queue(uint32_t process, uint32_t queue) {
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) { cp->unregister_queue(queue, process); });
}

void SimdojoDeviceBackend::set_doorbell_base(uint32_t process, void *base) {
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) { cp->set_doorbell_base(process, base); });
}

uint64_t SimdojoDeviceBackend::read64(uint32_t process, uint64_t address) {
  if (auto *cp = soc_.assign_queue_owner_cp(0))
    return cp->read_process_memory64(address, process);
  return 0;
}

void SimdojoDeviceBackend::write64(uint32_t process, uint64_t address, uint64_t value) {
  if (auto *memory = soc_.memory())
    memory->write64(address, value, process);
}

} // namespace rocjitsu::kmd
