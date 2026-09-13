// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/simdojo_device_backend.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/soc.h"

#include <cerrno>

namespace rocjitsu::kmd {

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

void SimdojoDeviceBackend::update_queue(uint32_t process, uint32_t queue, uint64_t ring,
                                        uint32_t size, uint32_t percentage) {
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->update_queue(queue, process, ring, size, percentage);
  });
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
