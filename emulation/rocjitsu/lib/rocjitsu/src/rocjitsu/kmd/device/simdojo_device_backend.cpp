// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/simdojo_device_backend.h"

#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/soc.h"

#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unordered_map>
#include <unistd.h>
#include <utility>

namespace rocjitsu::kmd {

class SimdojoDeviceBackend::Impl {
public:
  struct ProcessMemory {
    KfdProcess::PageTable page_table;
    std::shared_mutex page_table_mutex;
  };

  Impl(SoC &soc, uint32_t gpu_id)
      : soc(soc), info{.gpu_id = gpu_id, .architecture = soc.arch()} {}

  std::shared_ptr<ProcessMemory> findProcess(ProcessId process_id) {
    std::lock_guard lock(process_mutex);
    auto it = processes.find(process_id);
    return it == processes.end() ? nullptr : it->second;
  }

  SoC &soc;
  DeviceInfo info;
  std::atomic<AllocationId> next_allocation_id{1};
  std::mutex process_mutex;
  std::unordered_map<ProcessId, std::shared_ptr<ProcessMemory>> processes;
};

SimdojoDeviceBackend::SimdojoDeviceBackend(SoC &soc, uint32_t gpu_id)
    : impl_(std::make_unique<Impl>(soc, gpu_id)) {}

SimdojoDeviceBackend::~SimdojoDeviceBackend() = default;

const DeviceInfo &SimdojoDeviceBackend::deviceInfo() const {
  return impl_->info;
}

int SimdojoDeviceBackend::openProcess(
    ProcessId process_id, int32_t client_pid,
    bool allow_host_pointer_passthrough) {
  auto context = std::make_shared<Impl::ProcessMemory>();
  {
    std::lock_guard lock(impl_->process_mutex);
    impl_->processes[process_id] = context;
  }

  if (auto *memory = impl_->soc.memory()) {
    memory->register_process(process_id, &context->page_table,
                             &context->page_table_mutex);
    memory->set_passthrough(allow_host_pointer_passthrough);
    if (client_pid > 0)
      memory->set_process_client_pid(process_id, client_pid);
  }
  return 0;
}

void SimdojoDeviceBackend::setProcessClientPid(ProcessId process_id,
                                                int32_t client_pid) {
  if (auto *memory = impl_->soc.memory())
    memory->set_process_client_pid(process_id, client_pid);
}

void SimdojoDeviceBackend::closeProcess(ProcessId process_id) {
  if (auto *memory = impl_->soc.memory())
    memory->unregister_process(process_id);
  std::lock_guard lock(impl_->process_mutex);
  impl_->processes.erase(process_id);
}

int SimdojoDeviceBackend::allocate(const AllocationRequest &request,
                                   BackingSlice &slice) {
  constexpr uint64_t page_size = KfdProcess::kPageSize;
  constexpr uint64_t page_mask = page_size - 1;
  const uint64_t page_offset = request.gpu_va & page_mask;
  if (request.size > std::numeric_limits<uint64_t>::max() - page_offset)
    return -EOVERFLOW;
  const uint64_t logical_extent = page_offset + request.size;
  if (logical_extent > std::numeric_limits<uint64_t>::max() - page_mask)
    return -EOVERFLOW;
  const uint64_t backing_size = (logical_extent + page_mask) & ~page_mask;

  slice = {
      .allocation_id =
          impl_->next_allocation_id.fetch_add(1, std::memory_order_relaxed),
      // Simdojo's physical identity is always the process-local host mapping,
      // which SimulatedDriver supplies when it installs the GPU mapping.
      .device_paddr = 0,
      .size = request.shared_backing_required ? backing_size : request.size,
      .fd = -1,
      .fd_offset = 0,
  };
  if (!request.shared_backing_required)
    return 0;

  int raw_fd = static_cast<int>(
      syscall(SYS_memfd_create, "rocjitsu_alloc",
              MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (raw_fd < 0)
    return -errno;

  int fd = fcntl(raw_fd, F_DUPFD_CLOEXEC, 4096);
  if (fd < 0) {
    fd = raw_fd;
  } else {
    syscall(SYS_close, raw_fd);
  }

  if (backing_size > static_cast<uint64_t>(
                         std::numeric_limits<off_t>::max()) ||
      ftruncate(fd, static_cast<off_t>(backing_size)) != 0) {
    int saved_errno = errno;
    syscall(SYS_close, fd);
    return backing_size >
                   static_cast<uint64_t>(std::numeric_limits<off_t>::max())
               ? -EOVERFLOW
               : -saved_errno;
  }
  (void)fallocate(fd, 0, 0, static_cast<off_t>(backing_size));
  (void)fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK);
  slice.fd = fd;
  return 0;
}

void SimdojoDeviceBackend::free([[maybe_unused]] ProcessId process_id,
                                [[maybe_unused]] AllocationId allocation_id) {}

void SimdojoDeviceBackend::setApertures(const Apertures &apertures) {
  impl_->soc.set_apertures(apertures.lds_base, apertures.lds_limit,
                           apertures.scratch_base, apertures.scratch_limit);
}

void SimdojoDeviceBackend::setInterruptSink(InterruptSink sink) {
  impl_->soc.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_interrupt_callback(sink);
  });
}

void SimdojoDeviceBackend::setScratchCallbacks(
    ScratchBackingResolver resolver, ScratchBackingAllocator allocator) {
  impl_->soc.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_scratch_backing_resolver(resolver);
    cp->set_scratch_backing_allocator(allocator);
  });
}

int SimdojoDeviceBackend::map(const Mapping &mapping) {
  auto context = impl_->findProcess(mapping.process_id);
  if (!context)
    return -ESRCH;
  constexpr uint64_t page_mask = KfdProcess::kPageSize - 1;
  if ((mapping.gpu_va & page_mask) !=
      (mapping.device_paddr & page_mask))
    return -EINVAL;
  std::unique_lock lock(context->page_table_mutex);
  const uint64_t page_offset = mapping.gpu_va & page_mask;
  auto *base = reinterpret_cast<uint8_t *>(
      mapping.device_paddr - page_offset);
  const uint64_t mapping_extent = page_offset + mapping.size;
  for (uint64_t offset = 0; offset < mapping_extent;
       offset += KfdProcess::kPageSize) {
    context->page_table[((mapping.gpu_va - page_offset) + offset) >>
                        KfdProcess::kPageShift] = {
        base + offset, static_cast<amdgpu::Mtype>(mapping.memory_type)};
  }
  return 0;
}

int SimdojoDeviceBackend::unmap(ProcessId process_id, uint64_t gpu_va,
                                uint64_t size) {
  auto context = impl_->findProcess(process_id);
  if (!context)
    return -ESRCH;
  std::unique_lock lock(context->page_table_mutex);
  const uint64_t page_offset = gpu_va & (KfdProcess::kPageSize - 1);
  for (uint64_t offset = 0; offset < page_offset + size;
       offset += KfdProcess::kPageSize) {
    context->page_table.erase(
        ((gpu_va - page_offset) + offset) >> KfdProcess::kPageShift);
  }
  return 0;
}

void SimdojoDeviceBackend::setMappingMemoryType(
    ProcessId process_id, uint64_t gpu_va, uint64_t size, uint8_t memory_type) {
  auto context = impl_->findProcess(process_id);
  if (!context)
    return;
  std::unique_lock lock(context->page_table_mutex);
  for (uint64_t offset = 0; offset < size; offset += KfdProcess::kPageSize) {
    auto it =
        context->page_table.find((gpu_va + offset) >> KfdProcess::kPageShift);
    if (it != context->page_table.end())
      it->second.mtype = static_cast<amdgpu::Mtype>(memory_type);
  }
}

void SimdojoDeviceBackend::write64(ProcessId process_id, uint64_t gpu_va,
                                    uint64_t value) {
  if (auto *memory = impl_->soc.memory())
    memory->write64(gpu_va, value, process_id);
}

void SimdojoDeviceBackend::setDoorbellBase(ProcessId process_id,
                                            uintptr_t address) {
  impl_->soc.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_doorbell_base(process_id, reinterpret_cast<void *>(address));
  });
}

bool SimdojoDeviceBackend::createQueue(const QueueDescriptor &queue) {
  auto *target = impl_->soc.assign_queue_cp();
  if (!target)
    return false;

  amdgpu::HwQueue hardware_queue{};
  hardware_queue.process_id = queue.process_id;
  hardware_queue.queue_id = queue.queue_id;
  hardware_queue.ring_base_va = queue.ring_base_va;
  hardware_queue.ring_size = queue.ring_size;
  hardware_queue.read_ptr_va = queue.read_pointer_va;
  hardware_queue.write_ptr_va = queue.write_pointer_va;
  hardware_queue.doorbell_offset = queue.doorbell_offset;
  hardware_queue.doorbell_base =
      reinterpret_cast<void *>(queue.doorbell_address);
  hardware_queue.last_doorbell = ~uint64_t(0);
  hardware_queue.host_accessible = queue.host_accessible;
  hardware_queue.is_sdma = queue.is_sdma;
  hardware_queue.queue_desc_va = queue.queue_descriptor_va;
  target->register_queue(std::move(hardware_queue));
  return true;
}

void SimdojoDeviceBackend::updateQueue(ProcessId process_id, QueueId queue_id,
                                        uint64_t ring_base_va,
                                        uint32_t ring_size) {
  impl_->soc.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->update_queue(queue_id, process_id, ring_base_va, ring_size);
  });
}

void SimdojoDeviceBackend::destroyQueue(ProcessId process_id,
                                         QueueId queue_id) {
  impl_->soc.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->unregister_queue(queue_id, process_id);
  });
}

std::shared_ptr<DeviceBackend> makeSimdojoDeviceBackend(SoC &soc,
                                                        uint32_t gpu_id) {
  return std::make_shared<SimdojoDeviceBackend>(soc, gpu_id);
}

} // namespace rocjitsu::kmd
