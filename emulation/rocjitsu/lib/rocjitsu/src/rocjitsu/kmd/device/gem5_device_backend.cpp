// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/gem5_device_backend.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stop_token>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <unistd.h>

namespace rocjitsu::kmd {
namespace {

template <typename T> T abiStruct() {
  T value{};
  value.abi_version = RJ_KMD_SERVER_ABI_VERSION;
  value.struct_size = sizeof(T);
  return value;
}

uint64_t queueKey(ProcessId process_id, QueueId queue_id) {
  return (static_cast<uint64_t>(process_id) << 32) | queue_id;
}

} // namespace

class Gem5DeviceBackend::Impl {
public:
  struct WatchedDoorbell {
    ProcessId process_id = 0;
    QueueId queue_id = 0;
    uint32_t offset = 0;
    const uint64_t *address = nullptr;
    uint64_t last_value = ~uint64_t(0);
  };

  Impl(DeviceInfo device_info, const rj_kmd_backend_ops_t &backend_operations)
      : info(device_info), operations(backend_operations) {
    notification_fd =
        static_cast<int>(syscall(SYS_eventfd2, 0, EFD_CLOEXEC | EFD_NONBLOCK));
    watcher = std::jthread(
        [this](std::stop_token stop) { watchDoorbells(stop); });
  }

  ~Impl() {
    watcher.request_stop();
    if (watcher.joinable())
      watcher.join();
    if (notification_fd >= 0)
      syscall(SYS_close, notification_fd);
    if (operations.shutdown)
      operations.shutdown(operations.context);
  }

  void watchDoorbells(std::stop_token stop) {
    while (!stop.stop_requested()) {
      bool signaled = false;
      {
        std::lock_guard lock(doorbell_mutex);
        for (auto &[key, watched] : watched_doorbells) {
          (void)key;
          if (!watched.address)
            continue;
          uint64_t value =
              __atomic_load_n(watched.address, __ATOMIC_ACQUIRE);
          if (value == watched.last_value)
            continue;
          watched.last_value = value;
          auto record = abiStruct<rj_kmd_doorbell_record_t>();
          record.process_id = watched.process_id;
          record.queue_id = watched.queue_id;
          record.offset = watched.offset;
          record.value = value;
          pending_doorbells.push_back(record);
          signaled = true;
        }
      }
      if (signaled && notification_fd >= 0) {
        uint64_t one = 1;
        (void)syscall(SYS_write, notification_fd, &one, sizeof(one));
      }
      std::this_thread::sleep_for(std::chrono::microseconds(250));
    }
  }

  int processCallback(rj_kmd_process_fn callback, ProcessId process_id,
                      int32_t client_pid) const {
    if (!callback)
      return -ENOTSUP;
    auto request = abiStruct<rj_kmd_process_request_t>();
    request.process_id = process_id;
    request.client_pid = client_pid;
    return callback(operations.context, &request);
  }

  DeviceInfo info;
  rj_kmd_backend_ops_t operations{};
  InterruptSink interrupt_sink;
  int notification_fd = -1;
  std::jthread watcher;
  mutable std::mutex doorbell_mutex;
  std::unordered_map<ProcessId, uintptr_t> doorbell_bases;
  std::unordered_map<uint64_t, WatchedDoorbell> watched_doorbells;
  std::deque<rj_kmd_doorbell_record_t> pending_doorbells;
};

Gem5DeviceBackend::Gem5DeviceBackend(
    DeviceInfo info, const rj_kmd_backend_ops_t &operations)
    : impl_(std::make_unique<Impl>(info, operations)) {}

Gem5DeviceBackend::~Gem5DeviceBackend() = default;

const DeviceInfo &Gem5DeviceBackend::deviceInfo() const {
  return impl_->info;
}

int Gem5DeviceBackend::openProcess(
    ProcessId process_id, int32_t client_pid,
    [[maybe_unused]] bool allow_host_pointer_passthrough) {
  return impl_->processCallback(impl_->operations.open_process, process_id,
                                client_pid);
}

void Gem5DeviceBackend::setProcessClientPid(ProcessId process_id,
                                            int32_t client_pid) {
  (void)impl_->processCallback(impl_->operations.set_process_client_pid,
                               process_id, client_pid);
}

void Gem5DeviceBackend::closeProcess(ProcessId process_id) {
  {
    std::lock_guard lock(impl_->doorbell_mutex);
    impl_->doorbell_bases.erase(process_id);
    std::erase_if(impl_->watched_doorbells, [process_id](const auto &entry) {
      return entry.second.process_id == process_id;
    });
  }
  (void)impl_->processCallback(impl_->operations.close_process, process_id, 0);
}

int Gem5DeviceBackend::allocate(const AllocationRequest &request,
                                BackingSlice &slice) {
  if (!impl_->operations.allocate)
    return -ENOTSUP;
  auto abi_request = abiStruct<rj_kmd_allocation_request_t>();
  abi_request.process_id = request.process_id;
  abi_request.gpu_id = request.gpu_id;
  abi_request.gpu_va = request.gpu_va;
  abi_request.size = request.size;
  abi_request.flags = request.flags;
  auto abi_slice = abiStruct<rj_kmd_backing_slice_t>();
  abi_slice.fd = -1;
  int32_t result =
      impl_->operations.allocate(impl_->operations.context, &abi_request,
                                 &abi_slice);
  if (result != 0)
    return result;
  if (abi_slice.abi_version != RJ_KMD_SERVER_ABI_VERSION ||
      abi_slice.struct_size < sizeof(rj_kmd_backing_slice_t))
    return -EPROTO;
  slice = {
      .allocation_id = abi_slice.allocation_id,
      .device_paddr = abi_slice.device_paddr,
      .size = abi_slice.size,
      .fd = abi_slice.fd,
      .fd_offset = abi_slice.fd_offset,
  };
  return 0;
}

void Gem5DeviceBackend::free(ProcessId process_id,
                             AllocationId allocation_id) {
  if (!impl_->operations.free)
    return;
  auto request = abiStruct<rj_kmd_free_request_t>();
  request.process_id = process_id;
  request.allocation_id = allocation_id;
  (void)impl_->operations.free(impl_->operations.context, &request);
}

void Gem5DeviceBackend::setApertures(const Apertures &apertures) {
  if (!impl_->operations.set_apertures)
    return;
  auto request = abiStruct<rj_kmd_apertures_t>();
  request.lds_base = apertures.lds_base;
  request.lds_limit = apertures.lds_limit;
  request.scratch_base = apertures.scratch_base;
  request.scratch_limit = apertures.scratch_limit;
  (void)impl_->operations.set_apertures(impl_->operations.context, &request);
}

void Gem5DeviceBackend::setInterruptSink(InterruptSink sink) {
  impl_->interrupt_sink = std::move(sink);
}

void Gem5DeviceBackend::setScratchCallbacks(
    [[maybe_unused]] ScratchBackingResolver resolver,
    [[maybe_unused]] ScratchBackingAllocator allocator) {}

int Gem5DeviceBackend::map(const Mapping &mapping) {
  if (!impl_->operations.map)
    return -ENOTSUP;
  auto request = abiStruct<rj_kmd_mapping_t>();
  request.process_id = mapping.process_id;
  request.memory_type = mapping.memory_type;
  request.allocation_id = mapping.allocation_id;
  request.gpu_va = mapping.gpu_va;
  request.device_paddr = mapping.device_paddr;
  request.size = mapping.size;
  return impl_->operations.map(impl_->operations.context, &request);
}

int Gem5DeviceBackend::unmap(ProcessId process_id, uint64_t gpu_va,
                             uint64_t size) {
  if (!impl_->operations.unmap)
    return -ENOTSUP;
  auto request = abiStruct<rj_kmd_unmap_request_t>();
  request.process_id = process_id;
  request.gpu_va = gpu_va;
  request.size = size;
  return impl_->operations.unmap(impl_->operations.context, &request);
}

void Gem5DeviceBackend::setMappingMemoryType(
    ProcessId process_id, uint64_t gpu_va, uint64_t size,
    uint8_t memory_type) {
  if (!impl_->operations.set_mapping_memory_type)
    return;
  auto request = abiStruct<rj_kmd_mapping_t>();
  request.process_id = process_id;
  request.memory_type = memory_type;
  request.gpu_va = gpu_va;
  request.size = size;
  (void)impl_->operations.set_mapping_memory_type(impl_->operations.context,
                                                  &request);
}

void Gem5DeviceBackend::write64(ProcessId process_id, uint64_t gpu_va,
                                uint64_t value) {
  if (!impl_->operations.write64)
    return;
  auto request = abiStruct<rj_kmd_write64_t>();
  request.process_id = process_id;
  request.gpu_va = gpu_va;
  request.value = value;
  (void)impl_->operations.write64(impl_->operations.context, &request);
}

void Gem5DeviceBackend::setDoorbellBase(ProcessId process_id,
                                        uintptr_t address) {
  std::lock_guard lock(impl_->doorbell_mutex);
  impl_->doorbell_bases[process_id] = address;
  for (auto &[key, watched] : impl_->watched_doorbells) {
    (void)key;
    if (watched.process_id != process_id)
      continue;
    watched.address =
        reinterpret_cast<const uint64_t *>(address + watched.offset);
    watched.last_value = ~uint64_t(0);
  }
}

bool Gem5DeviceBackend::createQueue(const QueueDescriptor &queue) {
  if (!impl_->operations.create_queue)
    return false;
  auto request = abiStruct<rj_kmd_queue_t>();
  request.process_id = queue.process_id;
  request.queue_id = queue.queue_id;
  request.ring_base_va = queue.ring_base_va;
  request.ring_size = queue.ring_size;
  request.doorbell_offset = queue.doorbell_offset;
  request.read_pointer_va = queue.read_pointer_va;
  request.write_pointer_va = queue.write_pointer_va;
  request.queue_descriptor_va = queue.queue_descriptor_va;
  request.host_accessible = queue.host_accessible;
  request.is_sdma = queue.is_sdma;
  if (impl_->operations.create_queue(impl_->operations.context, &request) != 0)
    return false;

  uintptr_t doorbell_base = queue.doorbell_address;
  if (doorbell_base == 0) {
    std::lock_guard lock(impl_->doorbell_mutex);
    auto it = impl_->doorbell_bases.find(queue.process_id);
    if (it != impl_->doorbell_bases.end())
      doorbell_base = it->second;
  }
  std::lock_guard lock(impl_->doorbell_mutex);
  impl_->watched_doorbells[queueKey(queue.process_id, queue.queue_id)] = {
      .process_id = queue.process_id,
      .queue_id = queue.queue_id,
      .offset = queue.doorbell_offset,
      .address =
          doorbell_base == 0
              ? nullptr
              : reinterpret_cast<const uint64_t *>(
                    doorbell_base + queue.doorbell_offset),
      .last_value = ~uint64_t(0),
  };
  return true;
}

void Gem5DeviceBackend::updateQueue(ProcessId process_id, QueueId queue_id,
                                    uint64_t ring_base_va,
                                    uint32_t ring_size) {
  if (!impl_->operations.update_queue)
    return;
  auto request = abiStruct<rj_kmd_queue_update_t>();
  request.process_id = process_id;
  request.queue_id = queue_id;
  request.ring_base_va = ring_base_va;
  request.ring_size = ring_size;
  (void)impl_->operations.update_queue(impl_->operations.context, &request);
}

void Gem5DeviceBackend::destroyQueue(ProcessId process_id, QueueId queue_id) {
  {
    std::lock_guard lock(impl_->doorbell_mutex);
    impl_->watched_doorbells.erase(queueKey(process_id, queue_id));
  }
  if (!impl_->operations.destroy_queue)
    return;
  auto request = abiStruct<rj_kmd_queue_key_t>();
  request.process_id = process_id;
  request.queue_id = queue_id;
  (void)impl_->operations.destroy_queue(impl_->operations.context, &request);
}

void Gem5DeviceBackend::postInterrupt(ProcessId process_id,
                                      uint32_t event_id) {
  if (impl_->interrupt_sink)
    impl_->interrupt_sink(process_id, event_id);
}

int Gem5DeviceBackend::notificationFd() const {
  return impl_->notification_fd;
}

size_t Gem5DeviceBackend::activeQueues() const {
  std::lock_guard lock(impl_->doorbell_mutex);
  return impl_->watched_doorbells.size();
}

size_t Gem5DeviceBackend::pendingDoorbells() const {
  std::lock_guard lock(impl_->doorbell_mutex);
  return impl_->pending_doorbells.size();
}

size_t Gem5DeviceBackend::drainDoorbells(
    rj_kmd_doorbell_record_t *records, size_t capacity) {
  if (impl_->notification_fd >= 0) {
    uint64_t value = 0;
    while (syscall(SYS_read, impl_->notification_fd, &value, sizeof(value)) >
           0) {
    }
  }
  std::lock_guard lock(impl_->doorbell_mutex);
  size_t count = 0;
  while (count < capacity && !impl_->pending_doorbells.empty()) {
    records[count++] = impl_->pending_doorbells.front();
    impl_->pending_doorbells.pop_front();
  }
  if (!impl_->pending_doorbells.empty() && impl_->notification_fd >= 0) {
    uint64_t one = 1;
    (void)syscall(SYS_write, impl_->notification_fd, &one, sizeof(one));
  }
  return count;
}

} // namespace rocjitsu::kmd
