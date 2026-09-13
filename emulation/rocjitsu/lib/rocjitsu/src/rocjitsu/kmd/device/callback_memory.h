// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_CALLBACK_MEMORY_H_
#define ROCJITSU_KMD_DEVICE_CALLBACK_MEMORY_H_

#include "rocjitsu/kmd/device/device_memory.h"
#include "rocjitsu/kmd/linux/callback_dispatcher.h"
#include "rocjitsu/kmd/server.h"

namespace rocjitsu::kmd {

/// Converts the C embedding memory callbacks to common KFD allocation leases.
/// The owner and callback context must outlive every lease. Terminal cleanup
/// failures are latched for the server's owner-thread error reporting; further
/// allocation/mapping is refused but unmap/free still attempt cleanup.
class CallbackMemory final : public DeviceMemory {
public:
  CallbackMemory(uint32_t gpu_id, const rj_kmd_backend_ops_t &operations,
                 CallbackDispatcher &dispatcher)
      : gpu_id_(gpu_id), operations_(operations), dispatcher_(dispatcher) {
    if (operations.abi_version != RJ_KMD_SERVER_ABI_VERSION ||
        operations.struct_size < sizeof(operations) || !operations.allocate || !operations.free ||
        !operations.map || !operations.unmap)
      throw std::invalid_argument("invalid embedded KMD memory callbacks");
  }

  int failure() const { return failure_.load(std::memory_order_acquire); }
  void fail(int status) noexcept { latch(status); }

  int allocate(const DeviceAllocationRequest &source,
               std::shared_ptr<DeviceAllocation> &allocation) override {
    if (int status = failure())
      return status;
    // Reserve C++ bookkeeping before acquiring storage from the host.
    auto lease = std::make_shared<Lease>(*this, source.process_id);
    auto request = record<rj_kmd_allocation_request_t>();
    request.process_id = source.process_id;
    request.gpu_id = gpu_id_;
    request.gpu_va = source.gpu_va;
    request.size = source.size;
    request.flags = source.flags;
    auto slice = record<rj_kmd_backing_slice_t>();
    slice.fd = -1;
    const int status =
        call([&] { return operations_.allocate(operations_.context, &request, &slice); });
    if (status)
      return status;
    lease->id = slice.allocation_id;
    lease->device_address = slice.device_paddr;
    lease->fd = slice.fd;
    lease->file_offset = slice.fd_offset;
    lease->size = slice.size;
    lease->active = true;
    if (slice.abi_version != RJ_KMD_SERVER_ABI_VERSION || slice.struct_size < sizeof(slice) ||
        slice.fd < 0)
      return -EPROTO; // The temporary lease rolls back successful host allocation.
    allocation = std::move(lease);
    return 0;
  }

  int map(const DeviceMapping &source) override {
    if (int status = failure())
      return status;
    auto request = record<rj_kmd_mapping_t>();
    request.process_id = source.process_id;
    request.allocation_id = source.allocation_id;
    request.gpu_va = source.gpu_va;
    request.device_paddr = source.device_address;
    request.size = source.size;
    request.memory_type = source.memory_type;
    return call([&] { return operations_.map(operations_.context, &request); });
  }

  int unmap(uint32_t process, uint64_t gpu_va, uint64_t size) override {
    auto request = record<rj_kmd_unmap_request_t>();
    request.process_id = process;
    request.gpu_va = gpu_va;
    request.size = size;
    return call([&] { return operations_.unmap(operations_.context, &request); });
  }

private:
  struct Lease final : DeviceAllocation {
    Lease(CallbackMemory &owner, uint32_t process) : owner(owner), process(process) {}
    ~Lease() override {
      if (active)
        owner.release(process, id);
      if (fd >= 0)
        ::close(fd);
    }
    CallbackMemory &owner;
    uint32_t process;
    bool active = false;
  };

  template <typename T> static T record() {
    T value{};
    value.abi_version = RJ_KMD_SERVER_ABI_VERSION;
    value.struct_size = sizeof(T);
    return value;
  }

  template <typename Function> int call(Function &&function) {
    const int status = dispatcher_.invoke(std::forward<Function>(function));
    if (status > 0) {
      latch(-EPROTO);
      return -EPROTO;
    }
    return status;
  }

  void latch(int status) noexcept {
    int expected = 0;
    failure_.compare_exchange_strong(expected, status, std::memory_order_acq_rel);
  }

  void release(uint32_t process, uint64_t id) noexcept {
    auto request = record<rj_kmd_free_request_t>();
    request.process_id = process;
    request.allocation_id = id;
    try {
      if (int status = call([&] { return operations_.free(operations_.context, &request); }))
        latch(status);
    } catch (const std::bad_alloc &) {
      latch(-ENOMEM);
    } catch (...) {
      latch(-EIO);
    }
  }

  uint32_t gpu_id_;
  rj_kmd_backend_ops_t operations_;
  CallbackDispatcher &dispatcher_;
  std::atomic<int> failure_{0};
};

} // namespace rocjitsu::kmd

#endif
