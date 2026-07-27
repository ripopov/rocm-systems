// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file device_backend.h
/// @brief Device-facing operations used by the rocjitsu KFD implementation.

#ifndef ROCJITSU_KMD_DEVICE_DEVICE_BACKEND_H_
#define ROCJITSU_KMD_DEVICE_DEVICE_BACKEND_H_

#include "rocjitsu/code/rj_code.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace rocjitsu::kmd {

using ProcessId = uint32_t;
using QueueId = uint32_t;
using AllocationId = uint64_t;

/// @brief Immutable identity needed by KFD policy.
struct DeviceInfo {
  uint32_t gpu_id = 0;
  rj_code_arch_t architecture = ROCJITSU_CODE_ARCH_INVALID;
};

/// @brief Process apertures programmed into a device backend.
struct Apertures {
  uint64_t lds_base = 0;
  uint64_t lds_limit = 0;
  uint64_t scratch_base = 0;
  uint64_t scratch_limit = 0;
};

/// @brief Request for backend-owned storage.
struct AllocationRequest {
  ProcessId process_id = 0;
  uint32_t gpu_id = 0;
  uint64_t gpu_va = 0;
  uint64_t size = 0;
  uint32_t flags = 0;
  bool user_provided_va = false;
  bool shared_backing_required = false;
};

/// @brief A bounded slice of device-visible shared backing.
///
/// fd ownership transfers to SimulatedDriver on success. device_paddr is the
/// address programmed into the backend's GPUVM. A zero device_paddr asks an
/// in-process backend to use the server mapping as its physical identity.
/// size covers the complete fd extent beginning at fd_offset. For an
/// unaligned GPU VA, that extent begins at the containing page and therefore
/// includes the leading page offset before the requested logical allocation.
struct BackingSlice {
  AllocationId allocation_id = 0;
  uint64_t device_paddr = 0;
  uint64_t size = 0;
  int fd = -1;
  uint64_t fd_offset = 0;
};

/// @brief A GPU virtual mapping backed by a stable device allocation.
struct Mapping {
  ProcessId process_id = 0;
  AllocationId allocation_id = 0;
  uint64_t gpu_va = 0;
  uint64_t device_paddr = 0;
  uint64_t size = 0;
  uint8_t memory_type = 0;
};

/// @brief Hardware queue description after KFD has assigned queue identities.
struct QueueDescriptor {
  ProcessId process_id = 0;
  QueueId queue_id = 0;
  uint64_t ring_base_va = 0;
  uint32_t ring_size = 0;
  uint64_t read_pointer_va = 0;
  uint64_t write_pointer_va = 0;
  uint32_t doorbell_offset = 0;
  uintptr_t doorbell_address = 0;
  uint64_t queue_descriptor_va = 0;
  bool host_accessible = false;
  bool is_sdma = false;
};

using InterruptSink = std::function<void(ProcessId, uint32_t)>;
using ScratchBackingResolver = std::function<uint64_t(ProcessId)>;
using ScratchBackingAllocator = std::function<bool(ProcessId, uint64_t, size_t)>;

/// @brief Device behavior consumed by SimulatedDriver.
///
/// SimulatedDriver owns KFD policy and resource identity. Implementations own
/// GPU address translation, queue installation, doorbell routing, and
/// interrupt production. The interface deliberately contains no SoC,
/// simdojo, gem5, or configuration-loader types.
class DeviceBackend {
public:
  virtual ~DeviceBackend() = default;

  virtual const DeviceInfo &deviceInfo() const = 0;

  virtual int openProcess(ProcessId process_id, int32_t client_pid,
                          bool allow_host_pointer_passthrough) = 0;
  virtual void setProcessClientPid(ProcessId process_id, int32_t client_pid) = 0;
  virtual void closeProcess(ProcessId process_id) = 0;

  virtual int allocate(const AllocationRequest &request,
                       BackingSlice &slice) = 0;
  virtual void free(ProcessId process_id, AllocationId allocation_id) = 0;

  virtual void setApertures(const Apertures &apertures) = 0;
  virtual void setInterruptSink(InterruptSink sink) = 0;
  virtual void setScratchCallbacks(ScratchBackingResolver resolver,
                                   ScratchBackingAllocator allocator) = 0;

  virtual int map(const Mapping &mapping) = 0;
  virtual int unmap(ProcessId process_id, uint64_t gpu_va, uint64_t size) = 0;
  virtual void setMappingMemoryType(ProcessId process_id, uint64_t gpu_va,
                                    uint64_t size, uint8_t memory_type) = 0;
  virtual void write64(ProcessId process_id, uint64_t gpu_va, uint64_t value) = 0;

  virtual void setDoorbellBase(ProcessId process_id, uintptr_t address) = 0;
  virtual bool createQueue(const QueueDescriptor &queue) = 0;
  virtual void updateQueue(ProcessId process_id, QueueId queue_id,
                           uint64_t ring_base_va, uint32_t ring_size) = 0;
  virtual void destroyQueue(ProcessId process_id, QueueId queue_id) = 0;
};

} // namespace rocjitsu::kmd

#endif // ROCJITSU_KMD_DEVICE_DEVICE_BACKEND_H_
