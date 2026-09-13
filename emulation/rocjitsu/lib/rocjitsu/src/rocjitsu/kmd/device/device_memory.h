// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_DEVICE_MEMORY_H_
#define ROCJITSU_KMD_DEVICE_DEVICE_MEMORY_H_

#include <cstdint>
#include <memory>

namespace rocjitsu::kmd {

struct DeviceAllocationRequest {
  uint32_t process_id;
  uint64_t gpu_va;
  uint64_t size;
  uint32_t flags;
};

/// A stable allocation lease, shared by KFD allocations and IPC imports.
/// Destruction releases device storage, after all mappings referring to it
/// have been removed. The lease may outlive its originating process, but not
/// the DeviceBackend. Its descriptor is borrowed: KFD duplicates it for each
/// allocation record. The backend must not resize or recycle the range while
/// a lease exists. The range starts at file_offset and contains size bytes.
class DeviceAllocation {
public:
  virtual ~DeviceAllocation() = default;
  uint64_t id = 0;
  uint64_t device_address = 0;
  int fd = -1;
  uint64_t file_offset = 0;
  uint64_t size = 0;
};

struct DeviceMapping {
  uint32_t process_id;
  uint64_t allocation_id;
  uint64_t gpu_va;
  uint64_t device_address;
  uint64_t size;
  uint8_t memory_type;
};

/// Optional backend-owned storage. Without this capability, common KFD uses
/// host memory as upstream's in-process simulator does. An allocation failure
/// never falls back to host storage when a backend provides this capability.
class DeviceMemory {
public:
  virtual ~DeviceMemory() = default;
  /// Return zero with a lease, or negative errno without retained resources.
  virtual int allocate(const DeviceAllocationRequest &request,
                       std::shared_ptr<DeviceAllocation> &allocation) = 0;
  /// Map/reconfigure a complete allocation atomically. A negative errno leaves
  /// the previous mapping unchanged. KFD retains its allocation lease until
  /// unmap succeeds or unregister_process has quiesced all process mappings.
  virtual int map(const DeviceMapping &mapping) = 0;
  virtual int unmap(uint32_t process, uint64_t gpu_va, uint64_t size) = 0;
};

} // namespace rocjitsu::kmd

#endif
