// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_SIMDOJO_DEVICE_BACKEND_H_
#define ROCJITSU_KMD_DEVICE_SIMDOJO_DEVICE_BACKEND_H_

#include "rocjitsu/kmd/device/device_backend.h"

#include <memory>

namespace rocjitsu {

class SoC;

namespace kmd {

/// @brief DeviceBackend adapter for rocjitsu's existing functional SoC.
class SimdojoDeviceBackend final : public DeviceBackend {
public:
  explicit SimdojoDeviceBackend(SoC &soc, uint32_t gpu_id);
  ~SimdojoDeviceBackend() override;

  const DeviceInfo &deviceInfo() const override;
  int openProcess(ProcessId process_id, int32_t client_pid,
                  bool allow_host_pointer_passthrough) override;
  void setProcessClientPid(ProcessId process_id, int32_t client_pid) override;
  void closeProcess(ProcessId process_id) override;
  int allocate(const AllocationRequest &request,
               BackingSlice &slice) override;
  void free(ProcessId process_id, AllocationId allocation_id) override;
  void setApertures(const Apertures &apertures) override;
  void setInterruptSink(InterruptSink sink) override;
  void setScratchCallbacks(ScratchBackingResolver resolver,
                           ScratchBackingAllocator allocator) override;
  int map(const Mapping &mapping) override;
  int unmap(ProcessId process_id, uint64_t gpu_va, uint64_t size) override;
  void setMappingMemoryType(ProcessId process_id, uint64_t gpu_va,
                            uint64_t size, uint8_t memory_type) override;
  void write64(ProcessId process_id, uint64_t gpu_va, uint64_t value) override;
  void setDoorbellBase(ProcessId process_id, uintptr_t address) override;
  bool createQueue(const QueueDescriptor &queue) override;
  void updateQueue(ProcessId process_id, QueueId queue_id,
                   uint64_t ring_base_va, uint32_t ring_size) override;
  void destroyQueue(ProcessId process_id, QueueId queue_id) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::shared_ptr<DeviceBackend> makeSimdojoDeviceBackend(SoC &soc,
                                                        uint32_t gpu_id);

} // namespace kmd
} // namespace rocjitsu

#endif // ROCJITSU_KMD_DEVICE_SIMDOJO_DEVICE_BACKEND_H_
