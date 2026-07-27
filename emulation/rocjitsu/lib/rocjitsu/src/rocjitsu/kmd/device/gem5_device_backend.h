// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_GEM5_DEVICE_BACKEND_H_
#define ROCJITSU_KMD_DEVICE_GEM5_DEVICE_BACKEND_H_

#include "rocjitsu/kmd/device/device_backend.h"
#include "rocjitsu/kmd/server.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace rocjitsu::kmd {

class Gem5DeviceBackend final : public DeviceBackend {
public:
  Gem5DeviceBackend(DeviceInfo info, const rj_kmd_backend_ops_t &operations);
  ~Gem5DeviceBackend() override;

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

  void postInterrupt(ProcessId process_id, uint32_t event_id);
  int notificationFd() const;
  size_t activeQueues() const;
  size_t pendingDoorbells() const;
  size_t drainDoorbells(rj_kmd_doorbell_record_t *records, size_t capacity);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rocjitsu::kmd

#endif // ROCJITSU_KMD_DEVICE_GEM5_DEVICE_BACKEND_H_
