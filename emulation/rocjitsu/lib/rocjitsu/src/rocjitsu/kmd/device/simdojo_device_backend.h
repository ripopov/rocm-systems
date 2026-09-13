// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_SIMDOJO_DEVICE_BACKEND_H_
#define ROCJITSU_KMD_DEVICE_SIMDOJO_DEVICE_BACKEND_H_

#include "rocjitsu/kmd/device/device_backend.h"

namespace rocjitsu {
class SoC;
namespace kmd {

class SimdojoDeviceBackend final : public DeviceBackend, private DeviceDebugger {
public:
  explicit SimdojoDeviceBackend(SoC &soc) : soc_(soc) {}
  DeviceDebugger *debugger() override { return this; }
  rj_code_arch_t architecture() const override;
  void configure(const DeviceApertures &apertures, uint32_t scratch_wave_divisor,
                 const DeviceCallbacks &callbacks) override;
  void set_memory_fault_reporter(MemoryFaultReporter *reporter) override;
  int register_process(KfdProcess &process, bool passthrough) override;
  void unregister_process(uint32_t process) override;
  void set_process_client_pid(uint32_t process, int32_t client_pid) override;
  void set_process_mem_fd(uint32_t process, int fd) override;
  std::optional<uint32_t> queue_xcc(uint32_t ordinal) const override;
  int create_queue(const QueueDescriptor &queue) override;
  void update_queue(uint32_t process, uint32_t queue, uint64_t ring, uint32_t size,
                    uint32_t percentage) override;
  void destroy_queue(uint32_t process, uint32_t queue) override;
  void set_doorbell_base(uint32_t process, void *base) override;
  uint64_t read64(uint32_t process, uint64_t address) override;
  void write64(uint32_t process, uint64_t address, uint64_t value) override;

private:
  void configure_debug_callbacks(const DeviceCallbacks &callbacks);
  void set_active(bool active) override;
  void release_queue(uint32_t process, uint32_t queue) override;
  bool signal_queue_exception(uint32_t process, uint32_t queue, uint64_t mask) override;
  std::optional<uint64_t> save_queue(const DebugQueueContext &context) override;
  bool restore_queue(const DebugQueueContext &context) override;
  bool suspend_queue(uint32_t process, uint32_t queue,
                     const std::function<bool()> &publish) override;
  void clear_completed_queue(const DebugQueueContext &context) override;
  SoC &soc_;
};

} // namespace kmd
} // namespace rocjitsu

#endif
