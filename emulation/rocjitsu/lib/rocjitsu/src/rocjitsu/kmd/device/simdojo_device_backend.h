// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_SIMDOJO_DEVICE_BACKEND_H_
#define ROCJITSU_KMD_DEVICE_SIMDOJO_DEVICE_BACKEND_H_

#include "rocjitsu/kmd/device/device_backend.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/interrupt_sink.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace rocjitsu {
class SoC;
class SimulatedKfd;
namespace amdgpu {
class CommandProcessor;
}
namespace kmd {

class SimdojoDeviceBackend final : public DeviceBackend, private DeviceDebugger {
public:
  explicit SimdojoDeviceBackend(SoC &soc, uint32_t ordinal);
  ~SimdojoDeviceBackend() override;
  void bind_kfd(SimulatedKfd &kfd) { kfd_ = &kfd; }
  DeviceDebugger *debugger() override { return this; }
  uint32_t num_xcc() const override;
  bool supports_gws() const override { return true; }
  bool supports_xnack() const override { return true; }
  rj_code_arch_t architecture() const override;
  void configure(const DeviceApertures &apertures, uint32_t scratch_wave_divisor,
                 const DeviceCallbacks &callbacks) override;
  void set_memory_fault_reporter(MemoryFaultReporter *reporter) override;
  int register_process(KfdProcess &process, bool passthrough) override;
  void unregister_process(uint32_t process) override;
  void set_process_client_pid(uint32_t process, int32_t client_pid) override;
  void set_process_mem_fd(uint32_t process, int fd) override;
  std::optional<uint32_t> queue_xcc(uint32_t ordinal) const override;
  int create_queue(KfdProcess &process, const QueueDescriptor &queue) override;
  int update_queue(uint32_t process, uint32_t queue, uint64_t ring, uint32_t size,
                   uint32_t percentage) override;
  void destroy_queue(uint32_t process, uint32_t queue) override;
  int set_cu_mask(uint32_t process, uint32_t queue, std::span<const uint32_t> mask,
                  CuMaskGeometry geometry, bool is_sdma) override;
  void set_doorbell_base(uint32_t process, void *base) override;
  uint64_t read64(uint32_t process, uint64_t address) override;
  bool write64(uint32_t process, uint64_t address, uint64_t value) override;
  int submit_pm4(uint32_t process, uint64_t queue_key, amdgpu::Pm4Submission submission);
  void retire_pm4_queue(uint64_t queue_key);

private:
  void configure_debug_callbacks(const DeviceCallbacks &callbacks);
  void set_active(bool active) override;
  void release_queue(uint32_t process, uint32_t queue) override;
  std::function<bool(bool wait_for_ack)> prepare_queue_exception(uint32_t process, uint32_t queue,
                                                                 uint64_t mask) override;
  bool save_queue(const DebugQueueContext &context) override;
  int wave_state(const DebugQueueContext &, uint64_t, uint32_t &, uint32_t &) override;
  bool restore_queue(const DebugQueueContext &context) override;
  bool suspend_queue(uint32_t process, uint32_t queue,
                     const std::function<bool()> &publish) override;
  void clear_completed_queue(const DebugQueueContext &context) override;
  class FaultReporterBridge final : public amdgpu::MemoryFaultReporter {
  public:
    void set(kmd::MemoryFaultReporter *reporter) {
      reporter_.store(reporter, std::memory_order_release);
    }
    void report_memory_fault(uint32_t vmid, uint64_t address,
                             amdgpu::MemoryFaultCause cause) override;

  private:
    std::atomic<kmd::MemoryFaultReporter *> reporter_{nullptr};
  };

  static uint64_t queue_key(uint32_t process, uint32_t queue) {
    return (static_cast<uint64_t>(process) << 32) | queue;
  }
  SimulatedKfd *kfd_ = nullptr;
  SoC &soc_;
  uint32_t ordinal_;
  amdgpu::LegacyGpuVmAdapter legacy_vm_;
  amdgpu::InterruptSubscription interrupt_subscription_;
  FaultReporterBridge fault_reporter_;
  std::mutex state_mutex_;
  std::unordered_map<uint32_t, amdgpu::AddressSpaceHandle> processes_;
  std::unordered_map<uint64_t, amdgpu::QueueHandle> queues_;
  struct Pm4Queue {
    amdgpu::CommandProcessor *cp;
    uint32_t process;
    uint32_t queue_id;
  };
  std::mutex pm4_mutex_;
  std::unordered_map<uint64_t, Pm4Queue> pm4_queues_;
  uint32_t next_pm4_queue_id_ = 1;
};

} // namespace kmd
} // namespace rocjitsu

#endif
