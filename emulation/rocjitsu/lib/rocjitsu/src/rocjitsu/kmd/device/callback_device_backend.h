// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_CALLBACK_DEVICE_BACKEND_H_
#define ROCJITSU_KMD_DEVICE_CALLBACK_DEVICE_BACKEND_H_

#include "rocjitsu/kmd/device/callback_memory.h"
#include "rocjitsu/kmd/device/device_backend.h"
#include "rocjitsu/kmd/linux/doorbell_monitor.h"

namespace rocjitsu::kmd {

/// External device implementation for the C embedding boundary. KFD policy
/// remains in SimulatedKfd; only device operations cross owner-thread callbacks.
/// The server destroys the KFD and its leases before this backend/context, and
/// must report failure() as a terminal error instead of accepting more work.
class CallbackDeviceBackend final : public DeviceBackend {
public:
  CallbackDeviceBackend(rj_code_arch_t architecture, uint32_t gpu_id,
                        const rj_kmd_backend_ops_t &operations, CallbackDispatcher &dispatcher);
  ~CallbackDeviceBackend() override;
  DeviceMemory *memory() override { return &memory_; }
  rj_code_arch_t architecture() const override { return architecture_; }
  void configure(const DeviceApertures &, uint32_t, const DeviceCallbacks &) override;
  void set_memory_fault_reporter(MemoryFaultReporter *reporter) override { reporter_ = reporter; }
  int register_process(KfdProcess &, bool passthrough) override;
  void prepare_process_teardown(uint32_t process) override;
  void unregister_process(uint32_t process) override;
  void set_process_client_pid(uint32_t process, int32_t pid) override;
  void set_process_mem_fd(uint32_t, int fd) override;
  std::optional<uint32_t> queue_xcc(uint32_t) const override { return 0; }
  int create_queue(KfdProcess &, const QueueDescriptor &) override;
  int update_queue(uint32_t process, uint32_t queue, uint64_t ring, uint32_t size,
                   uint32_t percentage) override;
  void destroy_queue(uint32_t process, uint32_t queue) override;
  void set_doorbell_base(uint32_t process, void *base) override;
  uint64_t read64(uint32_t process, uint64_t address) override;
  bool write64(uint32_t process, uint64_t address, uint64_t value) override;

  int failure() const { return memory_.failure(); }
  uint64_t active_processes() const { return active_processes_.load(); }
  DoorbellMonitor &doorbells() { return doorbells_; }
  void post_interrupt(uint32_t process, uint32_t event);
  void post_memory_fault(uint32_t process, uint64_t address, MemoryFaultCause cause);
  void set_quiescence_probe(rj_kmd_quiescence_fn probe, void *context) {
    quiescence_probe_ = probe;
    quiescence_context_ = context;
  }

private:
  rj_kmd_quiescence_fn quiescence_probe_ = nullptr;
  void *quiescence_context_ = nullptr;
  template <typename T> static T record() {
    T value{};
    value.abi_version = RJ_KMD_SERVER_ABI_VERSION;
    value.struct_size = sizeof(T);
    return value;
  }
  static void require_success(int status);
  int status(int result);
  int process_call(rj_kmd_process_fn, uint32_t process, int32_t pid);
  void shutdown(); // owner-thread only; mandatory global quiescence boundary
  void teardown_result(int result);

  rj_code_arch_t architecture_;
  rj_kmd_backend_ops_t operations_;
  CallbackDispatcher &dispatcher_;
  CallbackMemory memory_;
  DoorbellMonitor doorbells_;
  DeviceCallbacks callbacks_;
  MemoryFaultReporter *reporter_ = nullptr;
  struct Process {
    int32_t pid;
    void *doorbell_base = nullptr;
  };
  std::map<uint32_t, Process> processes_; // owner-thread only
  std::atomic<uint64_t> active_processes_{0};
  bool shutdown_ = false;
};

} // namespace rocjitsu::kmd

#endif
