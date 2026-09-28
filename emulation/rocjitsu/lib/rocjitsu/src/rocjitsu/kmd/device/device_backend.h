// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_DEVICE_BACKEND_H_
#define ROCJITSU_KMD_DEVICE_DEVICE_BACKEND_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/kmd/device/device_memory.h"
#include "rocjitsu/kmd/device/memory_fault.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

namespace rocjitsu {
class KfdProcess;
namespace kmd {

struct DeviceApertures {
  uint64_t lds_base = 0;
  uint64_t lds_limit = 0;
  uint64_t scratch_base = 0;
  uint64_t scratch_limit = 0;
};

/// Device execution reports events and requests scratch backing from KFD.
/// Callbacks must run without locks that KFD's device calls also acquire.
struct DeviceTrapHandler {
  uint64_t tba;
  uint64_t tma;
  bool debug_enabled;
};

struct DeviceCallbacks {
  std::function<void(uint32_t process, uint32_t event)> interrupt;
  std::function<uint64_t(uint32_t process)> scratch_backing;
  std::function<bool(uint32_t process, uint64_t address, size_t size)> allocate_scratch;
  std::function<std::optional<DeviceTrapHandler>(uint32_t process)> trap_handler;
};

/// Optional device execution support for KFD debugger sessions. Session
/// authorization and event bookkeeping remain in the common KFD implementation.
struct DebugQueueContext {
  uint32_t process_id;
  uint32_t queue_id;
  uint64_t base;
  uint32_t size;
  int32_t client_pid;
  /// Authorized target-memory FD borrowed only for the duration of the call.
  int target_mem_fd;
  /// KFD commits runtime ownership after a valid restore, before execution wakes.
  std::function<void()> resolve_runtime_exceptions{};
  uint32_t control_stack_capacity = 0;
  bool runtime_save = false;
  std::function<void()> release_debug_image{};
};

class DeviceDebugger {
public:
  virtual ~DeviceDebugger() = default;
  virtual void set_active(bool active) = 0;
  virtual void release_queue(uint32_t process, uint32_t queue) = 0;
  /// Stop queue replicas before the caller takes its publication lock.
  /// The returned publisher is synchronous and must not outlive the operation.
  /// Its argument selects whether it waits for the runtime to acknowledge.
  virtual std::function<bool(bool wait_for_ack)>
  prepare_queue_exception(uint32_t process, uint32_t queue, uint64_t mask) = 0;
  /// Publish settled stopped waves, or fail without claiming an available
  /// debugger/runtime image. The caller owns queue and image bookkeeping.
  virtual bool save_queue(const DebugQueueContext &context) = 0;
  virtual int wave_state(const DebugQueueContext &context, uint64_t destination,
                         uint32_t &control_size, uint32_t &wave_size) = 0;
  virtual bool restore_queue(const DebugQueueContext &context) = 0;
  /// Stop execution, publish outside device locks, and undo a failed stop.
  virtual bool suspend_queue(uint32_t process, uint32_t queue,
                             const std::function<bool()> &publish) = 0;
  virtual void clear_completed_queue(const DebugQueueContext &context) = 0;
};

/// @brief A queue programmed by KFD, independent of the execution engine.
struct QueueDescriptor {
  uint32_t process_id = 0;
  uint32_t queue_id = 0;
  uint32_t ordinal = 0;
  uint64_t ring_base_va = 0;
  uint32_t ring_size = 0;
  uint64_t read_ptr_va = 0;
  uint64_t write_ptr_va = 0;
  uint32_t doorbell_offset = 0;
  void *doorbell_base = nullptr;
  bool host_accessible = false;
  bool is_sdma = false;
  bool is_pm4 = false;
  uint32_t scheduling_percentage = 100;
  uint32_t sdma_engine_id = 0;
  bool xcd_fanout = false;
  uint64_t queue_desc_va = 0;
  uint64_t exception_status_va = 0;
  uint32_t exception_event_id = 0;
};

struct CuMaskGeometry {
  uint32_t shader_engines = 0;
  uint32_t arrays_per_engine = 0;
  uint32_t cu_per_array = 0;
  uint32_t active_cus = 0;
};

/// @brief Device operations used by the common KFD policy implementation.
/// @details KFD owns queue identities and backing mappings. Queue installation
/// borrows those mappings until destroy_queue() returns. Implementations must
/// stop all accesses to a destroyed queue before returning to KFD teardown.
class DeviceBackend {
public:
  virtual ~DeviceBackend() = default;
  /// Null means debugger execution is unsupported, not a successful no-op.
  virtual DeviceDebugger *debugger() { return nullptr; }
  virtual DeviceMemory *memory() { return nullptr; }
  virtual uint32_t num_xcc() const { return 1; }
  virtual bool supports_gws() const { return false; }
  virtual bool supports_xnack() const { return false; }
  virtual rj_code_arch_t architecture() const = 0;
  virtual void configure(const DeviceApertures &apertures, uint32_t scratch_wave_divisor,
                         const DeviceCallbacks &callbacks) = 0;
  virtual void set_memory_fault_reporter(MemoryFaultReporter *reporter) = 0;
  /// Borrow the common process address space until unregister_process returns.
  /// A negative errno leaves no device-side process state. Passthrough is only
  /// valid when the application and device execute in the same host process.
  virtual int register_process(KfdProcess &process, bool passthrough) = 0;
  /// Stop all process accesses and remove remaining device mappings before
  /// returning. Allocation leases can survive this process through KFD IPC.
  virtual void unregister_process(uint32_t process) = 0;
  virtual void set_process_client_pid(uint32_t process, int32_t client_pid) = 0;
  /// Borrow a debugger-authorized /proc/pid/mem descriptor; -1 revokes it.
  virtual void set_process_mem_fd(uint32_t process, int fd) = 0;
  /// Return the debugger-visible XCC of a queue, or no value if unavailable.
  virtual std::optional<uint32_t> queue_xcc(uint32_t ordinal) const = 0;
  /// Return zero on installation, or a negative errno with no retained queue
  /// resources or references to the borrowed mappings.
  virtual int create_queue(KfdProcess &, const QueueDescriptor &queue) = 0;
  /// Return zero after updating the queue, or negative errno with its previous
  /// configuration intact. KFD publishes the new snapshot only after success.
  virtual int update_queue(uint32_t process, uint32_t queue, uint64_t ring, uint32_t size,
                           uint32_t percentage) = 0;
  virtual void destroy_queue(uint32_t process, uint32_t queue) = 0;
  virtual int set_cu_mask(uint32_t process, uint32_t queue, std::span<const uint32_t> mask,
                          CuMaskGeometry geometry, bool is_sdma) {
    (void)process;
    (void)queue;
    (void)mask;
    (void)geometry;
    (void)is_sdma;
    return -ENOTSUP;
  }
  virtual void set_doorbell_base(uint32_t process, void *base) = 0;
  virtual uint64_t read64(uint32_t process, uint64_t address) = 0;
  virtual bool write64(uint32_t process, uint64_t address, uint64_t value) = 0;
};

} // namespace kmd
} // namespace rocjitsu

#endif
