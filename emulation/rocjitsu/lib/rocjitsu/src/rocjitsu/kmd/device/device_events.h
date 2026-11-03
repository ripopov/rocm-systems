// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>

namespace rocjitsu::kmd {

/// Value snapshot for native event handling, never a borrowed process or wave.
/// A missing queue has zero context/runtime metadata; its process may still exist.
struct DeviceEventContext {
  uint32_t process_id = 0;
  uint32_t queue_id = 0;
  uint32_t gpu_id = 0;
  uint64_t context_address = 0;
  uint32_t context_size = 0;
  bool debugger_attached = false;
  bool debug_stop_publishable = false;
  bool has_runtime_signal = false;
  bool xnack_enabled = false;
  std::optional<uint32_t> doorbell_id;
};

struct DeviceDebugEvent {
  DeviceEventContext context;
  uint64_t exception_mask = 0;
  bool save_queue = false;
  /// Native retained stops also reserve runtime delivery when publication loses its claim.
  bool retain_on_rejection = false;
};

struct DeviceDebugEventResult {
  bool delivered = false;
  bool runtime_failure_debuggable = false;
};

/// @brief KFD policy used by native device event handlers.
/// @details KFD owns this interface and outlives attached device callbacks. Each
/// operation owns its KFD locking; callers must not hold locks acquired by KFD's
/// device calls. Snapshot metadata does not authorize publication: publication
/// rechecks the live session/queue claim. The owner quiesces execution and detaches
/// callbacks before destroying policy state. No method retains native objects.
class DeviceEvents {
public:
  virtual ~DeviceEvents() = default;
  /// No value means the process is gone. Missing queues retain no image metadata.
  virtual std::optional<DeviceEventContext> event_context(uint32_t process, uint32_t queue) = 0;
  /// Match KFD access modes and return a mask of hardware watch slots.
  virtual uint32_t matching_watchpoints(uint32_t process, uint64_t address, uint32_t bytes,
                                        bool is_write, bool is_atomic) = 0;
  /// Return the subscribed debugger mask, optionally reserving an unclaimed
  /// exception for runtime delivery as one atomic KFD policy decision.
  virtual uint64_t claim_debug_exception(uint32_t process, uint32_t queue, uint64_t mask,
                                         bool reserve_runtime_if_unclaimed) = 0;
  virtual void reserve_runtime_exception(uint32_t process, uint32_t queue, uint64_t mask) = 0;
  /// The adapter has already stopped its wave. Save a queue image when asked,
  /// then attempt notification; rejection tells the adapter to undo its stop.
  virtual DeviceDebugEventResult publish_debug_stop(const DeviceDebugEvent &event) = 0;
  /// Deliver after releasing CU locks, then settle the KFD reservation. Waiting
  /// for acknowledgment uses KFD liveness/cancellation; no native object is kept.
  virtual bool deliver_runtime_exception(uint32_t gpu_ordinal, uint32_t process, uint32_t queue,
                                         uint64_t mask, bool wait_for_ack, bool retain_failure) = 0;
  /// Settle a delivery attempt, including failure to schedule a native delivery.
  virtual void finish_runtime_exception(uint32_t process, uint32_t queue, uint64_t mask,
                                        bool delivered, bool retain_failure) = 0;
};

} // namespace rocjitsu::kmd
