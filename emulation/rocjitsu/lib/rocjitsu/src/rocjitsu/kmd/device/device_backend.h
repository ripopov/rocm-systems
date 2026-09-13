// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_DEVICE_BACKEND_H_
#define ROCJITSU_KMD_DEVICE_DEVICE_BACKEND_H_

#include <cstdint>
#include <optional>

namespace rocjitsu::kmd {

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
  bool xcd_fanout = false;
  uint64_t queue_desc_va = 0;
  uint64_t exception_status_va = 0;
  uint32_t exception_event_id = 0;
};

/// @brief Device operations used by the common KFD policy implementation.
/// @details KFD owns queue identities and backing mappings. Queue installation
/// borrows those mappings until destroy_queue() returns. Implementations must
/// stop all accesses to a destroyed queue before returning to KFD teardown.
class DeviceBackend {
public:
  virtual ~DeviceBackend() = default;
  /// Return the debugger-visible XCC of a queue, or no value if unavailable.
  virtual std::optional<uint32_t> queue_xcc(uint32_t ordinal) const = 0;
  /// Return zero on installation, or a negative errno with no retained queue
  /// resources or references to the borrowed mappings.
  virtual int create_queue(const QueueDescriptor &queue) = 0;
  virtual void update_queue(uint32_t process, uint32_t queue, uint64_t ring, uint32_t size,
                            uint32_t percentage) = 0;
  virtual void destroy_queue(uint32_t process, uint32_t queue) = 0;
  virtual void set_doorbell_base(uint32_t process, void *base) = 0;
  virtual uint64_t read64(uint32_t process, uint64_t address) = 0;
  virtual void write64(uint32_t process, uint64_t address, uint64_t value) = 0;
};

} // namespace rocjitsu::kmd

#endif
