// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_MEMORY_FAULT_H_
#define ROCJITSU_KMD_DEVICE_MEMORY_FAULT_H_

#include <cstdint>

namespace rocjitsu::kmd {

/// @brief Why an access was refused, mapped onto what the driver reports.
/// @details The KFD ABI separates a page that is not there from one that is
/// there but refuses the write, and the runtime reads the two fields
/// separately, so collapsing them would misstate the cause.
enum class MemoryFaultCause : uint8_t {
  NotPresent, ///< No mapping at the address, or it is inaccessible outright.
  ReadOnly,   ///< The page is readable, but rejected the write.
  /// The backend could not establish the cause. Report neither failure bit;
  /// this does not by itself mean the reported fault address is imprecise.
  Indeterminate,
};

class MemoryFaultReporter {
public:
  virtual ~MemoryFaultReporter() = default;

  /// @brief Report that @p addr could not be serviced for @p vmid.
  ///
  /// @details May be called from device execution threads. Each device binds
  /// its own reporter, identifying the faulting GPU independently of @p vmid.
  ///
  /// Delivery must occur after releasing device address-space and page-table
  /// locks: the reporter may acquire KFD locks and call back into the device.
  ///
  /// Reporters must not perform an access that itself faults, which would
  /// recursively reenter fault delivery.
  virtual void report_memory_fault(uint32_t vmid, uint64_t addr, MemoryFaultCause cause) = 0;
};

} // namespace rocjitsu::kmd

#endif
