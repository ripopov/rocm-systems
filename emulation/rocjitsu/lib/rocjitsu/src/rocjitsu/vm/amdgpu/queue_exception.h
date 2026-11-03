// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

namespace rocjitsu::amdgpu {
enum class QueueExceptionDeliveryPolicy {
  /// Preserve the trap-handler acknowledgment and debugger-ownership protocol.
  WaitForAcknowledgment,
  /// Publish without waiting for ROCr; fail the engine if publication fails.
  PublishOrFail,
  /// Publish without waiting; the caller cancels all queue replicas and fails
  /// the engine after their acknowledgments, even if publication fails.
  PublishAndCancel,
};
} // namespace rocjitsu::amdgpu
