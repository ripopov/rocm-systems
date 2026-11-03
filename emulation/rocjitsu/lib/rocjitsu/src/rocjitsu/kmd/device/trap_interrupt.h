// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include <cstdint>

namespace rocjitsu::kmd::detail {

enum class TrapInterruptSite { Unknown, Profiling, QueueException };

enum class TrapInterruptAbi { Unsupported, PreGfx12, Gfx12 };

constexpr TrapInterruptAbi trap_interrupt_abi(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return TrapInterruptAbi::PreGfx12;
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_CDNA5:
    return TrapInterruptAbi::Gfx12;
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_INVALID:
    return TrapInterruptAbi::Unsupported;
  }
  return TrapInterruptAbi::Unsupported;
}

constexpr bool uses_pre_gfx12_trap_interrupt_sites(rj_code_arch_t arch) {
  return trap_interrupt_abi(arch) == TrapInterruptAbi::PreGfx12;
}

constexpr TrapInterruptSite
classify_pre_gfx12_trap_interrupt_site(rj_code_arch_t arch, uint32_t payload_move, uint32_t delay) {
  constexpr uint32_t kSNop0 = 0xBF800000u;
  if (delay != kSNop0)
    return TrapInterruptSite::Unknown;

  uint32_t profiling_move = 0;
  uint32_t exception_move = 0;
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    profiling_move = 0xBEFC0073u;
    exception_move = 0xBEFC006Fu;
    break;
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    profiling_move = 0xBEFC0373u;
    exception_move = 0xBEFC036Fu;
    break;
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    profiling_move = 0xBEFD0073u;
    exception_move = 0xBEFD006Fu;
    break;
  default:
    return TrapInterruptSite::Unknown;
  }

  if (payload_move == profiling_move)
    return TrapInterruptSite::Profiling;
  if (payload_move == exception_move)
    return TrapInterruptSite::QueueException;
  return TrapInterruptSite::Unknown;
}

} // namespace rocjitsu::kmd::detail
