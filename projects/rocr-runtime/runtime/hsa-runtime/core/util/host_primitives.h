////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_UTIL_HOST_PRIMITIVES_H_
#define HSA_RUNTIME_CORE_UTIL_HOST_PRIMITIVES_H_

#if defined(__i386__) || defined(__x86_64__)
#include <x86intrin.h>
#elif defined(_M_IX86) || defined(_M_X64)
#include <intrin.h>
#endif

#if defined(_MSC_VER)
#define ROCR_HOST_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ROCR_HOST_FORCEINLINE inline __attribute__((always_inline))
#else
#define ROCR_HOST_FORCEINLINE inline
#endif

namespace rocr {
namespace host {

// Flush one data-cache line to the point of coherency. Linux enables EL0
// cache maintenance on AArch64, or traps and emulates the operation for CPUs
// that require a kernel workaround. The DSB makes completion visible to
// devices in the outer-shareable domain before this function returns.
static ROCR_HOST_FORCEINLINE void FlushCacheLine(const void* address) {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
  _mm_clflush(address);
#elif defined(__aarch64__)
  __asm__ __volatile__("dc civac, %0" : : "r"(address) : "memory");
  __asm__ __volatile__("dsb osh" : : : "memory");
#else
#error "ROCr host cache-line flush is not implemented for this architecture."
#endif
}

// Hint that the processor is in a contended spin loop.
static ROCR_HOST_FORCEINLINE void CpuRelax() {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
  _mm_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield" : : : "memory");
#else
#error "ROCr host spin-loop hint is not implemented for this architecture."
#endif
}

// Order prior writes, including writes to write-combining and device memory,
// before subsequent writes observed outside the CPU's inner-shareable domain.
static ROCR_HOST_FORCEINLINE void StoreFence() {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
  _mm_sfence();
#elif defined(__aarch64__)
  __asm__ __volatile__("dmb oshst" : : : "memory");
#else
#error "ROCr host store fence is not implemented for this architecture."
#endif
}

// Order prior loads and stores before subsequent loads and stores, including
// accesses to device memory in the outer-shareable domain.
static ROCR_HOST_FORCEINLINE void FullFence() {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
  _mm_mfence();
#elif defined(__aarch64__)
  __asm__ __volatile__("dmb osh" : : : "memory");
#else
#error "ROCr host full fence is not implemented for this architecture."
#endif
}

}  // namespace host
}  // namespace rocr

#undef ROCR_HOST_FORCEINLINE

#endif  // HSA_RUNTIME_CORE_UTIL_HOST_PRIMITIVES_H_
