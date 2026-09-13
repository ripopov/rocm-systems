// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_DEVICE_SIMDOJO_DEVICE_BACKEND_H_
#define ROCJITSU_KMD_DEVICE_SIMDOJO_DEVICE_BACKEND_H_

#include "rocjitsu/kmd/device/device_backend.h"

namespace rocjitsu {
class SoC;
namespace kmd {

class SimdojoDeviceBackend final : public DeviceBackend {
public:
  explicit SimdojoDeviceBackend(SoC &soc) : soc_(soc) {}
  std::optional<uint32_t> queue_xcc(uint32_t ordinal) const override;
  int create_queue(const QueueDescriptor &queue) override;
  void update_queue(uint32_t process, uint32_t queue, uint64_t ring, uint32_t size,
                    uint32_t percentage) override;
  void destroy_queue(uint32_t process, uint32_t queue) override;
  void set_doorbell_base(uint32_t process, void *base) override;
  uint64_t read64(uint32_t process, uint64_t address) override;
  void write64(uint32_t process, uint64_t address, uint64_t value) override;

private:
  SoC &soc_;
};

} // namespace kmd
} // namespace rocjitsu

#endif
