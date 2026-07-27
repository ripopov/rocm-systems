// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_DEVICE_INFO_H_
#define ROCJITSU_KMD_LINUX_DEVICE_INFO_H_

#include "rocjitsu/kmd/linux/sysfs.h"

#include <cstdint>
#include <vector>

namespace rocjitsu::config {
struct KfdDeviceConfig;
}

namespace rocjitsu::kmd {

Sysfs::GpuInfo deviceInfoFromConfig(const config::KfdDeviceConfig &device,
                                    uint32_t num_xcc);
std::vector<Sysfs::GpuInfo>
deviceInfoFromConfig(const std::vector<config::KfdDeviceConfig> &devices,
                     uint32_t num_xcc);

} // namespace rocjitsu::kmd

#endif // ROCJITSU_KMD_LINUX_DEVICE_INFO_H_
