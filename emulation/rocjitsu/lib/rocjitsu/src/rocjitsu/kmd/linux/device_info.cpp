// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/device_info.h"

#include "rocjitsu/config/config_loader.h"

namespace rocjitsu::kmd {

Sysfs::GpuInfo deviceInfoFromConfig(const config::KfdDeviceConfig &dev,
                                    uint32_t num_xcc) {
  Sysfs::GpuInfo gpu{};
  if (!dev.present)
    return gpu;
  gpu.gpu_id = dev.gpu_id;
  gpu.gfx_target_version = dev.gfx_target_version;
  gpu.vendor_id = dev.vendor_id;
  gpu.device_id = dev.device_id;
  gpu.family_id = dev.family_id;
  gpu.unique_id = dev.unique_id;
  gpu.marketing_name = dev.marketing_name;
  gpu.drm_render_minor = dev.drm_render_minor;
  gpu.revision_id = dev.revision_id;
  gpu.pci_revision_id = dev.pci_revision_id;
  gpu.simd_count = dev.simd_count;
  gpu.max_waves_per_simd = dev.max_waves_per_simd;
  gpu.num_shader_engines = dev.num_shader_engines;
  gpu.num_shader_arrays_per_engine = dev.num_shader_arrays_per_engine;
  gpu.num_cu_per_sh = dev.num_cu_per_sh;
  gpu.simd_per_cu = dev.simd_per_cu;
  gpu.wave_front_size = dev.wave_front_size;
  gpu.max_slots_scratch_cu = dev.max_slots_scratch_cu;
  gpu.local_mem_size = dev.local_mem_size;
  gpu.vram_type = dev.vram_type;
  gpu.lds_size_kb = dev.lds_size_kb;
  gpu.mem_width = dev.mem_width;
  gpu.mem_clk_max = dev.mem_clk_max;
  gpu.l1_size_kb = dev.l1_size_kb;
  gpu.l1_line_size = dev.l1_line_size;
  gpu.l1_assoc = dev.l1_assoc;
  gpu.l2_size_kb = dev.l2_size_kb;
  gpu.l2_line_size = dev.l2_line_size;
  gpu.l2_assoc = dev.l2_assoc;
  gpu.num_sdma_engines = dev.num_sdma_engines;
  gpu.num_sdma_xgmi_engines = dev.num_sdma_xgmi_engines;
  gpu.num_cp_queues = dev.num_cp_queues;
  gpu.max_engine_clk_fcompute = dev.max_engine_clk_fcompute;
  gpu.location_id = dev.location_id;
  gpu.hive_id = dev.hive_id;
  gpu.domain = dev.domain;
  gpu.num_xcc = num_xcc;
  return gpu;
}

std::vector<Sysfs::GpuInfo>
deviceInfoFromConfig(const std::vector<config::KfdDeviceConfig> &devices,
                     uint32_t num_xcc) {
  std::vector<Sysfs::GpuInfo> result;
  result.reserve(devices.size());
  for (const auto &device : devices) {
    if (device.present)
      result.push_back(deviceInfoFromConfig(device, num_xcc));
  }
  return result;
}

} // namespace rocjitsu::kmd
