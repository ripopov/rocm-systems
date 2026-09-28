// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocjitsu/kmd/linux/kfd_rpc_device.h"
#include "rocjitsu/kmd/linux/kmd_command.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"

#include <cerrno>
#include <cstring>
#include <sys/mman.h>

namespace rocjitsu {

rj_status_t KfdRpcDevice::open(rj_client_pid_t pid, uint32_t *process) {
  const auto id = driver_.open_process(static_cast<pid_t>(pid));
  if (!id)
    return ROCJITSU_STATUS_ERROR;
  if (process)
    *process = id;
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t KfdRpcDevice::close(uint32_t process) {
  if (process == 0)
    driver_.close();
  else
    driver_.close(process);
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t KfdRpcDevice::gpu_id(uint32_t *id) {
  if (!id)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *id = driver_.gpu_id();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t KfdRpcDevice::gpu_info(RpcGpuInfo *info) { return gpu_info_at(0, info); }

rj_status_t KfdRpcDevice::gpu_count(uint32_t *count) {
  if (!count)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *count = driver_.num_gpus();
  return ROCJITSU_STATUS_SUCCESS;
}

bool KfdRpcDevice::strict_render_minor() const { return driver_.strict_render_minor(); }

rj_status_t KfdRpcDevice::gpu_info_at(uint32_t ordinal, RpcGpuInfo *info) {
  if (!info)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  const auto *gpu_ptr = driver_.gpu_info_at(ordinal);
  if (!gpu_ptr && ordinal == 0)
    gpu_ptr = &driver_.topology().gpu_info();
  if (!gpu_ptr)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  const auto &gpu = *gpu_ptr;
  *info = {};
  info->present = 1;
  info->gpu_id = gpu.gpu_id;
  info->gfx_target_version = gpu.gfx_target_version;
  info->vendor_id = gpu.vendor_id;
  info->device_id = gpu.device_id;
  info->family_id = gpu.family_id;
  info->unique_id = gpu.unique_id;
  info->location_id = gpu.location_id;
  info->domain = gpu.domain;
  info->hive_id = gpu.hive_id;
  info->drm_render_minor = gpu.drm_render_minor;
  info->revision_id = gpu.revision_id;
  info->pci_revision_id = gpu.pci_revision_id;
  info->simd_count = gpu.simd_count;
  info->max_waves_per_simd = gpu.max_waves_per_simd;
  info->num_shader_engines = gpu.num_shader_engines;
  info->num_shader_arrays_per_engine = gpu.num_shader_arrays_per_engine;
  info->num_cu_per_sh = gpu.num_cu_per_sh;
  info->simd_per_cu = gpu.simd_per_cu;
  info->wave_front_size = gpu.wave_front_size;
  info->num_xcc = gpu.num_xcc;
  info->max_slots_scratch_cu = gpu.max_slots_scratch_cu;
  info->local_mem_size = gpu.local_mem_size;
  info->visible_vram_size = gpu.visible_vram_size;
  info->control_stack_size = gpu.control_stack_size;
  info->context_save_size = gpu.context_save_size;
  info->vram_type = gpu.vram_type;
  info->lds_size_kb = gpu.lds_size_kb;
  info->mem_width = gpu.mem_width;
  info->mem_clk_max = gpu.mem_clk_max;
  info->l1_size_kb = gpu.l1_size_kb;
  info->l1_line_size = gpu.l1_line_size;
  info->l1_assoc = gpu.l1_assoc;
  info->l2_size_kb = gpu.l2_size_kb;
  info->l2_line_size = gpu.l2_line_size;
  info->l2_assoc = gpu.l2_assoc;
  info->num_sdma_engines = gpu.num_sdma_engines;
  info->num_sdma_xgmi_engines = gpu.num_sdma_xgmi_engines;
  info->num_cp_queues = gpu.num_cp_queues;
  info->max_engine_clk_fcompute = gpu.max_engine_clk_fcompute;
  info->capability = gpu.capability;
  info->capability2 = gpu.capability2;
  info->debug_prop = gpu.debug_prop;
  info->fw_version = gpu.fw_version;
  info->sdma_fw_version = gpu.sdma_fw_version;

  size_t name_len = gpu.marketing_name.size();
  if (name_len >= sizeof(info->marketing_name))
    name_len = sizeof(info->marketing_name) - 1;
  std::memcpy(info->marketing_name, gpu.marketing_name.data(), name_len);
  info->marketing_name[name_len] = '\0';
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t KfdRpcDevice::topology_path(const char **path) {
  if (!path)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  static thread_local std::string cached;
  cached = driver_.topology_path();
  *path = cached.c_str();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t KfdRpcDevice::drm_path(const char **path) {
  if (!path)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  static thread_local std::string cached;
  cached = driver_.topology().drm_path();
  *path = cached.c_str();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t KfdRpcDevice::map(uint32_t process, rj_vm_map_t *request) {
  if (!request)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  auto *mapped =
      driver_.mmap(process, reinterpret_cast<void *>(request->addr),
                   static_cast<size_t>(request->length), static_cast<int>(request->prot),
                   static_cast<int>(request->flags), static_cast<off_t>(request->offset));
  // Capture errno before any other call can overwrite the mapping failure.
  request->map_errno = mapped == MAP_FAILED ? errno : 0;
  request->mapped_addr = reinterpret_cast<uint64_t>(mapped);
  request->shared_handle = driver_.get_mmap_memfd(process, static_cast<off_t>(request->offset),
                                                  &request->shared_offset, &request->shared_size);
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t KfdRpcDevice::unmap(uint32_t process, rj_vm_unmap_t *request) {
  if (!request)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  driver_.munmap(process, reinterpret_cast<void *>(request->addr),
                 static_cast<size_t>(request->length));
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t KfdRpcDevice::shared_mem(uint32_t process, uint64_t offset, rj_handle_t *handle) {
  if (!handle)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *handle = driver_.get_mmap_memfd(process, static_cast<off_t>(offset));
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t KfdRpcDevice::execute(uint32_t process, rj_vm_cmd_t *request) {
  return execute_kmd_command(&driver_, process, request);
}

void KfdRpcDevice::close_all() { driver_.close_all_processes(); }

} // namespace rocjitsu
