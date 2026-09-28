// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCJITSU_KMD_LINUX_KFD_RPC_DEVICE_H_
#define ROCJITSU_KMD_LINUX_KFD_RPC_DEVICE_H_

#include "rocjitsu/kmd/linux/kmd_rpc_server.h"

namespace rocjitsu {
class SimulatedKfd;

/// Adapt the shared KFD implementation to RPC without owning an execution VM.
/// The driver outlives this adapter and every server using it.
class KfdRpcDevice final : public KmdRpcDevice {
public:
  explicit KfdRpcDevice(SimulatedKfd &driver) : driver_(driver) {}
  rj_status_t open(rj_client_pid_t pid, uint32_t *process) override;
  rj_status_t close(uint32_t process) override;
  rj_status_t gpu_id(uint32_t *id) override;
  rj_status_t gpu_info(RpcGpuInfo *info) override;
  rj_status_t gpu_count(uint32_t *count) override;
  rj_status_t gpu_info_at(uint32_t ordinal, RpcGpuInfo *info) override;
  bool strict_render_minor() const override;
  rj_status_t topology_path(const char **path) override;
  rj_status_t drm_path(const char **path) override;
  rj_status_t map(uint32_t process, rj_vm_map_t *request) override;
  rj_status_t unmap(uint32_t process, rj_vm_unmap_t *request) override;
  rj_status_t shared_mem(uint32_t process, uint64_t offset, rj_handle_t *handle);
  rj_status_t execute(uint32_t process, rj_vm_cmd_t *request) override;
  void close_all() override;

private:
  SimulatedKfd &driver_;
};
} // namespace rocjitsu

#endif
