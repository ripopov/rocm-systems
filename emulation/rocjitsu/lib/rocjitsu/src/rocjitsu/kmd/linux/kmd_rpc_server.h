// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_KMD_RPC_SERVER_H_
#define ROCJITSU_KMD_LINUX_KMD_RPC_SERVER_H_

#include "rocjitsu/daemon/rj_daemon.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/vm/rj_vm.h"

#include <functional>
#include <memory>

namespace rocjitsu {

struct KmdRpcState;

struct KmdRpcStatistics {
  uint64_t active_clients = 0;
  uint64_t pending_requests = 0;
};

/// @brief Process operations served by the shared KFD RPC transport.
/// @details The owner keeps this object alive until the server has stopped.
/// Calls may run concurrently for different client connections. close_all()
/// must wake blocked device operations so transport shutdown can join clients.
class KmdRpcDevice {
public:
  virtual ~KmdRpcDevice() = default;
  virtual rj_status_t open(rj_client_pid_t pid, uint32_t *process) = 0;
  virtual rj_status_t close(uint32_t process) = 0;
  virtual rj_status_t gpu_id(uint32_t *id) = 0;
  virtual rj_status_t gpu_info(RpcGpuInfo *info) = 0;
  virtual rj_status_t gpu_count(uint32_t *count) {
    if (!count)
      return ROCJITSU_STATUS_INVALID_ARGUMENT;
    *count = 1;
    return ROCJITSU_STATUS_SUCCESS;
  }
  virtual rj_status_t gpu_info_at(uint32_t ordinal, RpcGpuInfo *info) {
    return ordinal == 0 ? gpu_info(info) : ROCJITSU_STATUS_INVALID_ARGUMENT;
  }
  virtual bool strict_render_minor() const { return false; }
  virtual rj_status_t topology_path(const char **path) = 0;
  virtual rj_status_t drm_path(const char **path) = 0;
  virtual rj_status_t map(uint32_t process, rj_vm_map_t *request) = 0;
  virtual rj_status_t unmap(uint32_t process, rj_vm_unmap_t *request) = 0;
  virtual rj_status_t execute(uint32_t process, rj_vm_cmd_t *request) = 0;
  virtual void close_all() = 0;
  virtual void transport_failed() { close_all(); }
};

/// @brief Unix-socket KFD RPC server shared by simulated and embedded devices.
/// @details bind() reserves the endpoint before the owner starts its device.
/// start() publishes it to clients. stop() drains clients before returning;
/// only then may the owner destroy the device. Lifecycle calls are serialized
/// by the owner; fail() may be called by its engine thread.
class KmdRpcServer {
public:
  /// state_changed is a nonblocking, nonthrowing wakeup only. It can run on
  /// any transport thread and must not call back into the server or device.
  explicit KmdRpcServer(KmdRpcDevice &device, std::function<void()> state_changed = {});
  ~KmdRpcServer();
  KmdRpcServer(const KmdRpcServer &) = delete;
  KmdRpcServer &operator=(const KmdRpcServer &) = delete;
  rj_status_t bind(const char *path);
  rj_status_t start();
  /// Stop admitting connections without interrupting existing sessions. This
  /// is irreversible and nonblocking; stop() still drains and joins clients.
  void stop_accepting();
  void stop();
  void fail();
  rj_daemon_status_t status() const;
  /// Consistent transport-only snapshot, never acquires device/KFD locks.
  KmdRpcStatistics statistics() const;

private:
  std::unique_ptr<KmdRpcState> state_;
};

} // namespace rocjitsu

#endif
