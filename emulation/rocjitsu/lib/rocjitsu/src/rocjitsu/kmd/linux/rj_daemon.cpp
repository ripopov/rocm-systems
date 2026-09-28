// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/daemon/rj_daemon.h"
#include "rocjitsu/kmd/linux/kmd_rpc_server.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"

#include <atomic>
#include <cstring>
#include <memory>
#include <new>
#include <sys/un.h>
#include <system_error>
#include <thread>

namespace {

class VmRpcDevice final : public rocjitsu::KmdRpcDevice {
public:
  rj_vm_t *vm = nullptr;
  ~VmRpcDevice() override {
    if (vm)
      rj_vm_destroy(vm);
  }
  rj_status_t open(rj_client_pid_t pid, uint32_t *process) override {
    return rj_vm_device_open(vm, pid, process);
  }
  rj_status_t close(uint32_t process) override { return rj_vm_device_close(vm, process); }
  rj_status_t gpu_id(uint32_t *id) override { return rj_vm_gpu_id(vm, id); }
  rj_status_t gpu_info(rocjitsu::RpcGpuInfo *info) override { return rj_vm_gpu_info(vm, info); }
  rj_status_t gpu_count(uint32_t *count) override { return rj_vm_gpu_count(vm, count); }
  rj_status_t gpu_info_at(uint32_t ordinal, rocjitsu::RpcGpuInfo *info) override {
    return rj_vm_gpu_info_at(vm, ordinal, info);
  }
  bool strict_render_minor() const override { return vm->vm->driver()->strict_render_minor(); }
  rj_status_t topology_path(const char **path) override { return rj_vm_topology_path(vm, path); }
  rj_status_t drm_path(const char **path) override { return rj_vm_drm_path(vm, path); }
  rj_status_t map(uint32_t process, rj_vm_map_t *request) override {
    return rj_vm_device_map_as(vm, process, request);
  }
  rj_status_t unmap(uint32_t process, rj_vm_unmap_t *request) override {
    return rj_vm_device_unmap_as(vm, process, request);
  }
  rj_status_t execute(uint32_t process, rj_vm_cmd_t *request) override {
    return rj_vm_execute_as(vm, process, request);
  }
  void close_all() override {
    if (vm)
      rj_vm_close_all_devices(vm);
  }
  void transport_failed() override {
    close_all();
    if (vm)
      rj_vm_request_exit(vm, "daemon transport failure");
  }
};

} // namespace

struct rj_daemon_t {
  VmRpcDevice device;
  rocjitsu::KmdRpcServer server{device};
  std::atomic<bool> stopping{false};
  std::jthread engine;

  ~rj_daemon_t() {
    stopping.store(true, std::memory_order_release);
    server.stop();
    if (device.vm)
      rj_vm_request_exit(device.vm, "daemon shutdown");
    engine.request_stop();
    if (engine.joinable())
      engine.join();
    // Member destruction releases the transport before destroying its VM.
  }
};

rj_status_t rj_daemon_start(const char *json, const char *socket_path, rj_daemon_t **daemon) {
  if (daemon)
    *daemon = nullptr;
  if (!json || !socket_path || !daemon)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  const size_t length = std::strlen(socket_path);
  if (length == 0 || length >= sizeof(sockaddr_un::sun_path))
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  try {
    auto created = std::make_unique<rj_daemon_t>();
    auto status = rj_vm_create_from_string(json, RJ_VM_MODE_DAEMON, &created->device.vm);
    if (status != ROCJITSU_STATUS_SUCCESS)
      return status;
    status = rj_vm_load_plugins(created->device.vm, json, nullptr);
    if (status != ROCJITSU_STATUS_SUCCESS)
      return status;
    status = created->server.bind(socket_path);
    if (status != ROCJITSU_STATUS_SUCCESS)
      return status;

    created->engine = std::jthread([state = created.get()] {
      const auto result = rj_vm_run(state->device.vm, nullptr);
      if (result != ROCJITSU_STATUS_SUCCESS || !state->stopping.load(std::memory_order_acquire))
        state->server.fail();
    });
    status = created->server.start();
    if (status != ROCJITSU_STATUS_SUCCESS)
      return status;
    *daemon = created.release();
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::bad_alloc &) {
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (const std::system_error &) {
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (...) {
    return ROCJITSU_STATUS_ERROR;
  }
}

rj_status_t rj_daemon_stop(rj_daemon_t *daemon) {
  delete daemon;
  return ROCJITSU_STATUS_SUCCESS;
}

rj_daemon_status_t rj_daemon_status(const rj_daemon_t *daemon) {
  return daemon ? daemon->server.status() : RJ_DAEMON_STATUS_STOPPED;
}
