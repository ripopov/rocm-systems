// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/daemon/rj_daemon.h"
#include "rocjitsu/kmd/linux/kfd_rpc_device.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"

#include <atomic>
#include <cstring>
#include <memory>
#include <new>
#include <sys/un.h>
#include <system_error>
#include <thread>
#include <utility>

namespace {

using VmHandle = std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)>;

} // namespace

class rj_daemon_t {
public:
  explicit rj_daemon_t(VmHandle vm)
      : vm_(std::move(vm)), device_(*vm_->vm->driver()), server_(device_, {}, [this] {
          rj_vm_request_exit(vm_.get(), "daemon transport failure");
        }) {}

  ~rj_daemon_t() {
    stopping_.store(true, std::memory_order_release);
    server_.stop();
    rj_vm_request_exit(vm_.get(), "daemon shutdown");
    engine_.request_stop();
    if (engine_.joinable())
      engine_.join();
    // Members release the transport and borrowed driver before destroying the VM.
  }

  rj_status_t start(const char *socket_path) {
    const rj_status_t status = server_.bind(socket_path);
    if (status != ROCJITSU_STATUS_SUCCESS)
      return status;
    engine_ = std::jthread([this] {
      const rj_status_t result = rj_vm_run(vm_.get(), nullptr);
      if (result != ROCJITSU_STATUS_SUCCESS || !stopping_.load(std::memory_order_acquire))
        server_.fail();
    });
    return server_.start();
  }

  rj_daemon_status_t status() const { return server_.status(); }

private:
  VmHandle vm_;
  rocjitsu::KfdRpcDevice device_;
  rocjitsu::KmdRpcServer server_;
  std::atomic<bool> stopping_{false};
  std::jthread engine_;
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
    rj_vm_t *raw_vm = nullptr;
    rj_status_t status = rj_vm_create_from_string(json, RJ_VM_MODE_DAEMON, &raw_vm);
    VmHandle vm(raw_vm, rj_vm_destroy);
    if (status != ROCJITSU_STATUS_SUCCESS)
      return status;
    status = rj_vm_load_plugins(vm.get(), json, nullptr);
    if (status != ROCJITSU_STATUS_SUCCESS)
      return status;
    auto created = std::make_unique<rj_daemon_t>(std::move(vm));
    status = created->start(socket_path);
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
  return daemon ? daemon->status() : RJ_DAEMON_STATUS_STOPPED;
}
