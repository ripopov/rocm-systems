// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/daemon/rj_daemon.h"
#include "rocjitsu/kmd/linux/kmd_rpc_server.h"
#include "rocjitsu/vm/rj_vm.h"

#include <atomic>
#include <cstring>
#include <memory>
#include <new>
#include <sys/un.h>
#include <system_error>
#include <thread>

namespace {

using namespace rocjitsu;

bool checked_product(size_t lhs, size_t rhs, size_t *product) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    return false;
  *product = lhs * rhs;
  return true;
}

bool validate_ioctl_payload(uint32_t command, const void *buffer, size_t buffer_size) {
  size_t argument_size = 0;
  if (buffer_size < ioctl_arg_size(command))
    return false;
  if (!validate_ioctl_arg_size(command, buffer, argument_size) || buffer_size < argument_size)
    return false;

  size_t inline_size = 0;
  switch (canonical_ioctl_request(command)) {
  case AMDKFD_IOC_SET_CU_MASK: {
    const kfd_ioctl_set_cu_mask_args *args =
        static_cast<const kfd_ioctl_set_cu_mask_args *>(buffer);
    if (args->num_cu_mask == 0 || args->num_cu_mask % 32 != 0 || args->cu_mask_ptr == 0)
      return false;
    inline_size = (std::min(args->num_cu_mask, 1024u) / 32) * sizeof(uint32_t);
    break;
  }
  case AMDKFD_IOC_WAIT_EVENTS: {
    const auto *args = static_cast<const kfd_ioctl_wait_events_args *>(buffer);
    if (!checked_product(args->num_events, sizeof(kfd_event_data), &inline_size))
      return false;
    break;
  }
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: {
    const auto *args = static_cast<const kfd_ioctl_map_memory_to_gpu_args *>(buffer);
    if (!checked_product(args->n_devices, sizeof(uint32_t), &inline_size))
      return false;
    break;
  }
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: {
    const auto *args = static_cast<const kfd_ioctl_get_process_apertures_new_args *>(buffer);
    if (!checked_product(args->num_of_nodes, sizeof(kfd_process_device_apertures), &inline_size))
      return false;
    break;
  }
  case AMDKFD_IOC_DBG_TRAP: {
    // Every op the client reserves an inline tail for has to appear here, or
    // the exact-length test below rejects a well-formed request and the
    // connection is dropped. Keep this in step with the tail sizes
    // RemoteDriver::send_ioctl() computes and the pointers
    // reconstruct_embedded_pointers() repoints (rj_vm.cpp) -- all three must
    // agree on the same set of ops.
    const auto *args = static_cast<const kfd_ioctl_dbg_trap_args *>(buffer);
    switch (args->op) {
    case KFD_IOC_DBG_TRAP_ENABLE:
      inline_size =
          std::min(static_cast<size_t>(args->enable.rinfo_size), sizeof(kfd_runtime_info));
      break;
    case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
      if (!checked_product(args->device_snapshot.num_devices, args->device_snapshot.entry_size,
                           &inline_size))
        return false;
      break;
    case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT:
      if (!checked_product(args->queue_snapshot.num_queues, args->queue_snapshot.entry_size,
                           &inline_size))
        return false;
      break;
    case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
      inline_size = args->query_exception_info.info_size;
      break;
    case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
      if (!checked_product(args->suspend_queues.num_queues, sizeof(uint32_t), &inline_size))
        return false;
      break;
    case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
      if (!checked_product(args->resume_queues.num_queues, sizeof(uint32_t), &inline_size))
        return false;
      break;
    default:
      break;
    }
    break;
  }
  default:
    break;
  }

  return inline_size <= buffer_size - argument_size && argument_size + inline_size == buffer_size;
}

bool send_with_fd_exact(int socket, const void *data, size_t size, int fd) {
  ssize_t sent = 0;
  do {
    sent = rpc_send_msg(socket, data, size, &fd, 1);
  } while (sent < 0 && errno == EINTR);
  if (sent <= 0)
    return false;
  auto bytes_sent = static_cast<size_t>(sent);
  return bytes_sent == size ||
         rpc_send_exact(socket, static_cast<const uint8_t *>(data) + bytes_sent, size - bytes_sent);
}

/// @brief The SCM_RIGHTS descriptors a client may attach to one request.
///
/// @details DBG_TRAP ENABLE sends three: the debugger's notifier, and the
/// authorization the daemon needs to reach the debuggee's address space --
/// /proc/<pid>/mem, plus the /proc/<pid> directory that pins the identity that
/// fd belongs to. Every other request sends at most the notifier. Ownership is
/// RAII, so descriptors the VM declines to adopt are released with the request
/// rather than leaked for the life of the connection.
struct ReceivedFds {
  util::UniqueHandle notifier;
  util::UniqueHandle target_mem;
  util::UniqueHandle target_proc;
};

bool receive_header_with_fds(int socket, RpcHeader *header, ReceivedFds *fds) {
  int received_fds[3] = {-1, -1, -1};
  size_t received_fd_count = std::size(received_fds);
  ssize_t header_bytes = 0;
  do {
    header_bytes = rpc_recv_msg(socket, header, sizeof(*header), received_fds, &received_fd_count);
  } while (header_bytes < 0 && errno == EINTR);
  if (received_fd_count > 0)
    fds->notifier.reset(received_fds[0]);
  if (received_fd_count > 1)
    fds->target_mem.reset(received_fds[1]);
  if (received_fd_count > 2)
    fds->target_proc.reset(received_fds[2]);
  if (header_bytes <= 0)
    return false;
  const size_t bytes_received = static_cast<size_t>(header_bytes);
  return bytes_received == sizeof(*header) ||
         rpc_recv_exact(socket, reinterpret_cast<uint8_t *>(header) + bytes_received,
                        sizeof(*header) - bytes_received);
}

rj_client_pid_t peer_pid_for_socket(int fd) {
  struct ucred credentials {};
  socklen_t length = sizeof(credentials);
  if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials, &length) == 0 &&
      length == sizeof(credentials) && credentials.pid > 0) {
    return static_cast<rj_client_pid_t>(credentials.pid);
  }
  return 0;
}

void close_client(rj_daemon_t *daemon, int client_fd) {
  std::lock_guard lock(daemon->clients_mutex);
  auto position = std::find(daemon->client_fds.begin(), daemon->client_fds.end(), client_fd);
  if (position != daemon->client_fds.end())
    daemon->client_fds.erase(position);
  ::close(client_fd);
}

void interrupt_transport(rj_daemon_t *daemon) {
  daemon->stop_requested.store(true, std::memory_order_release);
  if (daemon->listen_fd >= 0)
    ::shutdown(daemon->listen_fd, SHUT_RDWR);

  std::lock_guard lock(daemon->clients_mutex);
  for (int fd : daemon->client_fds)
    ::shutdown(fd, SHUT_RDWR);
}

void mark_error(rj_daemon_t *daemon) {
  auto state = daemon->status.load(std::memory_order_acquire);
  while (state != RJ_DAEMON_STATUS_STOPPING && state != RJ_DAEMON_STATUS_STOPPED &&
         state != RJ_DAEMON_STATUS_ERROR &&
         !daemon->status.compare_exchange_weak(state, RJ_DAEMON_STATUS_ERROR,
                                               std::memory_order_acq_rel)) {
  }
  if (state == RJ_DAEMON_STATUS_STOPPING || state == RJ_DAEMON_STATUS_STOPPED)
    return;
  interrupt_transport(daemon);
  rj_vm_request_exit(daemon->vm, "daemon transport failure");
}

enum class RequestDisposition { Continue, Disconnect };

class ClientSession {
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
  rj_status_t topology_path(const char **path) override { return rj_vm_topology_path(vm, path); }
  rj_status_t drm_path(const char **path) override { return rj_vm_drm_path(vm, path); }
  rj_status_t map(uint32_t process, rj_vm_map_t *request) override {
    return rj_vm_device_map_as(vm, process, request);
  }
  rj_status_t unmap(uint32_t process, rj_vm_unmap_t *request) override {
    return rj_vm_device_unmap_as(vm, process, request);
  }
  rj_status_t shared_mem(uint32_t process, uint64_t offset, rj_handle_t *handle) override {
    return rj_vm_get_shared_mem_as(vm, process, offset, handle);
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
