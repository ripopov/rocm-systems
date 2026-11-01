// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocjitsu/kmd/server.h"
#include "rocjitsu/kmd/device/callback_device_backend.h"
#include "rocjitsu/kmd/linux/kfd_rpc_device.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <poll.h>
#include <thread>

struct rj_kmd_server {
  rocjitsu::CallbackDispatcher dispatcher;
  const std::thread::id owner = std::this_thread::get_id();
  std::unique_ptr<rocjitsu::SimulatedKfd> driver;
  rocjitsu::kmd::CallbackDeviceBackend *backend = nullptr; // owned by driver
  std::unique_ptr<rocjitsu::KfdRpcDevice> device;
  std::unique_ptr<rocjitsu::KmdRpcServer> rpc;
  std::string socket_path;
  rj_kmd_diagnostic_fn diagnostic = nullptr;
  void *diagnostic_context = nullptr;

  ~rj_kmd_server() {
    // Joining RPC workers on the owner would deadlock their queued callbacks.
    if (rpc)
      dispatcher.run_while_dispatching([&] { rpc->stop(); });
    if (driver && backend->failure() && diagnostic) {
      try {
        diagnostic(diagnostic_context, backend->failure(), "embedded device teardown failed");
      } catch (...) {
      }
    }
    rpc.reset();
    device.reset();
    driver.reset(); // includes backend and its final owner-thread shutdown
  }
};

namespace {
template <typename T> bool valid(const T *value) {
  return value && value->abi_version == RJ_KMD_SERVER_ABI_VERSION &&
         value->struct_size >= sizeof(T);
}
void message(char *buffer, size_t capacity, const char *text) noexcept {
  if (!buffer || !capacity)
    return;
  const auto count = std::min(capacity - 1, std::strlen(text));
  std::memcpy(buffer, text, count);
  buffer[count] = '\0';
}
void diagnose(const rj_kmd_server *server, int status, const char *text) noexcept {
  if (server && server->diagnostic) {
    try {
      server->diagnostic(server->diagnostic_context, status, text);
    } catch (...) {
    } // Never let a host exception escape the C API.
  }
}
template <typename Function>
int guard(const rj_kmd_server *server, char *error, size_t capacity,
          Function &&operation) noexcept {
  message(error, capacity, "");
  if (server && server->owner != std::this_thread::get_id()) {
    message(error, capacity, "operation requires the creating thread");
    return -EPERM; // Do not invoke a host diagnostic callback on the wrong thread.
  }
  int result = 0;
  const char *detail = nullptr;
  try {
    result = operation();
  } catch (const std::bad_alloc &) {
    result = -ENOMEM;
  } catch (const std::invalid_argument &exception) {
    result = -EINVAL;
    message(error, capacity, exception.what());
    diagnose(server, result, exception.what());
    return result;
  } catch (const std::system_error &exception) {
    result = -EIO;
    if (exception.code().category() == std::generic_category() && exception.code().value() > 0)
      result = -exception.code().value();
    message(error, capacity, exception.what());
    diagnose(server, result, exception.what());
    return result;
  } catch (const std::exception &exception) {
    message(error, capacity, exception.what());
    diagnose(server, -EIO, exception.what());
    return -EIO;
  } catch (...) {
    result = -EIO;
  }
  if (result) {
    detail = std::strerror(-result);
    message(error, capacity, detail);
    diagnose(server, result, detail);
  }
  return result;
}

rocjitsu::Sysfs::GpuInfo topologyInfo(const rj_kmd_device_info_t &source) {
  rocjitsu::Sysfs::GpuInfo target{};
  target.gpu_id = source.gpu_id;
  target.gfx_target_version = source.gfx_target_version;
  target.vendor_id = source.vendor_id;
  target.device_id = source.device_id;
  target.family_id = source.family_id;
  target.unique_id = source.unique_id;
  target.location_id = source.location_id;
  target.domain = source.domain;
  target.hive_id = source.hive_id;
  target.drm_render_minor = source.drm_render_minor;
  target.revision_id = source.revision_id;
  target.pci_revision_id = source.pci_revision_id;
  target.simd_count = source.simd_count;
  target.max_waves_per_simd = source.max_waves_per_simd;
  target.num_shader_engines = source.num_shader_engines;
  target.num_shader_arrays_per_engine = source.num_shader_arrays_per_engine;
  target.num_cu_per_sh = source.num_cu_per_sh;
  target.simd_per_cu = source.simd_per_cu;
  target.wave_front_size = source.wave_front_size;
  target.num_xcc = source.num_xcc;
  target.max_slots_scratch_cu = source.max_slots_scratch_cu;
  target.local_mem_size = source.local_mem_size;
  // ABI v2 describes the legacy fully CPU-visible shared-backing device.
  // Debugger context areas are absent because this backend rejects debugging.
  target.visible_vram_size = source.local_mem_size;
  target.vram_type = source.vram_type;
  target.lds_size_kb = source.lds_size_kb;
  target.mem_width = source.mem_width;
  target.mem_clk_max = source.mem_clk_max;
  target.l1_size_kb = source.l1_size_kb;
  target.l1_line_size = source.l1_line_size;
  target.l1_assoc = source.l1_assoc;
  target.l2_size_kb = source.l2_size_kb;
  target.l2_line_size = source.l2_line_size;
  target.l2_assoc = source.l2_assoc;
  target.num_sdma_engines = source.num_sdma_engines;
  target.num_sdma_xgmi_engines = source.num_sdma_xgmi_engines;
  target.num_cp_queues = source.num_cp_queues;
  target.max_engine_clk_fcompute = source.max_engine_clk_fcompute;
  target.capability = source.capability;
  target.capability2 = source.capability2;
  target.debug_prop = source.debug_prop;
  target.fw_version = source.fw_version;
  target.sdma_fw_version = source.sdma_fw_version;
  auto *end = static_cast<const char *>(
      std::memchr(source.marketing_name, '\0', sizeof(source.marketing_name)));
  size_t length =
      end ? static_cast<size_t>(end - source.marketing_name) : sizeof(source.marketing_name);
  target.marketing_name.assign(source.marketing_name, length);
  return target;
}

} // namespace

extern "C" {

int32_t rj_kmd_server_create(const rj_kmd_device_info_t *info, const rj_kmd_server_config_t *config,
                             const rj_kmd_backend_ops_t *operations, rj_kmd_server_t **output,
                             char *error, size_t capacity) {
  if (output)
    *output = nullptr;
  return guard(nullptr, error, capacity, [&] {
    if (!output || !valid(info) || !valid(config) || !valid(operations) || !config->socket_path ||
        !*config->socket_path || !info->gpu_id)
      return -EINVAL;
    if (info->architecture > ROCJITSU_CODE_ARCH_CDNA5 ||
        info->architecture == ROCJITSU_CODE_ARCH_RV32I ||
        info->architecture == ROCJITSU_CODE_ARCH_RV64I)
      return -ENOTSUP;
    // XCC count describes topology, not an embedding restriction. The host
    // owns placement of each hardware queue; simdojo's per-XCC CP objects
    // are not part of the external-device contract.
    auto server = std::make_unique<rj_kmd_server>();
    server->socket_path = config->socket_path;
    server->diagnostic = config->diagnostic;
    server->diagnostic_context = config->diagnostic_context;
    auto backend = std::make_unique<rocjitsu::kmd::CallbackDeviceBackend>(
        static_cast<rj_code_arch_t>(info->architecture), info->gpu_id, *operations,
        server->dispatcher);
    server->backend = backend.get();
    std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
    devices.push_back({info->gpu_id, std::move(backend)});
    server->driver = std::make_unique<rocjitsu::SimulatedKfd>(std::move(devices), true);
    server->driver->setup_topology(topologyInfo(*info),
                                   config->runtime_root ? config->runtime_root : "");
    if (server->driver->topology().path().empty() || server->driver->topology().drm_path().empty())
      return -EIO;
    server->device = std::make_unique<rocjitsu::KfdRpcDevice>(*server->driver);
    auto *owner = server.get();
    server->rpc = std::make_unique<rocjitsu::KmdRpcServer>(*server->device,
                                                           [owner] { owner->dispatcher.wake(); });
    *output = server.release();
    return 0;
  });
}

void rj_kmd_server_destroy(rj_kmd_server_t *server) {
  if (!server)
    return;
  if (server->owner != std::this_thread::get_id()) {
    return;
  }
  // Destruction must not return with worker threads retaining host callbacks.
  // Failure of the mandatory quiescence contract is unrecoverable.
  delete server;
}

int32_t rj_kmd_server_listen(rj_kmd_server_t *server, char *error, size_t capacity) {
  return guard(server, error, capacity, [&] {
    if (!server)
      return -EINVAL;
    if (int status = server->backend->failure())
      return status;
    if (server->rpc->bind(server->socket_path.c_str()) != ROCJITSU_STATUS_SUCCESS)
      return -EIO;
    return server->rpc->start() == ROCJITSU_STATUS_SUCCESS ? 0 : -EIO;
  });
}

int32_t rj_kmd_server_stop_accepting(rj_kmd_server_t *server) {
  return guard(server, nullptr, 0, [&] {
    if (!server)
      return -EINVAL;
    server->rpc->stop_accepting();
    return 0;
  });
}

int32_t rj_kmd_server_poll_fds(const rj_kmd_server_t *server, rj_kmd_poll_fd_t *fds,
                               size_t *count) {
  return guard(server, nullptr, 0, [&] {
    if (!server || !count)
      return -EINVAL;
    if (!fds) {
      *count = 1;
      return 0;
    }
    const size_t capacity = *count;
    *count = 1;
    if (!capacity)
      return -ENOSPC;
    fds[0] = {RJ_KMD_SERVER_ABI_VERSION, sizeof(rj_kmd_poll_fd_t), server->dispatcher.fd(), POLLIN,
              0};
    return 0;
  });
}

int32_t rj_kmd_server_process_fd(rj_kmd_server_t *server, int32_t fd, int16_t revents, char *error,
                                 size_t capacity) {
  return guard(server, error, capacity, [&] {
    if (!server || fd != server->dispatcher.fd())
      return -EINVAL;
    if (revents & (POLLERR | POLLHUP | POLLNVAL))
      return -EIO;
    if (revents & POLLIN)
      server->dispatcher.dispatch();
    if (int status = server->backend->failure()) {
      server->rpc->stop_accepting();
      return status;
    }
    return server->rpc->status() == RJ_DAEMON_STATUS_ERROR ? -EIO : 0;
  });
}

int32_t rj_kmd_server_notification_fd(const rj_kmd_server_t *server) {
  if (!server)
    return -EINVAL;
  if (server->owner != std::this_thread::get_id())
    return -EPERM;
  return server->backend->doorbells().fd();
}

int32_t rj_kmd_server_set_quiescence_probe(rj_kmd_server_t *server, rj_kmd_quiescence_fn probe,
                                           void *context) {
  return guard(server, nullptr, 0, [&] {
    if (!server)
      return -EINVAL;
    server->backend->set_quiescence_probe(probe, context);
    return 0;
  });
}

int32_t rj_kmd_server_drain_doorbells(rj_kmd_server_t *server, rj_kmd_doorbell_record_t *records,
                                      size_t *count) {
  return guard(server, nullptr, 0, [&] {
    if (!server || !count)
      return -EINVAL;
    auto &monitor = server->backend->doorbells();
    if (!records) {
      *count = monitor.pending();
      return 0;
    }
    const size_t capacity = *count;
    *count = monitor.drain(records, capacity);
    return monitor.pending() ? -ENOSPC : 0;
  });
}

int32_t rj_kmd_server_post_interrupt(rj_kmd_server_t *server, uint32_t process, uint32_t event) {
  return guard(server, nullptr, 0, [&] {
    if (!server)
      return -EINVAL;
    server->backend->post_interrupt(process, event);
    return 0;
  });
}

int32_t rj_kmd_server_get_stats(const rj_kmd_server_t *server, rj_kmd_server_stats_t *stats) {
  return guard(server, nullptr, 0, [&] {
    if (!server || !valid(stats))
      return -EINVAL;
    const auto transport = server->rpc->statistics();
    stats->active_clients = transport.active_clients;
    stats->pending_responses = transport.pending_requests;
    stats->active_processes = server->backend->active_processes();
    stats->active_queues = server->backend->doorbells().active();
    stats->pending_doorbells = server->backend->doorbells().pending();
    return 0;
  });
}

} // extern "C"
