// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/server.h"

#include "rocjitsu/kmd/device/gem5_device_backend.h"
#include "rocjitsu/kmd/linux/kmd_rpc_server.h"
#include "rocjitsu/kmd/linux/simulated_driver.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

struct rj_kmd_server {
  std::shared_ptr<rocjitsu::kmd::Gem5DeviceBackend> backend;
  std::unique_ptr<rocjitsu::SimulatedDriver> driver;
  std::unique_ptr<rocjitsu::KmdRpcServer> rpc;
  std::string socket_path;
  rj_kmd_diagnostic_fn diagnostic = nullptr;
  void *diagnostic_context = nullptr;
};

namespace {

template <typename T> bool validStruct(const T *value) {
  return value && value->abi_version == RJ_KMD_SERVER_ABI_VERSION &&
         value->struct_size >= sizeof(T);
}

void copyError(char *target, size_t target_size, const std::string &message) {
  if (!target || target_size == 0)
    return;
  size_t length = std::min(target_size - 1, message.size());
  std::memcpy(target, message.data(), length);
  target[length] = '\0';
}

void diagnose(rj_kmd_server_t *server, int32_t status,
              const std::string &message) {
  if (server && server->diagnostic)
    server->diagnostic(server->diagnostic_context, status, message.c_str());
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
  target.num_shader_arrays_per_engine =
      source.num_shader_arrays_per_engine;
  target.num_cu_per_sh = source.num_cu_per_sh;
  target.simd_per_cu = source.simd_per_cu;
  target.wave_front_size = source.wave_front_size;
  target.num_xcc = source.num_xcc;
  target.max_slots_scratch_cu = source.max_slots_scratch_cu;
  target.local_mem_size = source.local_mem_size;
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
      std::memchr(source.marketing_name, '\0',
                  sizeof(source.marketing_name)));
  size_t length =
      end ? static_cast<size_t>(end - source.marketing_name)
          : sizeof(source.marketing_name);
  target.marketing_name.assign(source.marketing_name, length);
  return target;
}

rocjitsu::RpcGpuInfo rpcInfo(const rj_kmd_device_info_t &source) {
  rocjitsu::RpcGpuInfo target{};
  target.present = 1;
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
  target.num_shader_arrays_per_engine =
      source.num_shader_arrays_per_engine;
  target.num_cu_per_sh = source.num_cu_per_sh;
  target.simd_per_cu = source.simd_per_cu;
  target.wave_front_size = source.wave_front_size;
  target.num_xcc = source.num_xcc;
  target.max_slots_scratch_cu = source.max_slots_scratch_cu;
  target.local_mem_size = source.local_mem_size;
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
  std::memcpy(target.marketing_name, source.marketing_name,
              sizeof(target.marketing_name));
  target.marketing_name[sizeof(target.marketing_name) - 1] = '\0';
  return target;
}

bool validBackend(const rj_kmd_backend_ops_t &backend) {
  return backend.open_process && backend.close_process && backend.allocate &&
         backend.free && backend.map && backend.unmap &&
         backend.set_apertures && backend.create_queue &&
         backend.destroy_queue;
}

} // namespace

extern "C" {

int32_t rj_kmd_server_create(const rj_kmd_device_info_t *device,
                             const rj_kmd_server_config_t *config,
                             const rj_kmd_backend_ops_t *backend,
                             rj_kmd_server_t **server, char *error,
                             size_t error_size) {
  if (server)
    *server = nullptr;
  if (!validStruct(device) || !validStruct(config) ||
      !validStruct(backend) || !server || !config->socket_path ||
      config->socket_path[0] == '\0' || !validBackend(*backend)) {
    copyError(error, error_size, "invalid KMD server arguments or ABI");
    return RJ_KMD_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto result = std::make_unique<rj_kmd_server_t>();
    result->socket_path = config->socket_path;
    result->diagnostic = config->diagnostic;
    result->diagnostic_context = config->diagnostic_context;

    rocjitsu::kmd::DeviceInfo backend_info{
        .gpu_id = device->gpu_id,
        .architecture =
            static_cast<rj_code_arch_t>(device->architecture),
    };
    result->backend = std::make_shared<rocjitsu::kmd::Gem5DeviceBackend>(
        backend_info, *backend);
    std::vector<std::shared_ptr<rocjitsu::kmd::DeviceBackend>> backends{
        result->backend};
    result->driver =
        std::make_unique<rocjitsu::SimulatedDriver>(std::move(backends), true);
    if (config->runtime_root && config->runtime_root[0] != '\0')
      result->driver->set_runtime_root(config->runtime_root);
    result->driver->setup_topology(topologyInfo(*device));
    if (result->driver->topology_path().empty()) {
      copyError(error, error_size, "failed to create KFD topology");
      return RJ_KMD_STATUS_IO_ERROR;
    }
    result->rpc = std::make_unique<rocjitsu::KmdRpcServer>(
        *result->driver, rpcInfo(*device));
    *server = result.release();
    return RJ_KMD_STATUS_SUCCESS;
  } catch (const std::bad_alloc &) {
    copyError(error, error_size, "out of memory creating KMD server");
    return RJ_KMD_STATUS_NO_MEMORY;
  } catch (const std::exception &exception) {
    copyError(error, error_size, exception.what());
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  } catch (...) {
    copyError(error, error_size, "unknown KMD server creation failure");
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  }
}

void rj_kmd_server_destroy(rj_kmd_server_t *server) {
  try {
    if (server && server->rpc)
      server->rpc->close();
    delete server;
  } catch (...) {
  }
}

int32_t rj_kmd_server_listen(rj_kmd_server_t *server, char *error,
                             size_t error_size) {
  if (!server || !server->rpc)
    return RJ_KMD_STATUS_INVALID_ARGUMENT;
  try {
    std::string message;
    if (!server->rpc->listen(server->socket_path, &message)) {
      copyError(error, error_size, message);
      diagnose(server, RJ_KMD_STATUS_IO_ERROR, message);
      return RJ_KMD_STATUS_IO_ERROR;
    }
    return RJ_KMD_STATUS_SUCCESS;
  } catch (const std::exception &exception) {
    copyError(error, error_size, exception.what());
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  } catch (...) {
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  }
}

int32_t rj_kmd_server_stop_accepting(rj_kmd_server_t *server) {
  if (!server || !server->rpc)
    return RJ_KMD_STATUS_INVALID_ARGUMENT;
  try {
    server->rpc->stopAccepting();
    return RJ_KMD_STATUS_SUCCESS;
  } catch (...) {
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  }
}

int32_t rj_kmd_server_poll_fds(const rj_kmd_server_t *server,
                               rj_kmd_poll_fd_t *fds, size_t *count) {
  if (!server || !server->rpc || !count)
    return RJ_KMD_STATUS_INVALID_ARGUMENT;
  try {
    auto descriptors = server->rpc->pollDescriptors();
    size_t needed = descriptors.size();
    if (!fds) {
      *count = needed;
      return RJ_KMD_STATUS_SUCCESS;
    }
    size_t capacity = *count;
    *count = needed;
    if (capacity < needed)
      return -ENOSPC;
    for (size_t index = 0; index < needed; ++index) {
      fds[index] = {
          .abi_version = RJ_KMD_SERVER_ABI_VERSION,
          .struct_size = sizeof(rj_kmd_poll_fd_t),
          .fd = descriptors[index].fd,
          .events = descriptors[index].events,
          .reserved = 0,
      };
    }
    return RJ_KMD_STATUS_SUCCESS;
  } catch (...) {
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  }
}

int32_t rj_kmd_server_process_fd(rj_kmd_server_t *server, int32_t fd,
                                 int16_t revents, char *error,
                                 size_t error_size) {
  if (!server || !server->rpc)
    return RJ_KMD_STATUS_INVALID_ARGUMENT;
  try {
    std::string message;
    if (!server->rpc->processFd(fd, revents, &message)) {
      copyError(error, error_size, message);
      diagnose(server, RJ_KMD_STATUS_IO_ERROR, message);
      return RJ_KMD_STATUS_IO_ERROR;
    }
    return RJ_KMD_STATUS_SUCCESS;
  } catch (const std::exception &exception) {
    copyError(error, error_size, exception.what());
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  } catch (...) {
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  }
}

int32_t rj_kmd_server_notification_fd(const rj_kmd_server_t *server) {
  if (!server || !server->backend)
    return RJ_KMD_STATUS_INVALID_ARGUMENT;
  return server->backend->notificationFd();
}

int32_t rj_kmd_server_drain_doorbells(
    rj_kmd_server_t *server, rj_kmd_doorbell_record_t *records,
    size_t *count) {
  if (!server || !server->backend || !count)
    return RJ_KMD_STATUS_INVALID_ARGUMENT;
  try {
    size_t pending = server->backend->pendingDoorbells();
    if (!records) {
      *count = pending;
      return RJ_KMD_STATUS_SUCCESS;
    }
    size_t capacity = *count;
    *count = server->backend->drainDoorbells(records, capacity);
    return capacity < pending ? -ENOSPC : RJ_KMD_STATUS_SUCCESS;
  } catch (...) {
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  }
}

int32_t rj_kmd_server_post_interrupt(rj_kmd_server_t *server,
                                     uint32_t process_id,
                                     uint32_t event_id) {
  if (!server || !server->backend)
    return RJ_KMD_STATUS_INVALID_ARGUMENT;
  try {
    server->backend->postInterrupt(process_id, event_id);
    return RJ_KMD_STATUS_SUCCESS;
  } catch (...) {
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  }
}

int32_t rj_kmd_server_get_stats(const rj_kmd_server_t *server,
                                rj_kmd_server_stats_t *stats) {
  if (!server || !server->rpc || !server->backend || !validStruct(stats))
    return RJ_KMD_STATUS_INVALID_ARGUMENT;
  try {
    stats->active_clients = server->rpc->activeClients();
    stats->active_processes = server->rpc->activeProcesses();
    stats->active_queues = server->backend->activeQueues();
    stats->pending_responses = server->rpc->pendingResponses();
    stats->pending_doorbells = server->backend->pendingDoorbells();
    return RJ_KMD_STATUS_SUCCESS;
  } catch (...) {
    return RJ_KMD_STATUS_INTERNAL_ERROR;
  }
}

} // extern "C"
