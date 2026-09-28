// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm.h"

#include "embedded_schema.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/kmd/linux/kfd_rpc_device.h"
#include "rocjitsu/kmd/linux/kmd_command.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/amdgpu/partitioning.h"
#include "rocjitsu/vm/plugins/plugin_loader.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"
#include "rocjitsu/vm/topology_assembly.h"

#include "rocjitsu/base/rj_compiler.h"
#include "util/diagnostic.h"
#include "util/log.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "uapi/linux/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <vector>

using namespace rocjitsu;

namespace {

void shutdown_plugin_group(rj_vm_t *vm) {
  // Host API preconditions keep this outside the simulation-callback interval:
  // either execution has not started, or the engine workers have stopped and
  // joined. callback_mutex_ is intentionally not lifecycle synchronization.
  if (vm && vm->soc && vm->plugin_group_active.exchange(false, std::memory_order_acq_rel))
    vm->soc->plugin_group().onShutdown();
}

rj_status_t create_from_loaded(config::LoadedConfig &loaded, rj_vm_mode_t mode, rj_vm_t **handle) {
  if (!loaded.soc())
    return ROCJITSU_STATUS_ERROR;

  auto s = std::make_unique<rj_vm_t>();
  s->soc = loaded.soc();
  auto num_xcds = s->soc->num_xcds();
  std::vector<SoC *> partition_socs;
  partition_socs.reserve(loaded.extra_gpu_builds.size() + 1);
  partition_socs.push_back(s->soc);
  if (loaded.num_gpus > 1) {
    for (auto &eb : loaded.extra_gpu_builds) {
      if (auto *extra_soc = dynamic_cast<SoC *>(eb.root.get()))
        partition_socs.push_back(extra_soc);
    }
  }
  loaded.apply_cpu_dispatch_threads();
  // XCD partitions (config num_threads): run each XCD on its own engine
  // partition/thread so the XCDs execute concurrently across their separate L2s.
  const uint32_t num_threads_requested = loaded.engine_config.num_threads;
  const uint32_t num_threads_used =
      amdgpu::clamp_xcd_partition_count(partition_socs, num_threads_requested);
  if (num_threads_used != num_threads_requested)
    util::Logger::warn("num_threads clamped: requested=", num_threads_requested,
                       ", effective=", num_threads_used);
  loaded.engine_config.num_threads = num_threads_used;
  bool serve = (mode == RJ_VM_MODE_LOCAL || mode == RJ_VM_MODE_DAEMON);
  bool daemon = (mode == RJ_VM_MODE_DAEMON);
  if (serve) {
    loaded.engine_config.max_ticks = 0;
    loaded.engine_config.await_primaries = true;
  }

  s->engine_config = loaded.engine_config;
  s->engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);

  s->vm = assemble_virtual_machine_topology(loaded, s->engine->topology(), daemon);
  if (num_threads_used > 1 && !amdgpu::partition_topology_by_xcds(
                                  s->engine->topology(), partition_socs, num_threads_used)) {
    throw std::invalid_argument("multi-threaded VM requires at least one XCD");
  }
  s->engine->create();

  // dispatch_wf() cannot enqueue work while a restored component tree is
  // detached from an engine. Once create() has assigned partitions and event
  // queues, resume any resident waves reconstructed from a checkpoint. The
  // schedule_work() guards make this a no-op for ordinary idle configurations.
  for (uint32_t i = 0; i < s->vm->num_socs(); ++i) {
    for (auto *cu : s->vm->soc(i)->all_cus()) {
      if (!cu->is_idle())
        cu->schedule_work();
    }
  }

  if (serve) {
    s->engine->register_as_primary();
    // Legacy replicated-GPU configs keep their existing XGMI discovery
    // metadata for compatibility. Manifest routes describe topology only;
    // until peer execution is granted, they must not advertise KFD XGMI
    // links, hive IDs, or XGMI SDMA engines to ROCr.
    if (!loaded.devices.empty())
      s->vm->driver()->setup_topology(loaded.devices, num_xcds,
                                      !loaded.system_graph.explicit_manifest,
                                      loaded.system_graph.explicit_manifest, loaded.pci);
    else
      s->vm->driver()->setup_topology(loaded.device, num_xcds, loaded.pci);
    s->vm->driver()->open();
  }

  s->loaded = std::move(loaded);
  *handle = s.release();
  return ROCJITSU_STATUS_SUCCESS;
}

} // namespace

rj_status_t rj_vm_create(const char *json_path, rj_vm_mode_t mode, rj_vm_t **vm) {
  return rj_vm_create_for_node(json_path, nullptr, mode, vm);
}

rj_status_t rj_vm_create_for_node(const char *json_path, const char *node_id, rj_vm_mode_t mode,
                                  rj_vm_t **vm) {
  if (!json_path || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    util::StringDiagnostic diagnostic;
    auto loaded = config::load_config_for_node(json_path, rocjitsu::kEmbeddedSchema,
                                               node_id ? node_id : "", diagnostic.emitter());
    if (loaded.failed()) {
      util::Logger::warn("rj_vm_create_for_node failed: ", diagnostic.message());
      return ROCJITSU_STATUS_INVALID_FILE;
    }
    return create_from_loaded(loaded.value(), mode, vm);
  } catch (const std::exception &e) {
    util::Logger::warn("rj_vm_create_for_node failed: ", e.what());
    return ROCJITSU_STATUS_INVALID_FILE;
  }
}

rj_status_t rj_vm_create_from_string(const char *json, rj_vm_mode_t mode, rj_vm_t **vm) {
  return rj_vm_create_from_string_for_node(json, nullptr, mode, vm);
}

rj_status_t rj_vm_create_from_string_for_node(const char *json, const char *node_id,
                                              rj_vm_mode_t mode, rj_vm_t **vm) {
  if (!json || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    util::StringDiagnostic diagnostic;
    auto loaded = config::load_config_from_string_for_node(
        json, rocjitsu::kEmbeddedSchema, node_id ? node_id : "", diagnostic.emitter());
    if (loaded.failed()) {
      util::Logger::warn("rj_vm_create_from_string_for_node failed: ", diagnostic.message());
      return ROCJITSU_STATUS_INVALID_ARGUMENT;
    }
    return create_from_loaded(loaded.value(), mode, vm);
  } catch (const std::exception &e) {
    util::Logger::warn("rj_vm_create_from_string_for_node failed: ", e.what());
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  }
}

rj_status_t rj_vm_load_plugins(rj_vm_t *vm, const char *config_json, const char *plugin_dir) {
  if (!vm || !config_json)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (!vm->soc)
    return ROCJITSU_STATUS_ERROR;
  try {
    auto group = PluginLoader::configure_plugin_group(config_json, plugin_dir ? plugin_dir : "");
    // rj_vm_load_plugins() is a pre-run operation, so replacing and initializing
    // the group cannot overlap simulation callbacks.
    shutdown_plugin_group(vm);
    vm->soc->set_plugin_group(group);
    group->onInit();
    vm->plugin_group_active.store(true, std::memory_order_release);
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::exception &) {
    return ROCJITSU_STATUS_ERROR;
  }
}

void rj_vm_retain(rj_vm_t *vm) {
  if (vm)
    vm->retain();
}

void rj_vm_release(rj_vm_t *vm) {
  if (!vm)
    return;
  if (vm->release())
    delete vm;
}

void rj_vm_destroy(rj_vm_t *vm) {
  if (!vm)
    return;
  // The API requires an asynchronous host to stop and join rj_vm_run() first.
  shutdown_plugin_group(vm);
  if (vm->destroy())
    delete vm;
}

rj_status_t rj_vm_step(rj_vm_t *vm, int *active) {
  if (!vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (!vm->soc)
    return ROCJITSU_STATUS_ERROR;
  if (vm->engine_config.num_threads != 1)
    return ROCJITSU_STATUS_UNSUPPORTED;

  bool any_active = vm->engine->step();
  if (vm->engine->exit_code() != 0)
    return ROCJITSU_STATUS_ERROR;
  if (active)
    *active = any_active ? 1 : 0;
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_run(rj_vm_t *vm, uint64_t *ticks_executed) {
  if (!vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (!vm->soc)
    return ROCJITSU_STATUS_ERROR;

  auto exit = vm->engine->run();

  if (ticks_executed)
    *ticks_executed = exit.tick;

  // shutdown() joins all engine workers before plugin state is torn down.
  vm->engine->shutdown();
  shutdown_plugin_group(vm);
  return (exit.code == 0) ? ROCJITSU_STATUS_SUCCESS : ROCJITSU_STATUS_ERROR;
}

void rj_vm_request_exit(rj_vm_t *vm, const char *reason) {
  if (!vm || !vm->engine)
    return;
  vm->engine->request_exit(reason ? reason : "shutdown");
}

rj_status_t rj_vm_save_checkpoint(const rj_vm_t *vm, const char *path, uint64_t tick) {
  if (!vm || !path)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (!vm->soc)
    return ROCJITSU_STATUS_ERROR;
  // The checkpoint format and restore path contain one SoC. Reject before
  // opening the output file rather than silently dropping other GPU state.
  if (vm->vm && vm->vm->num_socs() > 1)
    return ROCJITSU_STATUS_UNSUPPORTED;
  try {
    auto engine_config = vm->engine_config;
    if (engine_config.num_threads == vm->loaded.execution_threads.engines)
      engine_config.num_threads = vm->loaded.requested_engine_threads;
    config::save_checkpoint(path, *vm->soc, tick, engine_config, vm->loaded.cpu_dispatch_threads,
                            vm->loaded.cpu_thread_budget, vm->loaded.thread_allocations,
                            vm->loaded.legacy_auto_dispatch, vm->loaded.async_helper_threads);
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::exception &) {
    return ROCJITSU_STATUS_ERROR;
  }
}

rj_status_t rj_vm_restore_checkpoint(const char *path, rj_vm_t **vm) {
  if (!path || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    auto loaded = config::restore_checkpoint(path);
    return create_from_loaded(loaded, RJ_VM_MODE_DEFAULT, vm);
  } catch (const std::exception &) {
    return ROCJITSU_STATUS_INVALID_FILE;
  }
}

rj_status_t rj_vm_execute(rj_vm_t *vm, rj_vm_cmd_t *cmd) {
  if (!vm || !cmd || !cmd->buf || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  auto *driver = vm->vm->driver();
  return execute_kmd_command(driver, driver->local_process_id(), cmd);
}

rj_status_t rj_vm_execute_as(rj_vm_t *vm, uint32_t process_id, rj_vm_cmd_t *cmd) {
  if (!vm || !cmd || !cmd->buf || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return execute_kmd_command(vm->vm->driver(), process_id, cmd);
}

rj_status_t rj_vm_device_open(rj_vm_t *vm, rj_client_pid_t client_pid, uint32_t *process_id) {
  if (!vm || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return KfdRpcDevice(*vm->vm->driver()).open(client_pid, process_id);
}

rj_status_t rj_vm_device_close(rj_vm_t *vm, uint32_t process_id) {
  if (!vm || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return KfdRpcDevice(*vm->vm->driver()).close(process_id);
}

rj_status_t rj_vm_close_all_devices(rj_vm_t *vm) {
  if (!vm || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  KfdRpcDevice(*vm->vm->driver()).close_all();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_device_map(rj_vm_t *vm, rj_vm_map_t *map) {
  if (!vm || !map || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  auto *driver = vm->vm->driver();
  auto *result = driver->mmap(reinterpret_cast<void *>(map->addr), static_cast<size_t>(map->length),
                              static_cast<int>(map->prot), static_cast<int>(map->flags),
                              static_cast<off_t>(map->offset));
  map->mapped_addr = reinterpret_cast<uint64_t>(result);
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_device_map_as(rj_vm_t *vm, uint32_t process_id, rj_vm_map_t *map) {
  if (!vm || !map || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return KfdRpcDevice(*vm->vm->driver()).map(process_id, map);
}

rj_status_t rj_vm_device_unmap(rj_vm_t *vm, rj_vm_unmap_t *unmap) {
  if (!vm || !unmap || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  vm->vm->driver()->munmap(reinterpret_cast<void *>(unmap->addr),
                           static_cast<size_t>(unmap->length));
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_device_unmap_as(rj_vm_t *vm, uint32_t process_id, rj_vm_unmap_t *unmap) {
  if (!vm || !unmap || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return KfdRpcDevice(*vm->vm->driver()).unmap(process_id, unmap);
}

rj_status_t rj_vm_gpu_id(rj_vm_t *vm, uint32_t *gpu_id) {
  if (!vm || !gpu_id || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return KfdRpcDevice(*vm->vm->driver()).gpu_id(gpu_id);
}

rj_status_t rj_vm_topology_path(rj_vm_t *vm, const char **path) {
  if (!vm || !path || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return KfdRpcDevice(*vm->vm->driver()).topology_path(path);
}

rj_status_t rj_vm_drm_path(rj_vm_t *vm, const char **path) {
  if (!vm || !path || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return KfdRpcDevice(*vm->vm->driver()).drm_path(path);
}

rj_status_t rj_vm_gpu_info(rj_vm_t *vm, rj_vm_gpu_info_t *info) {
  return rj_vm_gpu_info_at(vm, 0, info);
}

rj_status_t rj_vm_gpu_count(rj_vm_t *vm, uint32_t *count) {
  if (!vm || !count || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *count = vm->vm->driver()->num_gpus();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_gpu_info_at(rj_vm_t *vm, uint32_t ordinal, rj_vm_gpu_info_t *info) {
  if (!vm || !info || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return KfdRpcDevice(*vm->vm->driver()).gpu_info_at(ordinal, info);
}

rj_status_t rj_vm_get_shared_mem(rj_vm_t *vm, int64_t offset, rj_handle_t *handle) {
  if (!vm || !handle || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *handle = vm->vm->driver()->get_mmap_memfd(static_cast<off_t>(offset));
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_get_shared_mem_as(rj_vm_t *vm, uint32_t process_id, int64_t offset,
                                    rj_handle_t *handle) {
  if (!vm || !handle || !vm->vm || !vm->vm->driver())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return KfdRpcDevice(*vm->vm->driver())
      .shared_mem(process_id, static_cast<uint64_t>(offset), handle);
}
