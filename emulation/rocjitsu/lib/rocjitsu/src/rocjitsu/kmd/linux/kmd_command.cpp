// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/kmd_command.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"

#include <algorithm>
#include <cerrno>

namespace rocjitsu {
namespace {

bool reconstruct_embedded_pointers(uint32_t cmd, void *arg, size_t arg_size, size_t total_size) {
  if (total_size < arg_size)
    return false;
  auto *extra = static_cast<uint8_t *>(arg) + arg_size;
  const size_t inline_size = total_size - arg_size;
  auto has_entries = [inline_size](size_t count, size_t entry_size) {
    return entry_size == 0 || count <= inline_size / entry_size;
  };
  switch (cmd) {
  case AMDKFD_IOC_SET_CU_MASK: {
    auto *args = static_cast<kfd_ioctl_set_cu_mask_args *>(arg);
    if (args->num_cu_mask == 0 || args->num_cu_mask % 32 != 0 ||
        !has_entries(std::min(args->num_cu_mask, 1024u) / 32, sizeof(uint32_t)))
      return false;
    args->cu_mask_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  }
  case AMDKFD_IOC_WAIT_EVENTS: {
    auto *args = static_cast<kfd_ioctl_wait_events_args *>(arg);
    if (!has_entries(args->num_events, sizeof(kfd_event_data)))
      return false;
    if (args->num_events > 0)
      args->events_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  }
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: {
    auto *args = static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg);
    if (!has_entries(args->n_devices, sizeof(uint32_t)))
      return false;
    if (args->n_devices > 0)
      args->device_ids_array_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  }
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: {
    auto *args = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg);
    if (!has_entries(args->num_of_nodes, sizeof(kfd_process_device_apertures)))
      return false;
    if (args->num_of_nodes > 0)
      args->kfd_process_device_apertures_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  }
  case AMDKFD_IOC_DBG_TRAP: {
    auto *args = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
    switch (args->op) {
    case KFD_IOC_DBG_TRAP_ENABLE: {
      const size_t required =
          std::min(static_cast<size_t>(args->enable.rinfo_size), sizeof(kfd_runtime_info));
      if (required > inline_size)
        return false;
      if (required > 0)
        args->enable.rinfo_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    }
    case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
      if (!has_entries(args->device_snapshot.num_devices, args->device_snapshot.entry_size))
        return false;
      if (args->device_snapshot.num_devices > 0 && args->device_snapshot.snapshot_buf_ptr != 0)
        args->device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT:
      if (!has_entries(args->queue_snapshot.num_queues, args->queue_snapshot.entry_size))
        return false;
      if (args->queue_snapshot.num_queues > 0 && args->queue_snapshot.snapshot_buf_ptr != 0)
        args->queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
      if (args->query_exception_info.info_size > inline_size)
        return false;
      if (args->query_exception_info.info_size > 0 && args->query_exception_info.info_ptr != 0)
        args->query_exception_info.info_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
      if (!has_entries(args->suspend_queues.num_queues, sizeof(uint32_t)))
        return false;
      if (args->suspend_queues.num_queues > 0 && args->suspend_queues.queue_array_ptr != 0)
        args->suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
      if (!has_entries(args->resume_queues.num_queues, sizeof(uint32_t)))
        return false;
      if (args->resume_queues.num_queues > 0 && args->resume_queues.queue_array_ptr != 0)
        args->resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(extra);
      break;
    default:
      break;
    }
    break;
  }
  default:
    break;
  }
  return true;
}

} // namespace

rj_status_t execute_kmd_command(SimulatedKfd *driver, uint32_t process_id, rj_vm_cmd_t *cmd) {
  if (!driver || !cmd || !cmd->buf)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  auto arg_size = _IOC_SIZE(cmd->cmd);
  if (!reconstruct_embedded_pointers(cmd->cmd, cmd->buf, arg_size, cmd->buf_size)) {
    cmd->result = -EINVAL;
    cmd->shared_handle = -1;
    return ROCJITSU_STATUS_SUCCESS;
  }

  // For DBG_TRAP ENABLE the debugger's notifier pipe arrives as an SCM_RIGHTS
  // fd in cmd->in_handle (already in the daemon's fd space). Substitute it for
  // the client-side fd number in the payload so the driver signals the right
  // pipe when a wave stops (the kernel receives the same fd via the ioctl). On
  // success the debug session takes ownership (cmd->in_handle cleared so the
  // transport does not close it); otherwise the transport reclaims it. Only
  // daemon mode transfers the fd; local mode passes the debugger's own fd
  // through the interposer and leaves cmd->in_handle at -1.
  //
  // The client-supplied dbg_fd is a number in the *client's* fd table and is
  // never trusted in the daemon's namespace. Overwrite it unconditionally for
  // ENABLE: with the transferred fd when one arrived via SCM_RIGHTS, otherwise
  // with KFD_INVALID_FD so the handler's fcntl() check rejects it. Leaving the
  // client's integer in place would let a client that omits the ancillary fd
  // point the session at an arbitrary daemon-owned descriptor (a confused-deputy
  // fd substitution).
  bool adopting_notifier = false;
  if (driver->daemon_mode() && cmd->cmd == AMDKFD_IOC_DBG_TRAP) {
    auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(cmd->buf);
    if (dbg->op == KFD_IOC_DBG_TRAP_ENABLE) {
      if (cmd->in_handle >= 0) {
        dbg->enable.dbg_fd = static_cast<uint32_t>(cmd->in_handle);
        adopting_notifier = true;
      } else {
        dbg->enable.dbg_fd = KFD_INVALID_FD;
      }
    }
  }

  cmd->result =
      driver->ioctl(process_id, cmd->cmd, cmd->buf, &cmd->in_mem_handle, cmd->in_proc_handle);
  cmd->shared_handle = -1;
  cmd->shared_offset = 0;
  cmd->shared_size = 0;
  if (adopting_notifier && cmd->result == 0)
    cmd->in_handle = -1;

  if (cmd->cmd == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU && cmd->result == 0) {
    auto *alloc_args = static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(cmd->buf);
    cmd->shared_handle =
        driver->get_mmap_memfd(process_id, static_cast<off_t>(alloc_args->mmap_offset),
                               &cmd->shared_offset, &cmd->shared_size);
  } else if (cmd->cmd == AMDKFD_IOC_IPC_IMPORT_HANDLE && cmd->result == 0) {
    auto *import_args = static_cast<kfd_ioctl_ipc_import_handle_args *>(cmd->buf);
    cmd->shared_handle =
        driver->get_mmap_memfd(process_id, static_cast<off_t>(import_args->mmap_offset),
                               &cmd->shared_offset, &cmd->shared_size);
  }

  return ROCJITSU_STATUS_SUCCESS;
}

} // namespace rocjitsu
