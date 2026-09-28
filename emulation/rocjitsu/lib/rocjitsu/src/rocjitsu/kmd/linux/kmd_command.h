// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCJITSU_KMD_LINUX_KMD_COMMAND_H_
#define ROCJITSU_KMD_LINUX_KMD_COMMAND_H_

#include "rocjitsu/vm/rj_vm.h"

namespace rocjitsu {
class SimulatedKfd;
/// Reconstruct the bounded RPC payload and execute the common KFD operation.
/// Descriptor adoption and returned shared backing follow the upstream VM API.
rj_status_t execute_kmd_command(SimulatedKfd *driver, uint32_t process_id, rj_vm_cmd_t *cmd);
} // namespace rocjitsu

#endif
