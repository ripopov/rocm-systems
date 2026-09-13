// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/simdojo_device_backend.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/soc.h"

namespace rocjitsu {
namespace {

std::vector<SimulatedKfd::BackendDevice> make_devices(const std::vector<SoC *> &socs,
                                                      const std::vector<uint32_t> &gpu_ids) {
  std::vector<SimulatedKfd::BackendDevice> devices;
  for (size_t i = 0; i < socs.size(); ++i)
    devices.push_back({i < gpu_ids.size() ? gpu_ids[i] : socs[i]->gpu_id(),
                       std::make_unique<kmd::SimdojoDeviceBackend>(*socs[i])});
  return devices;
}

} // namespace

SimulatedKfd::SimulatedKfd(SoC &soc, bool daemon_mode,
                           DebugIdentityValidationHook debug_identity_validation_hook)
    : SimulatedKfd(make_devices({&soc}, {0}), daemon_mode,
                   std::move(debug_identity_validation_hook)) {}

SimulatedKfd::SimulatedKfd(std::vector<SoC *> socs, std::vector<uint32_t> gpu_ids, bool daemon_mode,
                           DebugIdentityValidationHook debug_identity_validation_hook)
    : SimulatedKfd(make_devices(socs, gpu_ids), daemon_mode,
                   std::move(debug_identity_validation_hook)) {}

} // namespace rocjitsu
