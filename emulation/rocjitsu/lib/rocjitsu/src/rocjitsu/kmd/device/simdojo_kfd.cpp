// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/simdojo_device_backend.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/amdgpu/pm4.h"
#include "rocjitsu/vm/soc.h"

#include <cerrno>

namespace rocjitsu {
namespace {

std::vector<SimulatedKfd::BackendDevice> make_devices(const std::vector<SoC *> &socs,
                                                      const std::vector<uint32_t> &gpu_ids) {
  std::vector<SimulatedKfd::BackendDevice> devices;
  for (size_t i = 0; i < socs.size(); ++i)
    devices.push_back({i < gpu_ids.size() ? gpu_ids[i] : socs[i]->gpu_id(),
                       std::make_unique<kmd::SimdojoDeviceBackend>(*socs[i], i)});
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

int SimulatedKfd::submit_pm4(uint32_t render_minor, uint64_t queue_key,
                             amdgpu::Pm4Submission submission) {
  const auto process = find_process(local_process_id_);
  if (!process)
    return -ENODEV;
  std::lock_guard op_lock(process->op_mutex_);
  if (process->event_state_.is_closing())
    return -ENODEV;
  const uint32_t ordinal = num_gpus() == 1 ? 0 : render_minor - 128;
  if (ordinal >= gpus_.size())
    return -ENODEV;
  auto *backend = dynamic_cast<kmd::SimdojoDeviceBackend *>(gpus_[ordinal].backend.get());
  return backend ? backend->submit_pm4(local_process_id_, queue_key, std::move(submission))
                 : -ENOTSUP;
}

void SimulatedKfd::retire_pm4_queue(uint64_t queue_key) {
  for (auto &gpu : gpus_)
    if (auto *backend = dynamic_cast<kmd::SimdojoDeviceBackend *>(gpu.backend.get()))
      backend->retire_pm4_queue(queue_key);
}

} // namespace rocjitsu
