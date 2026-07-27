// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/virtual_machine.h"

#include "rocjitsu/kmd/device/simdojo_device_backend.h"
#include "rocjitsu/kmd/linux/simulated_driver.h"

#include <cassert>
#include <memory>
#include <sys/syscall.h>
#include <unistd.h>

namespace rocjitsu {

VirtualMachine::VirtualMachine(const Config &config)
    : simdojo::CompositeComponent("vm"), config_(config) {
  set_weight(0);
  auto soc = std::make_unique<SoC>("gpu_soc", config.soc);
  soc_ = soc.get();
  add_child(std::move(soc));
}

VirtualMachine::VirtualMachine(std::unique_ptr<SoC> soc, bool daemon_mode)
    : simdojo::CompositeComponent(soc->name()), soc_(soc.get()) {
  set_weight(0);
  adopt_children(*soc);
  add_child(std::move(soc));
  std::vector<std::shared_ptr<kmd::DeviceBackend>> backends;
  backends.push_back(kmd::makeSimdojoDeviceBackend(*soc_, soc_->gpu_id()));
  driver_ =
      std::make_unique<SimulatedDriver>(std::move(backends), daemon_mode);
}

VirtualMachine::VirtualMachine(std::vector<std::unique_ptr<SoC>> socs,
                               std::vector<uint32_t> gpu_ids, bool daemon_mode)
    : simdojo::CompositeComponent("vm") {
  set_weight(0);
  std::vector<SoC *> ptrs;
  ptrs.reserve(socs.size());
  for (size_t i = 0; i < socs.size(); ++i) {
    auto *p = socs[i].get();
    ptrs.push_back(p);
    if (i == 0)
      soc_ = p;
    adopt_children(*socs[i]);
    add_child(std::move(socs[i]));
  }
  std::vector<std::shared_ptr<kmd::DeviceBackend>> backends;
  backends.reserve(ptrs.size());
  for (size_t i = 0; i < ptrs.size(); ++i) {
    uint32_t gpu_id =
        i < gpu_ids.size() ? gpu_ids[i] : ptrs[i]->gpu_id();
    backends.push_back(kmd::makeSimdojoDeviceBackend(*ptrs[i], gpu_id));
  }
  driver_ =
      std::make_unique<SimulatedDriver>(std::move(backends), daemon_mode);
}

VirtualMachine::~VirtualMachine() = default;

} // namespace rocjitsu
