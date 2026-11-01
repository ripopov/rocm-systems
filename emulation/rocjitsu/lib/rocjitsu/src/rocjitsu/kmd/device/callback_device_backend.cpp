// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/callback_device_backend.h"
#include "rocjitsu/kmd/linux/kfd_process.h"

namespace rocjitsu::kmd {

CallbackDeviceBackend::CallbackDeviceBackend(rj_code_arch_t architecture, uint32_t gpu_id,
                                             const rj_kmd_backend_ops_t &operations,
                                             CallbackDispatcher &dispatcher)
    : architecture_(architecture), operations_(operations), dispatcher_(dispatcher),
      memory_(gpu_id, operations, dispatcher) {
  if (!operations.open_process || !operations.close_process || !operations.set_process_client_pid ||
      !operations.set_apertures || !operations.create_queue || !operations.update_queue ||
      !operations.destroy_queue || !operations.read64 || !operations.write64 ||
      !operations.shutdown)
    throw std::invalid_argument("incomplete embedded device callbacks");
}

CallbackDeviceBackend::~CallbackDeviceBackend() {
  dispatcher_.invoke([&] { shutdown(); });
}

void CallbackDeviceBackend::require_success(int result) {
  if (result)
    throw std::system_error(-result, std::generic_category(), "embedded device callback");
}

int CallbackDeviceBackend::status(int result) {
  if (result > 0) {
    memory_.fail(-EPROTO);
    return -EPROTO;
  }
  return result;
}

int CallbackDeviceBackend::process_call(rj_kmd_process_fn callback, uint32_t process, int32_t pid) {
  auto request = record<rj_kmd_process_request_t>();
  request.process_id = process;
  request.client_pid = pid;
  return status(callback(operations_.context, &request));
}

void CallbackDeviceBackend::configure(const DeviceApertures &source, uint32_t,
                                      const DeviceCallbacks &callbacks) {
  dispatcher_.invoke([&] {
    callbacks_ = callbacks;
    auto request = record<rj_kmd_apertures_t>();
    request.lds_base = source.lds_base;
    request.lds_limit = source.lds_limit;
    request.scratch_base = source.scratch_base;
    request.scratch_limit = source.scratch_limit;
    require_success(status(operations_.set_apertures(operations_.context, &request)));
  });
}

int CallbackDeviceBackend::register_process(KfdProcess &process, bool passthrough) {
  if (passthrough)
    return -ENOTSUP;
  return dispatcher_.invoke([&] {
    if (int result = failure())
      return result;
    auto [it, inserted] =
        processes_.try_emplace(process.process_id(), Process{process.client_pid()});
    if (!inserted)
      return -EEXIST;
    const int result =
        process_call(operations_.open_process, process.process_id(), process.client_pid());
    if (result)
      processes_.erase(it);
    else
      active_processes_.fetch_add(1);
    return result;
  });
}

void CallbackDeviceBackend::shutdown() {
  if (shutdown_)
    return;
  operations_.shutdown(operations_.context);
  shutdown_ = true;
  for (const auto &[process, state] : processes_)
    doorbells_.remove_process(process);
  processes_.clear();
  active_processes_.store(0);
}

void CallbackDeviceBackend::prepare_process_teardown(uint32_t process) {
  dispatcher_.invoke_when(
      [this, process] {
        return !quiescence_probe_ || quiescence_probe_(quiescence_context_, process) != 0;
      },
      [] {});
}

void CallbackDeviceBackend::teardown_result(int result) {
  if (result) {
    memory_.fail(result);
    shutdown(); // A failed per-resource cleanup must still stop device accesses.
  }
}

void CallbackDeviceBackend::unregister_process(uint32_t process) {
  dispatcher_.invoke([&] {
    auto it = processes_.find(process);
    if (it == processes_.end())
      return;
    const int result = process_call(operations_.close_process, process, it->second.pid);
    doorbells_.remove_process(process);
    processes_.erase(it);
    active_processes_.fetch_sub(1);
    teardown_result(result);
  });
}

void CallbackDeviceBackend::set_process_client_pid(uint32_t process, int32_t pid) {
  dispatcher_.invoke([&] {
    require_success(process_call(operations_.set_process_client_pid, process, pid));
    processes_.at(process).pid = pid;
  });
}

void CallbackDeviceBackend::set_process_mem_fd(uint32_t, int fd) {
  if (fd >= 0)
    require_success(-ENOTSUP); // No external-device debugger capability.
}

int CallbackDeviceBackend::create_queue(KfdProcess &, const QueueDescriptor &source) {
  // ABI v2 describes AQL/SDMA queues. A native PM4 queue cannot be represented.
  if (source.is_pm4 || source.scheduling_percentage == 0)
    return -ENOTSUP;
  // xcd_fanout controls simdojo's CP distribution. An external device owns
  // its hardware queue placement and validates it in create_queue itself.
  return dispatcher_.invoke([&] {
    if (int result = failure())
      return result;
    auto process = processes_.find(source.process_id);
    if (process == processes_.end())
      return -ESRCH;
    int result = doorbells_.reserve(source.process_id, source.queue_id, source.doorbell_offset);
    if (result)
      return result;
    auto request = record<rj_kmd_queue_t>();
    request.process_id = source.process_id;
    request.queue_id = source.queue_id;
    request.ring_base_va = source.ring_base_va;
    request.ring_size = source.ring_size;
    request.doorbell_offset = source.doorbell_offset;
    request.read_pointer_va = source.read_ptr_va;
    request.write_pointer_va = source.write_ptr_va;
    request.queue_descriptor_va = source.queue_desc_va;
    request.host_accessible = source.host_accessible;
    request.is_sdma = source.is_sdma;
    result = status(operations_.create_queue(operations_.context, &request));
    if (result)
      doorbells_.remove(source.process_id, source.queue_id);
    else
      doorbells_.activate(source.process_id, source.queue_id,
                          source.doorbell_base ? source.doorbell_base
                                               : process->second.doorbell_base);
    return result;
  });
}

int CallbackDeviceBackend::update_queue(uint32_t process, uint32_t queue, uint64_t ring,
                                        uint32_t size, uint32_t percentage) {
  return dispatcher_.invoke([&] {
    if (int result = failure())
      return result;
    if (!doorbells_.contains(process, queue))
      return -EINVAL;
    auto request = record<rj_kmd_queue_update_t>();
    request.process_id = process;
    request.queue_id = queue;
    request.ring_base_va = ring;
    request.ring_size = size;
    request.queue_percentage = percentage;
    return status(operations_.update_queue(operations_.context, &request));
  });
}

void CallbackDeviceBackend::destroy_queue(uint32_t process, uint32_t queue) {
  dispatcher_.invoke([&] {
    if (!doorbells_.contains(process, queue))
      return;
    auto request = record<rj_kmd_queue_key_t>();
    request.process_id = process;
    request.queue_id = queue;
    const int result = status(operations_.destroy_queue(operations_.context, &request));
    doorbells_.remove(process, queue);
    teardown_result(result);
  });
}

void CallbackDeviceBackend::set_doorbell_base(uint32_t process, void *base) {
  dispatcher_.invoke([&] {
    auto it = processes_.find(process);
    if (it == processes_.end()) {
      // Common KFD clears the private mapping after unregister_process.
      if (!base)
        return;
      require_success(-ESRCH);
    }
    it->second.doorbell_base = base;
    doorbells_.set_base(process, base);
  });
}

uint64_t CallbackDeviceBackend::read64(uint32_t process, uint64_t address) {
  return dispatcher_.invoke([&] {
    require_success(failure());
    auto request = record<rj_kmd_read64_t>();
    request.process_id = process;
    request.gpu_va = address;
    uint64_t value = 0;
    require_success(status(operations_.read64(operations_.context, &request, &value)));
    return value;
  });
}

bool CallbackDeviceBackend::write64(uint32_t process, uint64_t address, uint64_t value) {
  return dispatcher_.invoke([&] {
    require_success(failure());
    auto request = record<rj_kmd_write64_t>();
    request.process_id = process;
    request.gpu_va = address;
    request.value = value;
    require_success(status(operations_.write64(operations_.context, &request)));
    return true;
  });
}

void CallbackDeviceBackend::post_interrupt(uint32_t process, uint32_t event) {
  if (callbacks_.interrupt)
    callbacks_.interrupt(process, event);
}

void CallbackDeviceBackend::post_memory_fault(uint32_t process, uint64_t address,
                                              MemoryFaultCause cause) {
  if (reporter_)
    reporter_->report_memory_fault(process, address, cause);
}

} // namespace rocjitsu::kmd
