// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/simdojo_device_backend.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gws_device.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/shader_engine.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <system_error>

namespace rocjitsu::kmd {

SimdojoDeviceBackend::SimdojoDeviceBackend(SoC &soc, uint32_t ordinal)
    : soc_(soc), ordinal_(ordinal), legacy_vm_(soc.gpu_vm(), soc.memory()) {}

SimdojoDeviceBackend::~SimdojoDeviceBackend() = default;

void SimdojoDeviceBackend::FaultReporterBridge::report_memory_fault(
    uint32_t vmid, uint64_t address, amdgpu::MemoryFaultCause cause) {
  kmd::MemoryFaultReporter *reporter = reporter_.load(std::memory_order_acquire);
  if (!reporter)
    return;
  const auto translated =
      cause == amdgpu::MemoryFaultCause::ReadOnly     ? kmd::MemoryFaultCause::ReadOnly
      : cause == amdgpu::MemoryFaultCause::NotPresent ? kmd::MemoryFaultCause::NotPresent
                                                      : kmd::MemoryFaultCause::Indeterminate;
  reporter->report_memory_fault(vmid, address, translated);
}

void SimdojoDeviceBackend::set_active(bool active) {
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    for (auto *cu : cp->compute_units())
      cu->set_debug_active(active);
  });
}

void SimdojoDeviceBackend::release_queue(uint32_t process, uint32_t queue) {
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_queue_debug_suspended(queue, process, false);
    for (auto *cu : cp->compute_units()) {
      bool wake = false;
      cu->with_wave_state_locked([&] {
        for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
          auto *wave = cu->wf(slot);
          if (!wave || wave->is_halted() || wave->process_id() != process ||
              wave->queue_id() != queue)
            continue;
          if (wave->fatal_exception_pending()) {
            wave->set_fatal_exception_cwsr_valid(false);
            continue;
          }
          wave->set_debug_single_step(false);
          wave->set_debug_halted(false);
          wave->set_debug_suspended(false);
          // The architectural half of the same stop. s_sendmsghalt raises
          // STATUS.HALT and s_rfe consults it on the way out of the handler,
          // so leaving it set outlives the session: the wave would re-halt at
          // the handler return with no debugger left to resume it, and the
          // queue would never drain.
          wave->set_status_halt(false);
          wave->set_self_halted(false);
          wake = true;
        }
      });
      if (wake)
        cu->schedule_work_async();
    }
  });
}

std::function<bool(bool wait_for_ack)>
SimdojoDeviceBackend::prepare_queue_exception(uint32_t process, uint32_t queue, uint64_t mask) {
  amdgpu::CommandProcessor *publisher = nullptr;
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    if (cp->signal_queue_exception(queue, process, mask, false) && publisher == nullptr)
      publisher = cp;
  });
  if (!publisher)
    return {};
  return [publisher, queue, process, mask](bool wait_for_ack) {
    return publisher->publish_queue_exception(queue, process, mask, wait_for_ack);
  };
}

uint32_t SimdojoDeviceBackend::num_xcc() const { return soc_.num_xcds(); }

rj_code_arch_t SimdojoDeviceBackend::architecture() const { return soc_.arch(); }

void SimdojoDeviceBackend::configure(const DeviceApertures &apertures,
                                     uint32_t scratch_wave_divisor,
                                     const DeviceCallbacks &callbacks) {
  const uint32_t xcc_count = soc_.num_xcds();
  for (uint32_t xcc = 0; xcc < xcc_count; ++xcc) {
    auto *cp = soc_.xcd(xcc)->command_processor();
    if (!cp)
      continue;
    cp->set_scratch_wave_divisor(scratch_wave_divisor);
  }
  soc_.set_apertures(apertures.lds_base, apertures.lds_limit, apertures.scratch_base,
                     apertures.scratch_limit);
  interrupt_subscription_ = amdgpu::InterruptSubscription(callbacks.interrupt);
  soc_.for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_scratch_backing_resolver(callbacks.scratch_backing);
    cp->set_scratch_backing_allocator(callbacks.allocate_scratch);
    for (auto *cu : cp->compute_units()) {
      cu->set_trap_handler_resolver(
          [resolve = callbacks.trap_handler](const amdgpu::Wavefront &wave)
              -> std::optional<amdgpu::ComputeUnitCore::TrapHandlerConfig> {
            if (!resolve)
              return std::nullopt;
            const auto handler = resolve(wave.process_id());
            if (!handler)
              return std::nullopt;
            return amdgpu::ComputeUnitCore::TrapHandlerConfig{handler->tba, handler->tma,
                                                              handler->debug_enabled};
          });
    }
  });
  configure_debug_callbacks(callbacks);
}

void SimdojoDeviceBackend::set_memory_fault_reporter(MemoryFaultReporter *reporter) {
  fault_reporter_.set(reporter);
}

int SimdojoDeviceBackend::register_process(KfdProcess &process, bool passthrough) {
  const auto handle = legacy_vm_.register_address_space(
      process.process_id(), {.page_table = &process.page_table_,
                             .page_table_mutex = &process.page_table_mutex_,
                             .page_table_generation = process.page_table_generation(),
                             .request_mutex = process.page_table_request_mutex(),
                             .mutation_epoch = process.page_table_mutation_epoch(),
                             .page_table_cache_state = process.page_table_cache_state(),
                             // A local VM binding uses direct backing semantics.
                             // KFD still retains the real PID for debugger/session identity.
                             .client_pid = passthrough ? 0 : process.client_pid(),
                             .client_mem_fd = -1,
                             .passthrough = passthrough,
                             .fault_reporter = &fault_reporter_});
  if (!handle)
    return -ENOMEM;
  {
    std::lock_guard lock(state_mutex_);
    processes_[process.process_id()] = handle;
  }
  process.gpu(ordinal_).address_space = handle;
  return 0;
}

void SimdojoDeviceBackend::unregister_process(uint32_t process) {
  {
    std::lock_guard lock(pm4_mutex_);
    for (auto it = pm4_queues_.begin(); it != pm4_queues_.end();) {
      if (it->second.process != process) {
        ++it;
        continue;
      }
      it->second.cp->unregister_drm_queue(it->second.queue_id, process);
      it = pm4_queues_.erase(it);
    }
  }
  amdgpu::AddressSpaceHandle handle;
  {
    std::lock_guard lock(state_mutex_);
    auto it = processes_.find(process);
    if (it == processes_.end())
      return;
    handle = it->second;
    processes_.erase(it);
  }
  (void)legacy_vm_.unregister_address_space(handle);
  soc_.gws_device()->release_process(process);
}

void SimdojoDeviceBackend::set_process_client_pid(uint32_t process, int32_t client_pid) {
  std::lock_guard lock(state_mutex_);
  if (auto it = processes_.find(process); it != processes_.end())
    (void)legacy_vm_.set_client_pid(it->second, client_pid);
}

void SimdojoDeviceBackend::set_process_mem_fd(uint32_t process, int fd) {
  std::lock_guard lock(state_mutex_);
  if (auto it = processes_.find(process); it != processes_.end())
    (void)legacy_vm_.set_client_mem_fd(it->second, fd);
}

std::optional<uint32_t> SimdojoDeviceBackend::queue_xcc(uint32_t ordinal) const {
  if (!soc_.assign_queue_owner_cp(ordinal))
    return std::nullopt;
  return soc_.queue_xcd_id(ordinal);
}

int SimdojoDeviceBackend::create_queue(KfdProcess &process, const QueueDescriptor &queue) {
  amdgpu::AddressSpaceHandle address_space;
  {
    std::lock_guard lock(state_mutex_);
    auto it = processes_.find(queue.process_id);
    if (it == processes_.end() || queues_.contains(queue_key(queue.process_id, queue.queue_id)))
      return -EINVAL;
    address_space = it->second;
  }
  amdgpu::QueueRegistrationRequest request{};
  request.identity = {.address_space = address_space,
                      .interrupt_sink = interrupt_subscription_.sink(),
                      .process_id = queue.process_id,
                      .queue_id = queue.queue_id};
  request.ring = {.base_address = queue.ring_base_va,
                  .size_bytes = queue.ring_size,
                  .consumer_pointer_address = queue.read_ptr_va,
                  .producer_pointer_address = queue.write_ptr_va};
  request.doorbell = {.mode = queue.host_accessible ? amdgpu::QueueDoorbellMode::HostPolled
                                                    : amdgpu::QueueDoorbellMode::Explicit,
                      .offset = queue.doorbell_offset,
                      .host_base = queue.doorbell_base,
                      .last_value = ~uint64_t(0)};
  request.type = queue.is_sdma ? amdgpu::QueueType::Sdma : amdgpu::QueueType::Compute;
  request.packet_format = queue.is_sdma  ? amdgpu::QueuePacketFormat::Sdma
                          : queue.is_pm4 ? amdgpu::QueuePacketFormat::Pm4
                                         : amdgpu::QueuePacketFormat::Aql;
  request.abi =
      (queue.is_sdma || queue.is_pm4) ? amdgpu::QueueAbi::Generic : amdgpu::QueueAbi::KfdAql;
  request.initial_consumer_cursor =
      (queue.is_sdma || queue.is_pm4) ? std::optional<uint64_t>{0} : std::nullopt;
  request.engine_id = queue.sdma_engine_id;
  request.xcd_fanout = queue.xcd_fanout;
  request.queue_descriptor_address = queue.queue_desc_va;
  request.exception_status_address = queue.exception_status_va;
  request.exception_event_id = queue.exception_event_id;
  if (queue.is_sdma) {
    request.binding_factory = amdgpu::make_sdma_queue_binding_factory(soc_.sdma_queue_scheduler());
  } else {
    auto *cp = soc_.assign_queue_owner_cp(queue.ordinal);
    if (!cp)
      return -EINVAL;
    request.binding_factory = amdgpu::make_compute_queue_binding_factory(*cp);
  }
  request.scheduling_percentage = queue.scheduling_percentage;
  const amdgpu::QueueHandle handle = soc_.queue_registry().register_queue(request);
  if (!handle)
    return -EINVAL;
  try {
    {
      std::lock_guard lock(state_mutex_);
      queues_.emplace(queue_key(queue.process_id, queue.queue_id), handle);
    }
    {
      std::lock_guard lock(process.alloc_mutex_);
      process.queue_doorbell_map_.at(queue.queue_id).queue_handle = handle;
    }
  } catch (...) {
    (void)soc_.queue_registry().unregister_queue(handle, amdgpu::QueueCloseMode::ForceCancel);
    std::lock_guard lock(state_mutex_);
    queues_.erase(queue_key(queue.process_id, queue.queue_id));
    throw;
  }
  return 0;
}

int SimdojoDeviceBackend::update_queue(uint32_t process, uint32_t queue, uint64_t ring,
                                       uint32_t size, uint32_t percentage) {
  amdgpu::QueueHandle handle;
  {
    std::lock_guard lock(state_mutex_);
    auto it = queues_.find(queue_key(process, queue));
    if (it == queues_.end())
      return -ENOENT;
    handle = it->second;
  }
  const auto result = soc_.queue_registry().reconfigure_queue(
      handle,
      {.ring_base_address = ring, .ring_size_bytes = size, .scheduling_percentage = percentage});
  return result ? 0 : result.status == amdgpu::QueueReconfigureStatus::Busy ? -EBUSY : -EINVAL;
}

void SimdojoDeviceBackend::destroy_queue(uint32_t process, uint32_t queue) {
  amdgpu::QueueHandle handle;
  {
    std::lock_guard lock(state_mutex_);
    auto it = queues_.find(queue_key(process, queue));
    if (it == queues_.end())
      return;
    handle = it->second;
    queues_.erase(it);
  }
  (void)soc_.queue_registry().unregister_queue(handle, amdgpu::QueueCloseMode::ForceCancel);
}

int SimdojoDeviceBackend::set_cu_mask(uint32_t process, uint32_t queue,
                                      std::span<const uint32_t> mask, CuMaskGeometry geometry,
                                      bool is_sdma) {
  const bool rdna = arch_is_rdna(soc_.arch());
  const uint32_t bit_count = static_cast<uint32_t>(mask.size() * 32);
  if (rdna) {
    for (uint32_t bit = 0; bit < bit_count; bit += 2) {
      const uint32_t pair = (mask[bit / 32] >> (bit % 32)) & 3u;
      if (pair != 0 && pair != 3)
        return -EINVAL;
    }
  }
  if (is_sdma)
    return 0;
  {
    std::lock_guard lock(state_mutex_);
    if (!queues_.contains(queue_key(process, queue)))
      return -EFAULT;
  }
  const uint32_t arrays = std::max(1u, geometry.arrays_per_engine);
  const uint32_t step = rdna ? 2u : 1u;
  amdgpu::QueueCuSelection selected(std::in_place);
  uint32_t bit = 0;
  for (uint32_t cu = 0; cu < geometry.cu_per_array && bit < bit_count && bit < geometry.active_cus;
       cu += step) {
    for (uint32_t sh = 0; sh < arrays; ++sh) {
      for (uint32_t se = 0; se < geometry.shader_engines; ++se) {
        for (amdgpu::Xcd *xcd : soc_.xcds()) {
          if (se >= xcd->num_shader_engines())
            continue;
          amdgpu::ShaderEngine *shader = xcd->shader_engine(se);
          const uint32_t local = sh * geometry.cu_per_array + cu;
          if (local + step > shader->num_compute_units())
            continue;
          const bool enabled = bit < bit_count && bit < geometry.active_cus &&
                               ((mask[bit / 32] >> (bit % 32)) & ((1u << step) - 1));
          for (uint32_t half = 0; half < step; ++half, ++bit) {
            if (enabled)
              selected->push_back(shader->compute_unit(local + half));
          }
        }
      }
    }
  }
  soc_.for_each_cp(
      [&](amdgpu::CommandProcessor *cp) { cp->set_queue_cu_selection(queue, process, selected); });
  return 0;
}

void SimdojoDeviceBackend::set_doorbell_base(uint32_t process, void *base) {
  soc_.set_process_doorbell_base(process, base);
}

uint64_t SimdojoDeviceBackend::read64(uint32_t process, uint64_t address) {
  const auto access = soc_.gpu_vm().snapshot_vmid(process);
  if (!access)
    throw std::system_error(EFAULT, std::generic_category());
  uint64_t value = 0;
  auto bytes = std::as_writable_bytes(std::span<uint64_t>(&value, 1));
  if (access->read(address, bytes) == amdgpu::VmAccessOutcome::Complete)
    return value;
  throw std::system_error(EFAULT, std::generic_category());
}

bool SimdojoDeviceBackend::write64(uint32_t process, uint64_t address, uint64_t value) {
  const auto access = soc_.gpu_vm().snapshot_vmid(process);
  return access && access->write(address, std::as_bytes(std::span<const uint64_t>(&value, 1))) ==
                       amdgpu::VmAccessOutcome::Complete;
}

int SimdojoDeviceBackend::submit_pm4(uint32_t process, uint64_t queue_key,
                                     amdgpu::Pm4Submission submission) {
  std::lock_guard lock(pm4_mutex_);
  auto it = pm4_queues_.find(queue_key);
  if (it == pm4_queues_.end()) {
    amdgpu::AddressSpaceHandle address_space;
    {
      std::lock_guard state_lock(state_mutex_);
      auto process_it = processes_.find(process);
      if (process_it == processes_.end())
        return -ENODEV;
      address_space = process_it->second;
    }
    auto *cp = soc_.assign_queue_owner_cp(0);
    if (!cp)
      return -ENODEV;
    amdgpu::ComputeQueueConfig queue;
    queue.address_space = address_space;
    queue.process_id = process;
    queue.queue_id = next_pm4_queue_id_++;
    const uint32_t queue_id = queue.queue_id;
    if (!cp->register_drm_queue(std::move(queue)))
      return -EIO;
    it = pm4_queues_.emplace(queue_key, Pm4Queue{cp, process, queue_id}).first;
  }
  if (it->second.process != process)
    return -EINVAL;
  return it->second.cp->submit_pm4(it->second.queue_id, process, std::move(submission)) ? 0 : -EIO;
}

void SimdojoDeviceBackend::retire_pm4_queue(uint64_t queue_key) {
  std::lock_guard lock(pm4_mutex_);
  auto it = pm4_queues_.find(queue_key);
  if (it == pm4_queues_.end())
    return;
  it->second.cp->unregister_drm_queue(it->second.queue_id, it->second.process);
  pm4_queues_.erase(it);
}

} // namespace rocjitsu::kmd
