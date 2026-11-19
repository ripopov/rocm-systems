// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/callback_device_backend.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include <gtest/gtest.h>

namespace rocjitsu {
namespace {

class CallbackHost {
public:
  std::thread::id owner = std::this_thread::get_id();
  int opens = 0, closes = 0, creates = 0, destroys = 0, shutdowns = 0;
  int create_result = 0, update_result = 0, destroy_result = 0;
  int configure_result = 0, pid_result = 0, read_result = 0, write_result = 0, close_result = 0;
  unsigned reads = 0, writes = 0;
  uint32_t percentage = 0;
  uint64_t value = 0;
  static CallbackHost &get(void *context) {
    auto &host = *static_cast<CallbackHost *>(context);
    EXPECT_EQ(std::this_thread::get_id(), host.owner);
    return host;
  }
  rj_kmd_backend_ops_t operations() {
    rj_kmd_backend_ops_t ops{};
    ops.abi_version = RJ_KMD_SERVER_ABI_VERSION;
    ops.struct_size = sizeof(ops);
    ops.context = this;
    ops.open_process = [](void *c, const rj_kmd_process_request_t *) -> int32_t {
      ++get(c).opens;
      return 0;
    };
    ops.close_process = [](void *c, const rj_kmd_process_request_t *) -> int32_t {
      auto &host = get(c);
      ++host.closes;
      return host.close_result;
    };
    ops.set_process_client_pid = [](void *c, const rj_kmd_process_request_t *) -> int32_t {
      return get(c).pid_result;
    };
    ops.set_apertures = [](void *c, const rj_kmd_apertures_t *) -> int32_t {
      return get(c).configure_result;
    };
    ops.allocate = [](void *, const rj_kmd_allocation_request_t *,
                      rj_kmd_backing_slice_t *) -> int32_t { return -ENOTSUP; };
    ops.free = [](void *, const rj_kmd_free_request_t *) -> int32_t { return 0; };
    ops.map = [](void *, const rj_kmd_mapping_t *) -> int32_t { return -ENOTSUP; };
    ops.unmap = [](void *, const rj_kmd_unmap_request_t *) -> int32_t { return 0; };
    ops.create_queue = [](void *c, const rj_kmd_queue_t *) -> int32_t {
      auto &h = get(c);
      ++h.creates;
      return h.create_result;
    };
    ops.update_queue = [](void *c, const rj_kmd_queue_update_t *q) -> int32_t {
      auto &h = get(c);
      h.percentage = q->queue_percentage;
      return h.update_result;
    };
    ops.destroy_queue = [](void *c, const rj_kmd_queue_key_t *) -> int32_t {
      auto &h = get(c);
      ++h.destroys;
      return h.destroy_result;
    };
    ops.read64 = [](void *c, const rj_kmd_read64_t *, uint64_t *value) -> int32_t {
      auto &host = get(c);
      ++host.reads;
      *value = host.value; // Even an unsuccessful callback may write its out parameter.
      return host.read_result;
    };
    ops.write64 = [](void *c, const rj_kmd_write64_t *value) -> int32_t {
      auto &host = get(c);
      ++host.writes;
      if (!host.write_result)
        host.value = value->value;
      return host.write_result;
    };
    ops.shutdown = [](void *c) { ++get(c).shutdowns; };
    ops.read_clock = [](void *c, uint64_t *nanoseconds) -> int32_t {
      get(c);
      *nanoseconds = 12345;
      return 0;
    };
    return ops;
  }
};

TEST(CallbackDeviceBackendTest, CommonKfdUsesOwnerCallbacksAndRollsBackFailedQueues) {
  CallbackHost host;
  CallbackDispatcher dispatcher;
  {
    auto backend = std::make_unique<kmd::CallbackDeviceBackend>(ROCJITSU_CODE_ARCH_CDNA4, 7,
                                                                host.operations(), dispatcher);
    auto *device = backend.get();
    std::vector<SimulatedKfd::BackendDevice> devices;
    devices.push_back({7, std::move(backend)});
    SimulatedKfd driver(std::move(devices));
    dispatcher.run_while_dispatching([&] {
      const auto pid = driver.open_process();
      ASSERT_NE(pid, 0);
      EXPECT_EQ(device->active_processes(), 1u);
      kfd_ioctl_get_clock_counters_args clocks{};
      clocks.gpu_id = 7;
      EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_GET_CLOCK_COUNTERS, &clocks), 0);
      EXPECT_EQ(clocks.gpu_clock_counter, 12345u);
      EXPECT_EQ(clocks.system_clock_counter, 12345u);
      EXPECT_EQ(clocks.system_clock_freq, 1000000000u);
      kfd_ioctl_create_queue_args queue{};
      queue.gpu_id = 7;
      queue.queue_percentage = 100;
      queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
      kfd_ioctl_set_xnack_mode_args xnack{};
      xnack.xnack_enabled = 1;
      EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_SET_XNACK_MODE, &xnack), -ENOTSUP);
      queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE;
      EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &queue), -ENOTSUP);
      queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
      queue.queue_percentage = 0;
      EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &queue), -ENOTSUP);
      EXPECT_EQ(host.creates, 0);
      queue.queue_percentage = 100;
      dispatcher.invoke([&] { host.create_result = -EBUSY; });
      EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &queue), -EBUSY);
      EXPECT_EQ(device->doorbells().active(), 0u);
      dispatcher.invoke([&] { host.create_result = 0; });
      ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &queue), 0);
      EXPECT_EQ(device->doorbells().active(), 1u);
      kfd_ioctl_alloc_queue_gws_args gws{};
      gws.queue_id = queue.queue_id;
      gws.num_gws = 1;
      EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_ALLOC_QUEUE_GWS, &gws), -EOPNOTSUPP);

      kfd_ioctl_update_queue_args update{};
      update.queue_id = queue.queue_id;
      update.queue_percentage = 0;
      dispatcher.invoke([&] { host.update_result = -ENOTSUP; });
      EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_UPDATE_QUEUE, &update), -ENOTSUP);
      EXPECT_EQ(device->write64(pid, 0x10000, 123), 0);
      uint64_t value = 0;
      EXPECT_EQ(device->read64(pid, 0x10000, value), 0);
      EXPECT_EQ(value, 123u);
      EXPECT_EQ(driver.close(pid), 0);
      EXPECT_EQ(device->active_processes(), 0u);
      EXPECT_EQ(device->doorbells().active(), 0u);
    });
    EXPECT_EQ(host.opens, 1);
    EXPECT_EQ(host.closes, 1);
    EXPECT_EQ(host.creates, 2);
    EXPECT_EQ(host.destroys, 1);
  }
  EXPECT_EQ(host.shutdowns, 1);
}

TEST(CallbackDeviceBackendTest, FailedQueueTeardownQuiescesDeviceAndLatchesTerminalError) {
  CallbackHost host;
  host.destroy_result = -EIO;
  CallbackDispatcher dispatcher;
  {
    auto backend = std::make_unique<kmd::CallbackDeviceBackend>(ROCJITSU_CODE_ARCH_CDNA4, 7,
                                                                host.operations(), dispatcher);
    auto *device = backend.get();
    std::vector<SimulatedKfd::BackendDevice> devices;
    devices.push_back({7, std::move(backend)});
    SimulatedKfd driver(std::move(devices));
    dispatcher.run_while_dispatching([&] {
      const auto pid = driver.open_process();
      ASSERT_NE(pid, 0);
      kfd_ioctl_create_queue_args queue{};
      queue.gpu_id = 7;
      queue.queue_percentage = 100;
      queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
      ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &queue), 0);
      EXPECT_EQ(driver.close(pid), 0);
      EXPECT_EQ(device->failure(), -EIO);
      EXPECT_EQ(device->active_processes(), 0u);
      EXPECT_EQ(device->doorbells().active(), 0u);
      EXPECT_EQ(driver.open_process(), 0u);
    });
    EXPECT_EQ(host.shutdowns, 1);
  }
  EXPECT_EQ(host.shutdowns, 1);
}

TEST(CallbackDeviceBackendTest, MemoryErrorsPreserveOutputsAndAllowRetry) {
  CallbackHost host;
  CallbackDispatcher dispatcher;
  auto backend = std::make_unique<kmd::CallbackDeviceBackend>(ROCJITSU_CODE_ARCH_CDNA4, 7,
                                                              host.operations(), dispatcher);
  auto *device = backend.get();
  std::vector<SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  SimulatedKfd driver(std::move(devices));
  dispatcher.run_while_dispatching([&] {
    const uint32_t process = driver.open_process();
    ASSERT_NE(process, 0u);
    uint64_t observed = 99;
    EXPECT_EQ(device->read64(process, 0x10000, observed), 0);
    EXPECT_EQ(observed, 0u);
    dispatcher.invoke([&] {
      host.read_result = -ENXIO;
      host.write_result = -EBUSY;
    });
    observed = 77;
    EXPECT_EQ(device->read64(process, 0x10000, observed), -ENXIO);
    EXPECT_EQ(observed, 77u);
    EXPECT_EQ(device->write64(process, 0x10000, 123), -EBUSY);
    EXPECT_EQ(device->failure(), 0);
    EXPECT_EQ(device->set_process_mem_fd(process, 3), -ENOTSUP);
    EXPECT_EQ(device->set_process_mem_fd(process, -1), 0);
    dispatcher.invoke([&] {
      EXPECT_EQ(host.value, 0u);
      host.read_result = 0;
      host.write_result = 0;
    });
    EXPECT_EQ(device->write64(process, 0x10000, 123), 0);
    EXPECT_EQ(device->read64(process, 0x10000, observed), 0);
    EXPECT_EQ(observed, 123u);
    EXPECT_EQ(driver.close(process), 0);
  });
}

TEST(CallbackDeviceBackendTest, QueueMemoryErrorsPreserveErrnoAndReleaseDoorbellReservations) {
  CallbackHost host;
  host.read_result = -EACCES;
  CallbackDispatcher dispatcher;
  auto backend = std::make_unique<kmd::CallbackDeviceBackend>(ROCJITSU_CODE_ARCH_CDNA4, 7,
                                                              host.operations(), dispatcher);
  auto *device = backend.get();
  std::vector<SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  SimulatedKfd driver(std::move(devices), true);
  dispatcher.run_while_dispatching([&] {
    const uint32_t process = driver.open_process();
    ASSERT_NE(process, 0u);
    kfd_ioctl_create_queue_args queue{};
    queue.gpu_id = 7;
    queue.queue_percentage = 100;
    queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    queue.ctx_save_restore_address = 0x10000;
    EXPECT_EQ(driver.ioctl(process, AMDKFD_IOC_CREATE_QUEUE, &queue), -EACCES);
    EXPECT_EQ(device->doorbells().active(), 0u);
    dispatcher.invoke([&] {
      EXPECT_EQ(host.creates, 0);
      host.read_result = 0;
    });
    EXPECT_EQ(driver.ioctl(process, AMDKFD_IOC_CREATE_QUEUE, &queue), 0);
    EXPECT_EQ(queue.doorbell_offset & 0xFFF, 0u);
    EXPECT_EQ(device->doorbells().active(), 1u);
    queue.queue_type = KFD_IOC_QUEUE_TYPE_SDMA;
    queue.ctx_save_restore_address = 0;
    dispatcher.invoke([&] { host.write_result = -EROFS; });
    EXPECT_EQ(driver.ioctl(process, AMDKFD_IOC_CREATE_QUEUE, &queue), -EROFS);
    EXPECT_EQ(device->doorbells().active(), 1u);
    dispatcher.invoke([&] {
      EXPECT_EQ(host.creates, 1);
      host.write_result = 0;
    });
    EXPECT_EQ(driver.ioctl(process, AMDKFD_IOC_CREATE_QUEUE, &queue), 0);
    EXPECT_EQ(queue.doorbell_offset & 0xFFF, 8u);
    EXPECT_EQ(device->doorbells().active(), 2u);
    EXPECT_EQ(driver.close(process), 0);
  });
}

TEST(CallbackDeviceBackendTest, SetupErrorsDoNotPublishProcessesAndRollBackEveryDevice) {
  CallbackHost first, second;
  first.configure_result = -EACCES;
  second.pid_result = -EBUSY;
  CallbackDispatcher dispatcher;
  std::vector<SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::make_unique<kmd::CallbackDeviceBackend>(
                            ROCJITSU_CODE_ARCH_CDNA4, 7, first.operations(), dispatcher)});
  devices.push_back({8, std::make_unique<kmd::CallbackDeviceBackend>(
                            ROCJITSU_CODE_ARCH_CDNA4, 8, second.operations(), dispatcher)});
  SimulatedKfd driver(std::move(devices));
  dispatcher.run_while_dispatching([&] {
    EXPECT_EQ(driver.open_process(1234), 0u);
    EXPECT_EQ(errno, EACCES);
    dispatcher.invoke([&] {
      EXPECT_EQ(first.opens, 0);
      first.configure_result = 0;
    });
    EXPECT_EQ(driver.open_process(1234), 0u);
    EXPECT_EQ(errno, EBUSY);
    dispatcher.invoke([&] {
      EXPECT_EQ(first.opens, 1);
      EXPECT_EQ(second.opens, 1);
      EXPECT_EQ(first.closes, 1);
      EXPECT_EQ(second.closes, 1);
      second.pid_result = 0;
    });
    const uint32_t process = driver.open_process(1234);
    ASSERT_NE(process, 0u);
    EXPECT_EQ(driver.close(process), 0);
  });
  EXPECT_EQ(first.opens, 2);
  EXPECT_EQ(second.opens, 2);
  EXPECT_EQ(first.closes, 2);
  EXPECT_EQ(second.closes, 2);
}

TEST(CallbackDeviceBackendTest, PositiveMemoryStatusesLatchFailureAndRefuseFurtherOperations) {
  for (bool writing : {false, true}) {
    CallbackHost host;
    CallbackDispatcher dispatcher;
    auto backend = std::make_unique<kmd::CallbackDeviceBackend>(ROCJITSU_CODE_ARCH_CDNA4, 7,
                                                                host.operations(), dispatcher);
    auto *device = backend.get();
    std::vector<SimulatedKfd::BackendDevice> devices;
    devices.push_back({7, std::move(backend)});
    SimulatedKfd driver(std::move(devices));
    dispatcher.run_while_dispatching([&] {
      const uint32_t process = driver.open_process();
      ASSERT_NE(process, 0u);
      dispatcher.invoke([&] {
        host.read_result = 1;
        host.write_result = 1;
      });
      uint64_t observed = 77;
      EXPECT_EQ(writing ? device->write64(process, 0x10000, 123)
                        : device->read64(process, 0x10000, observed),
                -EPROTO);
      EXPECT_EQ(device->failure(), -EPROTO);
      EXPECT_EQ(observed, 77u);
      EXPECT_EQ(device->read64(process, 0x10000, observed), -EPROTO);
      EXPECT_EQ(device->write64(process, 0x10000, 456), -EPROTO);
      EXPECT_EQ(driver.close(process), 0);
    });
    EXPECT_EQ(host.reads + host.writes, 1u);
  }
}

TEST(CallbackDeviceBackendTest, FailedProcessCloseQuiescesDeviceAndLatchesError) {
  CallbackHost host;
  host.close_result = -EIO;
  CallbackDispatcher dispatcher;
  auto backend = std::make_unique<kmd::CallbackDeviceBackend>(ROCJITSU_CODE_ARCH_CDNA4, 7,
                                                              host.operations(), dispatcher);
  auto *device = backend.get();
  std::vector<SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  SimulatedKfd driver(std::move(devices));
  dispatcher.run_while_dispatching([&] {
    const uint32_t process = driver.open_process();
    ASSERT_NE(process, 0u);
    EXPECT_EQ(driver.close(process), 0);
    EXPECT_EQ(device->failure(), -EIO);
    EXPECT_EQ(device->active_processes(), 0u);
    EXPECT_EQ(driver.open_process(), 0u);
    EXPECT_EQ(errno, EIO);
  });
  EXPECT_EQ(host.shutdowns, 1);
}

TEST(CallbackDeviceBackendTest, ProcessCloseRetainsQueuesUntilQuiescence) {
  CallbackHost host;
  CallbackDispatcher dispatcher;
  bool ready = false;
  unsigned probes = 0;
  std::promise<void> parked;
  auto waiting = parked.get_future();
  auto backend = std::make_unique<kmd::CallbackDeviceBackend>(ROCJITSU_CODE_ARCH_CDNA4, 7,
                                                              host.operations(), dispatcher);
  auto *device = backend.get();
  struct Probe {
    CallbackHost &host;
    bool &ready;
    unsigned &calls;
    std::promise<void> &parked;
  } probe{host, ready, probes, parked};
  device->set_quiescence_probe(
      [](void *context, uint32_t) -> int32_t {
        auto &state = *static_cast<Probe *>(context);
        EXPECT_EQ(std::this_thread::get_id(), state.host.owner);
        if (++state.calls == 1)
          state.parked.set_value();
        if (!state.ready) {
          EXPECT_EQ(state.host.destroys, 0);
          EXPECT_EQ(state.host.closes, 0);
        }
        return state.ready;
      },
      &probe);
  std::vector<SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  SimulatedKfd driver(std::move(devices));
  dispatcher.run_while_dispatching([&] {
    const auto pid = driver.open_process();
    ASSERT_NE(pid, 0);
    kfd_ioctl_create_queue_args queue{};
    queue.gpu_id = 7;
    queue.queue_percentage = 100;
    queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &queue), 0);
    auto closing = std::async(std::launch::async, [&] { return driver.close(pid); });
    waiting.wait();
    // An ordinary owner callback can still run while teardown is deferred.
    dispatcher.invoke([&] {
      EXPECT_EQ(host.destroys, 0);
      EXPECT_EQ(host.closes, 0);
      ready = true;
    });
    EXPECT_EQ(closing.get(), 0);
  });
  EXPECT_GT(probes, 0u);
  EXPECT_EQ(host.destroys, 1);
  EXPECT_EQ(host.closes, 1);
}

} // namespace
} // namespace rocjitsu
