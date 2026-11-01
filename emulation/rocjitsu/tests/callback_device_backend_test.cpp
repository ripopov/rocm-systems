// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/callback_device_backend.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include <gtest/gtest.h>

namespace rocjitsu {
namespace {

struct CallbackHost {
  std::thread::id owner = std::this_thread::get_id();
  int opens = 0, closes = 0, creates = 0, destroys = 0, shutdowns = 0;
  int create_result = 0, update_result = 0, destroy_result = 0;
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
      ++get(c).closes;
      return 0;
    };
    ops.set_process_client_pid = [](void *c, const rj_kmd_process_request_t *) -> int32_t {
      get(c);
      return 0;
    };
    ops.set_apertures = [](void *c, const rj_kmd_apertures_t *) -> int32_t {
      get(c);
      return 0;
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
      *value = get(c).value;
      return 0;
    };
    ops.write64 = [](void *c, const rj_kmd_write64_t *value) -> int32_t {
      get(c).value = value->value;
      return 0;
    };
    ops.shutdown = [](void *c) { ++get(c).shutdowns; };
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
      device->write64(pid, 0x10000, 123);
      EXPECT_EQ(device->read64(pid, 0x10000), 123u);
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
