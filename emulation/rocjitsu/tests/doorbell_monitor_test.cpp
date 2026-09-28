// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/doorbell_monitor.h"

#include <gtest/gtest.h>
#include <poll.h>
#include <sys/mman.h>

namespace rocjitsu {
namespace {

bool wait_pending(DoorbellMonitor &monitor, size_t count) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (monitor.pending() < count && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return monitor.pending() >= count;
}

TEST(DoorbellMonitorTest, ReservedQueueDoesNotReadBeforeActivation) {
  alignas(8) uint64_t doorbell = 4;
  DoorbellMonitor monitor;
  ASSERT_EQ(monitor.reserve(7, 3, 0), 0);
  monitor.set_base(7, &doorbell);
  pollfd descriptor{monitor.fd(), POLLIN, 0};
  EXPECT_EQ(::poll(&descriptor, 1, 10), 0);
  monitor.activate(7, 3, &doorbell);
  ASSERT_EQ(::poll(&descriptor, 1, 5000), 1);
  rj_kmd_doorbell_record_t record{};
  ASSERT_EQ(monitor.drain(&record, 1), 1u);
  EXPECT_EQ(record.abi_version, RJ_KMD_SERVER_ABI_VERSION);
  EXPECT_EQ(record.process_id, 7u);
  EXPECT_EQ(record.queue_id, 3u);
  EXPECT_EQ(record.value, 4u);
  EXPECT_EQ(::poll(&descriptor, 1, 0), 0);
}

TEST(DoorbellMonitorTest, PartialDrainKeepsRemainingQueuesReadable) {
  alignas(8) uint64_t doorbells[] = {11, 22};
  DoorbellMonitor monitor;
  ASSERT_EQ(monitor.reserve(1, 1, 0), 0);
  ASSERT_EQ(monitor.reserve(1, 2, 8), 0);
  monitor.activate(1, 1, doorbells);
  monitor.activate(1, 2, doorbells);
  ASSERT_TRUE(wait_pending(monitor, 2));
  rj_kmd_doorbell_record_t record{};
  ASSERT_EQ(monitor.drain(&record, 1), 1u);
  EXPECT_EQ(record.value, 11u);
  pollfd descriptor{monitor.fd(), POLLIN, 0};
  EXPECT_EQ(::poll(&descriptor, 1, 0), 1);
  ASSERT_EQ(monitor.drain(&record, 1), 1u);
  EXPECT_EQ(record.value, 22u);
  EXPECT_EQ(record.offset, 8u);
  EXPECT_EQ(::poll(&descriptor, 1, 0), 0);
}

TEST(DoorbellMonitorTest, RemovalQuiescesMappingAndDropsStaleQueueNotifications) {
  struct Mapping {
    void *address =
        ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ~Mapping() {
      if (address != MAP_FAILED)
        ::munmap(address, 4096);
    }
  } mapping;
  ASSERT_NE(mapping.address, MAP_FAILED);
  alignas(8) uint64_t replacement = 99;
  DoorbellMonitor monitor;
  ASSERT_EQ(monitor.reserve(1, 5, 0), 0);
  monitor.activate(1, 5, mapping.address);
  ASSERT_TRUE(wait_pending(monitor, 1));
  monitor.remove(1, 5);
  EXPECT_EQ(monitor.pending(), 0u);
  ASSERT_EQ(::mprotect(mapping.address, 4096, PROT_NONE), 0);
  ASSERT_EQ(monitor.reserve(1, 5, 0), 0);
  monitor.activate(1, 5, &replacement);
  ASSERT_TRUE(wait_pending(monitor, 1));
  rj_kmd_doorbell_record_t record{};
  ASSERT_EQ(monitor.drain(&record, 1), 1u);
  EXPECT_EQ(record.value, 99u);
}

TEST(DoorbellMonitorTest, ProcessRemovalDoesNotDiscardAnotherProcessesNotifications) {
  alignas(8) uint64_t first = 1, second = 2;
  DoorbellMonitor monitor;
  ASSERT_EQ(monitor.reserve(1, 1, 0), 0);
  ASSERT_EQ(monitor.reserve(2, 1, 0), 0);
  monitor.activate(1, 1, &first);
  monitor.activate(2, 1, &second);
  ASSERT_TRUE(wait_pending(monitor, 2));
  monitor.remove_process(1);
  EXPECT_EQ(monitor.pending(), 1u);
  rj_kmd_doorbell_record_t record{};
  ASSERT_EQ(monitor.drain(&record, 1), 1u);
  EXPECT_EQ(record.process_id, 2u);
  EXPECT_EQ(record.value, 2u);
}

TEST(DoorbellMonitorTest, RejectsDuplicateIdentityAndUnalignedOffset) {
  DoorbellMonitor monitor;
  EXPECT_EQ(monitor.reserve(1, 1, 1), -EINVAL);
  ASSERT_EQ(monitor.reserve(1, 1, 8), 0);
  EXPECT_EQ(monitor.reserve(1, 1, 16), -EEXIST);
  monitor.remove(1, 1);
  EXPECT_EQ(monitor.reserve(1, 1, 16), 0);
}

} // namespace
} // namespace rocjitsu
