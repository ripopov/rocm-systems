// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/device/callback_memory.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>

namespace rocjitsu::kmd {
namespace {

class CallbackMemoryTest : public testing::Test {
protected:
  CallbackDispatcher dispatcher;
  const std::thread::id owner = std::this_thread::get_id();
  unsigned allocations = 0;
  unsigned releases = 0;
  int released_fd = -1;
  int allocate_status = 0;
  int free_status = 0;
  int map_status = 0;
  int unmap_status = 0;
  bool invalid_slice = false;
  rj_kmd_mapping_t last_map{};
  rj_kmd_unmap_request_t last_unmap{};

  rj_kmd_backend_ops_t operations() {
    rj_kmd_backend_ops_t ops{};
    ops.abi_version = RJ_KMD_SERVER_ABI_VERSION;
    ops.struct_size = sizeof(ops);
    ops.context = this;
    ops.allocate = [](void *context, const rj_kmd_allocation_request_t *request,
                      rj_kmd_backing_slice_t *slice) -> int32_t {
      auto &self = *static_cast<CallbackMemoryTest *>(context);
      EXPECT_EQ(std::this_thread::get_id(), self.owner);
      EXPECT_EQ(request->abi_version, RJ_KMD_SERVER_ABI_VERSION);
      EXPECT_EQ(request->gpu_id, 42u);
      EXPECT_EQ(request->process_id, 7u);
      if (self.allocate_status)
        return self.allocate_status;
      const int fd = memfd_create("callback-memory-test", MFD_CLOEXEC);
      if (fd < 0)
        return -errno;
      if (ftruncate(fd, 16384) != 0) {
        const int status = -errno;
        close(fd);
        return status;
      }
      ++self.allocations;
      self.released_fd = fd;
      slice->allocation_id = 123;
      slice->device_paddr = 0x8000;
      slice->size = 4096;
      slice->fd_offset = 8192;
      slice->fd = fd;
      if (self.invalid_slice)
        slice->abi_version = 0;
      return 0;
    };
    ops.free = [](void *context, const rj_kmd_free_request_t *request) -> int32_t {
      auto &self = *static_cast<CallbackMemoryTest *>(context);
      EXPECT_EQ(std::this_thread::get_id(), self.owner);
      EXPECT_EQ(request->process_id, 7u);
      EXPECT_EQ(request->allocation_id, 123u);
      ++self.releases;
      return self.free_status;
    };
    ops.map = [](void *context, const rj_kmd_mapping_t *request) -> int32_t {
      auto &self = *static_cast<CallbackMemoryTest *>(context);
      EXPECT_EQ(std::this_thread::get_id(), self.owner);
      self.last_map = *request;
      return self.map_status;
    };
    ops.unmap = [](void *context, const rj_kmd_unmap_request_t *request) -> int32_t {
      auto &self = *static_cast<CallbackMemoryTest *>(context);
      EXPECT_EQ(std::this_thread::get_id(), self.owner);
      self.last_unmap = *request;
      return self.unmap_status;
    };
    return ops;
  }
};

TEST_F(CallbackMemoryTest, WorkerLeaseTransfersFdAndReleasesOnlyLastReference) {
  CallbackMemory memory(42, operations(), dispatcher);
  dispatcher.run_while_dispatching([&] {
    std::shared_ptr<DeviceAllocation> allocation;
    ASSERT_EQ(memory.allocate({7, 0x10000, 4096, 0}, allocation), 0);
    EXPECT_EQ(allocation->file_offset, 8192u);
    EXPECT_EQ(allocation->device_address, 0x8000u);
    auto imported = allocation;
    allocation.reset();
    EXPECT_GE(fcntl(imported->fd, F_GETFD), 0);
    EXPECT_EQ(memory.map({8, imported->id, 0x20000, imported->device_address, 4096, 2}), 0);
    EXPECT_EQ(memory.unmap(8, 0x20000, 4096), 0);
    imported.reset();
  });
  EXPECT_EQ(allocations, 1u);
  EXPECT_EQ(releases, 1u);
  EXPECT_EQ(last_map.process_id, 8u);
  EXPECT_EQ(last_map.memory_type, 2u);
  EXPECT_EQ(last_map.device_paddr, 0x8000u);
  EXPECT_EQ(last_unmap.gpu_va, 0x20000u);
  EXPECT_EQ(fcntl(released_fd, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
}

TEST_F(CallbackMemoryTest, InvalidSuccessfulReplyRollsBackLeaseAndDescriptor) {
  invalid_slice = true;
  CallbackMemory memory(42, operations(), dispatcher);
  std::shared_ptr<DeviceAllocation> allocation;
  EXPECT_EQ(memory.allocate({7, 0x10000, 4096, 0}, allocation), -EPROTO);
  EXPECT_FALSE(allocation);
  EXPECT_EQ(releases, 1u);
  EXPECT_EQ(fcntl(released_fd, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
}

TEST_F(CallbackMemoryTest, FailedAllocationDoesNotPublishOrReleaseStorage) {
  allocate_status = -ENOMEM;
  CallbackMemory memory(42, operations(), dispatcher);
  std::shared_ptr<DeviceAllocation> allocation;
  EXPECT_EQ(memory.allocate({7, 0x10000, 4096, 0}, allocation), -ENOMEM);
  EXPECT_FALSE(allocation);
  EXPECT_EQ(allocations, 0u);
  EXPECT_EQ(releases, 0u);
}

TEST_F(CallbackMemoryTest, FailedReleaseIsLatchedAndPreventsFurtherUse) {
  free_status = -EBUSY;
  CallbackMemory memory(42, operations(), dispatcher);
  dispatcher.run_while_dispatching([&] {
    std::shared_ptr<DeviceAllocation> allocation;
    ASSERT_EQ(memory.allocate({7, 0x10000, 4096, 0}, allocation), 0);
    allocation.reset();
    EXPECT_EQ(memory.failure(), -EBUSY);
    EXPECT_EQ(memory.allocate({7, 0x10000, 4096, 0}, allocation), -EBUSY);
    EXPECT_EQ(memory.map({7, 123, 0x10000, 0x8000, 4096, 2}), -EBUSY);
    EXPECT_EQ(memory.unmap(7, 0x10000, 4096), 0);
  });
  EXPECT_EQ(allocations, 1u);
  EXPECT_EQ(releases, 1u);
  EXPECT_EQ(last_unmap.process_id, 7u);
}

TEST_F(CallbackMemoryTest, IncompleteOrOldAbiIsRejected) {
  auto ops = operations();
  ops.abi_version = 1;
  EXPECT_THROW(CallbackMemory(42, ops, dispatcher), std::invalid_argument);
  ops = operations();
  ops.free = nullptr;
  EXPECT_THROW(CallbackMemory(42, ops, dispatcher), std::invalid_argument);
}

TEST_F(CallbackMemoryTest, MappingFailuresPropagateWithoutBecomingSuccess) {
  CallbackMemory memory(42, operations(), dispatcher);
  map_status = -ENOTSUP;
  unmap_status = -EBUSY;
  EXPECT_EQ(memory.map({7, 123, 0x10000, 0x8000, 4096, 2}), -ENOTSUP);
  EXPECT_EQ(memory.unmap(7, 0x10000, 4096), -EBUSY);
  EXPECT_EQ(memory.failure(), 0);
  map_status = 0;
  unmap_status = 0;
  EXPECT_EQ(memory.map({7, 123, 0x10000, 0x8000, 4096, 2}), 0);
  EXPECT_EQ(memory.unmap(7, 0x10000, 4096), 0);
}

TEST_F(CallbackMemoryTest, PositiveCallbackStatusIsTerminalProtocolFailure) {
  CallbackMemory memory(42, operations(), dispatcher);
  map_status = 1;
  EXPECT_EQ(memory.map({7, 123, 0x10000, 0x8000, 4096, 2}), -EPROTO);
  EXPECT_EQ(memory.failure(), -EPROTO);
}

} // namespace
} // namespace rocjitsu::kmd
