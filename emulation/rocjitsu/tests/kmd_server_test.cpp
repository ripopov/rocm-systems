// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/server.h"

#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/remote_driver.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <poll.h>
#include <stop_token>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

template <typename T> T abiStruct() {
  T value{};
  value.abi_version = RJ_KMD_SERVER_ABI_VERSION;
  value.struct_size = sizeof(T);
  return value;
}

struct FakeBackend {
  int fd = -1;
  std::atomic<uint64_t> allocations{0};
  std::atomic<uint64_t> frees{0};
  std::atomic<uint64_t> maps{0};
};

int32_t processOk([[maybe_unused]] void *context,
                  [[maybe_unused]] const rj_kmd_process_request_t *request) {
  return 0;
}

int32_t allocateSlice(void *opaque,
                      const rj_kmd_allocation_request_t *request,
                      rj_kmd_backing_slice_t *slice) {
  auto &backend = *static_cast<FakeBackend *>(opaque);
  uint64_t index = backend.allocations.fetch_add(1);
  constexpr uint64_t page_mask = 4095;
  const uint64_t page_offset = request->gpu_va & page_mask;
  slice->allocation_id = index + 1;
  slice->device_paddr =
      0x80000000 + index * 0x10000 + page_offset;
  slice->size = (page_offset + request->size + page_mask) & ~page_mask;
  slice->fd = fcntl(backend.fd, F_DUPFD_CLOEXEC, 128);
  slice->fd_offset = index == 0 ? 0 : 8192;
  return slice->fd >= 0 ? 0 : -errno;
}

int32_t freeSlice(void *opaque,
                  [[maybe_unused]] const rj_kmd_free_request_t *request) {
  static_cast<FakeBackend *>(opaque)->frees.fetch_add(1);
  return 0;
}

int32_t mapSlice(void *opaque,
                 [[maybe_unused]] const rj_kmd_mapping_t *mapping) {
  static_cast<FakeBackend *>(opaque)->maps.fetch_add(1);
  return 0;
}

int32_t unmapOk([[maybe_unused]] void *context,
                [[maybe_unused]] const rj_kmd_unmap_request_t *request) {
  return 0;
}

int32_t aperturesOk([[maybe_unused]] void *context,
                    [[maybe_unused]] const rj_kmd_apertures_t *request) {
  return 0;
}

int32_t queueOk([[maybe_unused]] void *context,
                [[maybe_unused]] const rj_kmd_queue_t *request) {
  return 0;
}

int32_t queueKeyOk([[maybe_unused]] void *context,
                   [[maybe_unused]] const rj_kmd_queue_key_t *request) {
  return 0;
}

class ServerReactor {
public:
  explicit ServerReactor(rj_kmd_server_t *server) : server_(server) {
    thread_ = std::jthread(
        [this](std::stop_token stop) { run(stop); });
  }

  ~ServerReactor() {
    thread_.request_stop();
    if (thread_.joinable())
      thread_.join();
  }

private:
  void run(std::stop_token stop) {
    while (!stop.stop_requested()) {
      size_t count = 0;
      if (rj_kmd_server_poll_fds(server_, nullptr, &count) != 0)
        return;
      std::vector<rj_kmd_poll_fd_t> descriptors(count);
      if (count > 0 &&
          rj_kmd_server_poll_fds(server_, descriptors.data(), &count) != 0)
        return;
      std::vector<pollfd> poll_fds;
      poll_fds.reserve(count);
      for (size_t index = 0; index < count; ++index)
        poll_fds.push_back(
            {descriptors[index].fd, descriptors[index].events, 0});
      int ready = poll(poll_fds.data(), poll_fds.size(), 10);
      if (ready < 0 && errno == EINTR)
        continue;
      if (ready < 0)
        return;
      for (const auto &poll_fd : poll_fds) {
        if (poll_fd.revents == 0)
          continue;
        char error[256] = {};
        (void)rj_kmd_server_process_fd(
            server_, poll_fd.fd, poll_fd.revents, error, sizeof(error));
      }
    }
  }

  rj_kmd_server_t *server_;
  std::jthread thread_;
};

int connectSocket(const std::string &path) {
  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0)
    return -1;
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  for (int attempt = 0; attempt < 100; ++attempt) {
    if (connect(fd, reinterpret_cast<sockaddr *>(&address),
                sizeof(address)) == 0)
      return fd;
    if (errno != ENOENT && errno != ECONNREFUSED)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  syscall(SYS_close, fd);
  return -1;
}

auto testDevice() {
  auto device = abiStruct<rj_kmd_device_info_t>();
  device.architecture = 2;
  device.gpu_id = 38144;
  device.gfx_target_version = 90402;
  device.vendor_id = 0x1002;
  device.device_id = 0x74a1;
  device.drm_render_minor = 128;
  device.wave_front_size = 64;
  device.local_mem_size = 1ull << 30;
  std::strcpy(device.marketing_name, "rocjitsu KMD server test GPU");
  return device;
}

auto fakeOperations(FakeBackend *fake) {
  auto operations = abiStruct<rj_kmd_backend_ops_t>();
  operations.context = fake;
  operations.open_process = processOk;
  operations.close_process = processOk;
  operations.allocate = allocateSlice;
  operations.free = freeSlice;
  operations.map = mapSlice;
  operations.unmap = unmapOk;
  operations.set_apertures = aperturesOk;
  operations.create_queue = queueOk;
  operations.destroy_queue = queueKeyOk;
  return operations;
}

rj_kmd_server_t *createListeningServer(const std::string &runtime,
                                       const std::string &socket_path,
                                       FakeBackend *fake) {
  auto device = testDevice();
  auto config = abiStruct<rj_kmd_server_config_t>();
  config.runtime_root = runtime.c_str();
  config.socket_path = socket_path.c_str();
  auto operations = fakeOperations(fake);

  rj_kmd_server_t *server = nullptr;
  char error[256] = {};
  if (rj_kmd_server_create(&device, &config, &operations, &server, error,
                           sizeof(error)) != 0) {
    ADD_FAILURE() << error;
    return nullptr;
  }
  if (rj_kmd_server_listen(server, error, sizeof(error)) != 0) {
    ADD_FAILURE() << error;
    rj_kmd_server_destroy(server);
    return nullptr;
  }
  return server;
}

TEST(KmdServerTest, TransfersTwoSlicesOfOneFdAtNonzeroOffsets) {
  char runtime_template[] = "/tmp/rocjitsu_kmd_server_XXXXXX";
  char *runtime = mkdtemp(runtime_template);
  ASSERT_NE(runtime, nullptr);
  std::string socket_path = std::string(runtime) + "/kmd.sock";

  FakeBackend fake{};
  fake.fd = static_cast<int>(
      syscall(SYS_memfd_create, "kmd-server-test", MFD_CLOEXEC));
  ASSERT_GE(fake.fd, 0);
  ASSERT_EQ(ftruncate(fake.fd, 16384), 0);

  auto device = testDevice();
  rj_kmd_server_t *server =
      createListeningServer(runtime, socket_path, &fake);
  ASSERT_NE(server, nullptr);

  {
    ServerReactor reactor(server);
    int socket_fd = connectSocket(socket_path);
    ASSERT_GE(socket_fd, 0);
    rocjitsu::RemoteDriver remote(socket_fd);
    ASSERT_GE(remote.open(), 0);

    kfd_ioctl_alloc_memory_of_gpu_args first{};
    first.gpu_id = device.gpu_id;
    first.size = 4096;
    first.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
    ASSERT_EQ(remote.ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &first), 0);

    kfd_ioctl_alloc_memory_of_gpu_args second{};
    second.gpu_id = device.gpu_id;
    second.size = 4096;
    second.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
    ASSERT_EQ(remote.ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &second), 0);

    void *first_mapping =
        remote.mmap(nullptr, first.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    static_cast<off_t>(first.mmap_offset));
    ASSERT_NE(first_mapping, MAP_FAILED);
    void *second_mapping =
        remote.mmap(nullptr, second.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    static_cast<off_t>(second.mmap_offset));
    ASSERT_NE(second_mapping, MAP_FAILED);

    constexpr uint64_t first_value = 0x1111222233334444ull;
    constexpr uint64_t second_value = 0xaaaabbbbccccddddull;
    *static_cast<uint64_t *>(first_mapping) = first_value;
    *static_cast<uint64_t *>(second_mapping) = second_value;

    uint64_t observed = 0;
    ASSERT_EQ(pread(fake.fd, &observed, sizeof(observed), 0),
              static_cast<ssize_t>(sizeof(observed)));
    EXPECT_EQ(observed, first_value);
    ASSERT_EQ(pread(fake.fd, &observed, sizeof(observed), 8192),
              static_cast<ssize_t>(sizeof(observed)));
    EXPECT_EQ(observed, second_value);

    int found_fd = -1;
    off_t found_offset = -1;
    ASSERT_TRUE(remote.find_memfd_for_addr(
        reinterpret_cast<void *>(second.va_addr), sizeof(uint64_t), &found_fd,
        &found_offset));
    EXPECT_EQ(found_offset, 8192);
    struct stat source_stat {};
    struct stat received_stat {};
    ASSERT_EQ(fstat(fake.fd, &source_stat), 0);
    ASSERT_EQ(fstat(found_fd, &received_stat), 0);
    EXPECT_EQ(source_stat.st_dev, received_stat.st_dev);
    EXPECT_EQ(source_stat.st_ino, received_stat.st_ino);

    kfd_ioctl_free_memory_of_gpu_args free_first{.handle = first.handle};
    kfd_ioctl_free_memory_of_gpu_args free_second{.handle = second.handle};
    EXPECT_EQ(remote.ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_first), 0);
    EXPECT_EQ(remote.ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_second), 0);
    EXPECT_EQ(fake.frees.load(), 2);
    EXPECT_GE(fake.maps.load(), 2);

    syscall(SYS_munmap, first_mapping, first.size);
    syscall(SYS_munmap, second_mapping, second.size);
    EXPECT_EQ(remote.close(), 0);
  }

  rj_kmd_server_destroy(server);
  syscall(SYS_close, fake.fd);
  std::filesystem::remove_all(runtime);
}

TEST(KmdServerTest, PromotesUnalignedUserptrWithoutClobberingItsPages) {
  char runtime_template[] = "/tmp/rocjitsu_kmd_userptr_XXXXXX";
  char *runtime = mkdtemp(runtime_template);
  ASSERT_NE(runtime, nullptr);
  std::string socket_path = std::string(runtime) + "/kmd.sock";

  constexpr size_t page_size = 4096;
  constexpr size_t mapping_size = 2 * page_size;
  FakeBackend fake{};
  fake.fd = static_cast<int>(
      syscall(SYS_memfd_create, "kmd-userptr-test", MFD_CLOEXEC));
  ASSERT_GE(fake.fd, 0);
  ASSERT_EQ(ftruncate(fake.fd, mapping_size), 0);

  auto *original = static_cast<uint8_t *>(
      mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(original, MAP_FAILED);
  std::vector<uint8_t> expected(mapping_size);
  for (size_t index = 0; index < mapping_size; ++index) {
    expected[index] = static_cast<uint8_t>((index * 37 + 11) & 0xff);
    original[index] = expected[index];
  }

  auto device = testDevice();
  rj_kmd_server_t *server =
      createListeningServer(runtime, socket_path, &fake);
  ASSERT_NE(server, nullptr);

  {
    ServerReactor reactor(server);
    int socket_fd = connectSocket(socket_path);
    ASSERT_GE(socket_fd, 0);
    rocjitsu::RemoteDriver remote(socket_fd);
    ASSERT_GE(remote.open(), 0);

    kfd_ioctl_alloc_memory_of_gpu_args allocation{};
    allocation.gpu_id = device.gpu_id;
    allocation.va_addr = 0x1000000010;
    allocation.mmap_offset =
        reinterpret_cast<uint64_t>(original + 16);
    allocation.size = page_size;
    allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
                       KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    ASSERT_EQ(
        remote.ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), 0);

    EXPECT_EQ(std::memcmp(original, expected.data(), mapping_size), 0);
    original[0] = 0xa5;
    original[16] = 0x5a;
    original[mapping_size - 1] = 0xc3;
    uint8_t observed = 0;
    ASSERT_EQ(pread(fake.fd, &observed, 1, 0), 1);
    EXPECT_EQ(observed, 0xa5);
    ASSERT_EQ(pread(fake.fd, &observed, 1, 16), 1);
    EXPECT_EQ(observed, 0x5a);
    ASSERT_EQ(pread(fake.fd, &observed, 1, mapping_size - 1), 1);
    EXPECT_EQ(observed, 0xc3);

    int found_fd = -1;
    off_t found_offset = -1;
    ASSERT_TRUE(remote.find_memfd_for_addr(
        original + 16, allocation.size, &found_fd, &found_offset));
    EXPECT_EQ(found_offset, 16);

    kfd_ioctl_free_memory_of_gpu_args release{
        .handle = allocation.handle};
    EXPECT_EQ(remote.ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &release), 0);
    EXPECT_EQ(remote.close(), 0);
  }

  rj_kmd_server_destroy(server);
  syscall(SYS_munmap, original, mapping_size);
  syscall(SYS_close, fake.fd);
  std::filesystem::remove_all(runtime);
}

TEST(KmdServerTest, SharesOnlyAccessInPlaceSvmRanges) {
  char runtime_template[] = "/tmp/rocjitsu_kmd_svm_XXXXXX";
  char *runtime = mkdtemp(runtime_template);
  ASSERT_NE(runtime, nullptr);
  std::string socket_path = std::string(runtime) + "/kmd.sock";

  constexpr size_t page_size = 4096;
  constexpr size_t mapping_size = 2 * page_size;
  FakeBackend fake{};
  fake.fd = static_cast<int>(
      syscall(SYS_memfd_create, "kmd-svm-test", MFD_CLOEXEC));
  ASSERT_GE(fake.fd, 0);
  ASSERT_EQ(ftruncate(fake.fd, mapping_size), 0);

  auto *original = static_cast<uint8_t *>(
      mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(original, MAP_FAILED);
  std::vector<uint8_t> expected(mapping_size);
  for (size_t index = 0; index < mapping_size; ++index) {
    expected[index] = static_cast<uint8_t>((index * 19 + 7) & 0xff);
    original[index] = expected[index];
  }

  auto device = testDevice();
  rj_kmd_server_t *server =
      createListeningServer(runtime, socket_path, &fake);
  ASSERT_NE(server, nullptr);

  {
    ServerReactor reactor(server);
    int socket_fd = connectSocket(socket_path);
    ASSERT_GE(socket_fd, 0);
    rocjitsu::RemoteDriver remote(socket_fd);
    ASSERT_GE(remote.open(), 0);

    std::vector<uint8_t> buffer(
        sizeof(kfd_ioctl_svm_args) + sizeof(kfd_ioctl_svm_attribute));
    auto *svm_args =
        reinterpret_cast<kfd_ioctl_svm_args *>(buffer.data());
    auto *attribute =
        reinterpret_cast<kfd_ioctl_svm_attribute *>(svm_args + 1);
    svm_args->start_addr = reinterpret_cast<uint64_t>(original);
    svm_args->size = mapping_size;
    svm_args->op = KFD_IOCTL_SVM_OP_SET_ATTR;
    svm_args->nattr = 1;
    attribute->type = KFD_IOCTL_SVM_ATTR_ACCESS;
    attribute->value = device.gpu_id;
    const unsigned long svm_request =
        rocjitsu::ioctl_with_size(AMDKFD_IOC_SVM, buffer.size());

    ASSERT_EQ(remote.ioctl(svm_request, svm_args), 0);
    EXPECT_EQ(fake.allocations.load(), 0);
    EXPECT_EQ(fake.maps.load(), 0);
    EXPECT_EQ(std::memcmp(original, expected.data(), mapping_size), 0);

    attribute->type = KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE;
    ASSERT_EQ(remote.ioctl(svm_request, svm_args), 0);
    EXPECT_EQ(fake.allocations.load(), 1);
    EXPECT_GE(fake.maps.load(), 1);
    EXPECT_EQ(std::memcmp(original, expected.data(), mapping_size), 0);

    original[0] = 0xa5;
    original[page_size + 17] = 0x5a;
    original[mapping_size - 1] = 0xc3;
    uint8_t observed = 0;
    ASSERT_EQ(pread(fake.fd, &observed, 1, 0), 1);
    EXPECT_EQ(observed, 0xa5);
    ASSERT_EQ(pread(fake.fd, &observed, 1, page_size + 17), 1);
    EXPECT_EQ(observed, 0x5a);
    ASSERT_EQ(pread(fake.fd, &observed, 1, mapping_size - 1), 1);
    EXPECT_EQ(observed, 0xc3);

    int found_fd = -1;
    off_t found_offset = -1;
    ASSERT_TRUE(remote.find_memfd_for_addr(
        original, mapping_size, &found_fd, &found_offset));
    EXPECT_EQ(found_offset, 0);

    EXPECT_EQ(remote.close(), 0);
    for (int attempt = 0;
         attempt < 100 && fake.frees.load() == 0; ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(fake.frees.load(), 1);
  }

  rj_kmd_server_destroy(server);
  syscall(SYS_munmap, original, mapping_size);
  syscall(SYS_close, fake.fd);
  std::filesystem::remove_all(runtime);
}

TEST(KmdServerTest, BuffersAPartialFrameUntilItIsComplete) {
  char runtime_template[] = "/tmp/rocjitsu_kmd_partial_XXXXXX";
  char *runtime = mkdtemp(runtime_template);
  ASSERT_NE(runtime, nullptr);
  std::string socket_path = std::string(runtime) + "/kmd.sock";
  FakeBackend fake{};
  rj_kmd_server_t *server =
      createListeningServer(runtime, socket_path, &fake);
  ASSERT_NE(server, nullptr);

  {
    ServerReactor reactor(server);
    int socket_fd = connectSocket(socket_path);
    ASSERT_GE(socket_fd, 0);

    rocjitsu::RpcHeader request{};
    request.opcode = rocjitsu::RPC_HANDSHAKE;
    request.request_id = 37;
    const auto *bytes = reinterpret_cast<const uint8_t *>(&request);
    constexpr size_t split = 5;
    ASSERT_EQ(send(socket_fd, bytes, split, MSG_NOSIGNAL),
              static_cast<ssize_t>(split));

    pollfd descriptor{socket_fd, POLLIN, 0};
    EXPECT_EQ(poll(&descriptor, 1, 30), 0);

    ASSERT_EQ(send(socket_fd, bytes + split, sizeof(request) - split,
                   MSG_NOSIGNAL),
              static_cast<ssize_t>(sizeof(request) - split));
    ASSERT_GT(poll(&descriptor, 1, 1000), 0);
    ASSERT_TRUE(descriptor.revents & POLLIN);

    rocjitsu::RpcHeader response{};
    ASSERT_TRUE(
        rocjitsu::rpc_recv_exact(socket_fd, &response, sizeof(response)));
    EXPECT_EQ(response.opcode, rocjitsu::RPC_HANDSHAKE);
    EXPECT_EQ(response.request_id, request.request_id);
    EXPECT_GT(response.payload_bytes, 0u);
    std::vector<uint8_t> payload(response.payload_bytes);
    ASSERT_TRUE(
        rocjitsu::rpc_recv_exact(socket_fd, payload.data(), payload.size()));
    syscall(SYS_close, socket_fd);
  }

  rj_kmd_server_destroy(server);
  std::filesystem::remove_all(runtime);
}

TEST(KmdServerTest, RejectsAnInvalidHeaderAndRemainsUsable) {
  char runtime_template[] = "/tmp/rocjitsu_kmd_invalid_XXXXXX";
  char *runtime = mkdtemp(runtime_template);
  ASSERT_NE(runtime, nullptr);
  std::string socket_path = std::string(runtime) + "/kmd.sock";
  FakeBackend fake{};
  rj_kmd_server_t *server =
      createListeningServer(runtime, socket_path, &fake);
  ASSERT_NE(server, nullptr);

  {
    ServerReactor reactor(server);
    int bad_socket = connectSocket(socket_path);
    ASSERT_GE(bad_socket, 0);
    rocjitsu::RpcHeader invalid{};
    invalid.opcode = rocjitsu::RPC_HANDSHAKE;
    invalid.reserved = 1;
    ASSERT_EQ(send(bad_socket, &invalid, sizeof(invalid), MSG_NOSIGNAL),
              static_cast<ssize_t>(sizeof(invalid)));
    pollfd descriptor{bad_socket, POLLIN | POLLHUP, 0};
    ASSERT_GT(poll(&descriptor, 1, 1000), 0);
    char byte = 0;
    EXPECT_EQ(recv(bad_socket, &byte, sizeof(byte), 0), 0);
    syscall(SYS_close, bad_socket);

    int good_socket = connectSocket(socket_path);
    ASSERT_GE(good_socket, 0);
    rocjitsu::RemoteDriver remote(good_socket);
    EXPECT_GE(remote.open(), 0);
    EXPECT_EQ(remote.close(), 0);
  }

  rj_kmd_server_destroy(server);
  std::filesystem::remove_all(runtime);
}

TEST(KmdServerTest, WatchesQueueCreatedBeforeDoorbellPageIsMapped) {
  char runtime_template[] = "/tmp/rocjitsu_kmd_doorbell_XXXXXX";
  char *runtime = mkdtemp(runtime_template);
  ASSERT_NE(runtime, nullptr);
  std::string socket_path = std::string(runtime) + "/kmd.sock";

  FakeBackend fake{};
  fake.fd = static_cast<int>(
      syscall(SYS_memfd_create, "kmd-doorbell-test", MFD_CLOEXEC));
  ASSERT_GE(fake.fd, 0);
  ASSERT_EQ(ftruncate(fake.fd, 4096), 0);

  auto device = testDevice();
  rj_kmd_server_t *server =
      createListeningServer(runtime, socket_path, &fake);
  ASSERT_NE(server, nullptr);

  {
    ServerReactor reactor(server);
    int socket_fd = connectSocket(socket_path);
    ASSERT_GE(socket_fd, 0);
    rocjitsu::RemoteDriver remote(socket_fd);
    ASSERT_GE(remote.open(), 0);

    kfd_ioctl_alloc_memory_of_gpu_args allocation{};
    allocation.gpu_id = device.gpu_id;
    allocation.size = 4096;
    allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL;
    ASSERT_EQ(
        remote.ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), 0);

    kfd_ioctl_create_queue_args queue{};
    queue.ring_base_address = 0x100000;
    queue.ring_size = 4096;
    queue.gpu_id = device.gpu_id;
    queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    queue.read_pointer_address = 0x200000;
    queue.write_pointer_address = 0x200008;
    ASSERT_EQ(remote.ioctl(AMDKFD_IOC_CREATE_QUEUE, &queue), 0);

    rj_kmd_server_stats_t stats = abiStruct<rj_kmd_server_stats_t>();
    ASSERT_EQ(rj_kmd_server_get_stats(server, &stats), 0);
    EXPECT_EQ(stats.active_queues, 1u);
    EXPECT_EQ(stats.pending_doorbells, 0u);

    void *doorbells = remote.mmap(
        nullptr, allocation.size, PROT_READ | PROT_WRITE, MAP_SHARED,
        static_cast<off_t>(queue.doorbell_offset));
    ASSERT_NE(doorbells, MAP_FAILED);
    __atomic_store_n(static_cast<uint64_t *>(doorbells), uint64_t(0),
                     __ATOMIC_RELEASE);

    pollfd descriptor{
        rj_kmd_server_notification_fd(server), POLLIN, 0};
    ASSERT_GT(poll(&descriptor, 1, 2000), 0);
    ASSERT_TRUE(descriptor.revents & POLLIN);

    size_t count = 0;
    ASSERT_EQ(rj_kmd_server_drain_doorbells(server, nullptr, &count), 0);
    ASSERT_EQ(count, 1u);
    rj_kmd_doorbell_record_t record{};
    ASSERT_EQ(rj_kmd_server_drain_doorbells(server, &record, &count), 0);
    ASSERT_EQ(count, 1u);
    EXPECT_EQ(record.queue_id, queue.queue_id);
    EXPECT_EQ(record.offset, 0u);
    EXPECT_EQ(record.value, 0u);

    kfd_ioctl_destroy_queue_args destroy{};
    destroy.queue_id = queue.queue_id;
    EXPECT_EQ(remote.ioctl(AMDKFD_IOC_DESTROY_QUEUE, &destroy), 0);
    kfd_ioctl_free_memory_of_gpu_args release{.handle = allocation.handle};
    EXPECT_EQ(remote.ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &release), 0);
    syscall(SYS_munmap, doorbells, allocation.size);
    EXPECT_EQ(remote.close(), 0);
  }

  rj_kmd_server_destroy(server);
  syscall(SYS_close, fake.fd);
  std::filesystem::remove_all(runtime);
}

} // namespace
