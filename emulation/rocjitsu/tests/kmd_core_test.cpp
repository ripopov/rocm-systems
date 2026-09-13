// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/kfd_rpc_device.h"
#include "rocjitsu/kmd/linux/kmd_command.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "scoped_temp.h"
#include "util/unique_handle.h"

#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

class TestDeviceBackend : public rocjitsu::kmd::DeviceBackend {
public:
  int registration_result = 0;
  int queue_result = 0;
  int live_processes = 0;
  rj_code_arch_t architecture() const override { return ROCJITSU_CODE_ARCH_CDNA4; }
  void configure(const rocjitsu::kmd::DeviceApertures &, uint32_t,
                 const rocjitsu::kmd::DeviceCallbacks &) override {}
  void set_memory_fault_reporter(rocjitsu::kmd::MemoryFaultReporter *) override {}
  int register_process(rocjitsu::KfdProcess &, bool) override {
    if (registration_result == 0)
      ++live_processes;
    return registration_result;
  }
  void unregister_process(uint32_t) override { --live_processes; }
  void set_process_client_pid(uint32_t, int32_t) override {}
  void set_process_mem_fd(uint32_t, int) override {}
  std::optional<uint32_t> queue_xcc(uint32_t) const override { return 0; }
  int create_queue(const rocjitsu::kmd::QueueDescriptor &) override { return queue_result; }
  void update_queue(uint32_t, uint32_t, uint64_t, uint32_t, uint32_t) override {}
  void destroy_queue(uint32_t, uint32_t) override {}
  void set_doorbell_base(uint32_t, void *) override {}
  uint64_t read64(uint32_t, uint64_t) override { return 0; }
  void write64(uint32_t, uint64_t, uint64_t) override {}
};

TEST(KmdCoreTest, DeviceRegistrationFailureRollsBackEarlierDevices) {
  auto first = std::make_unique<TestDeviceBackend>();
  auto second = std::make_unique<TestDeviceBackend>();
  auto *first_state = first.get();
  auto *second_state = second.get();
  second->registration_result = -EIO;
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(first)});
  devices.push_back({8, std::move(second)});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  errno = 0;
  EXPECT_EQ(driver.open_process(), 0);
  EXPECT_EQ(errno, EIO);
  EXPECT_EQ(first_state->live_processes, 0);
  EXPECT_EQ(second_state->live_processes, 0);
  second_state->registration_result = 0;
  const auto process = driver.open_process();
  ASSERT_NE(process, 0);
  EXPECT_EQ(first_state->live_processes, 1);
  EXPECT_EQ(second_state->live_processes, 1);
  EXPECT_EQ(driver.close(process), 0);
  EXPECT_EQ(first_state->live_processes, 0);
  EXPECT_EQ(second_state->live_processes, 0);
}

TEST(KmdCoreTest, DeviceWithoutDebuggerRejectsDebugTrap) {
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::make_unique<TestDeviceBackend>()});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto process = driver.open_process();
  ASSERT_NE(process, 0);
  kfd_ioctl_dbg_trap_args args{};
  args.pid = getpid();
  args.op = KFD_IOC_DBG_TRAP_ENABLE;
  EXPECT_EQ(driver.ioctl(process, AMDKFD_IOC_DBG_TRAP, &args), -ENOTSUP);
  EXPECT_EQ(driver.close(process), 0);
}

TEST(KmdCoreTest, DeviceQueueFailureReleasesDoorbellReservation) {
  auto backend = std::make_unique<TestDeviceBackend>();
  auto *state = backend.get();
  state->queue_result = -ENOTSUP;
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto process_id = driver.open_process();
  ASSERT_NE(process_id, 0);
  kfd_ioctl_create_queue_args args{};
  args.gpu_id = 7;
  args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  EXPECT_EQ(driver.ioctl(process_id, AMDKFD_IOC_CREATE_QUEUE, &args), -ENOTSUP);
  const auto failed_doorbell = args.doorbell_offset;
  auto process = driver.find_process(process_id);
  ASSERT_NE(process, nullptr);
  EXPECT_TRUE(process->active_queue_ids_.empty());
  EXPECT_TRUE(process->queue_doorbell_map_.empty());
  EXPECT_TRUE(process->queue_snapshot_map_.empty());
  ASSERT_EQ(process->gpu(0).free_doorbell_offsets.size(), 1);
  state->queue_result = 0;
  ASSERT_EQ(driver.ioctl(process_id, AMDKFD_IOC_CREATE_QUEUE, &args), 0);
  EXPECT_EQ(args.doorbell_offset, failed_doorbell);
  EXPECT_TRUE(process->gpu(0).free_doorbell_offsets.empty());
  EXPECT_EQ(driver.close(process_id), 0);
}

TEST(KmdCoreTest, SharedCommandExecutorValidatesInlinePayload) {
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::make_unique<TestDeviceBackend>()});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto process = driver.open_process();
  ASSERT_NE(process, 0);

  kfd_ioctl_wait_events_args wait{};
  wait.num_events = 1;
  wait.events_ptr = 0xdeadbeef;
  rj_vm_cmd_t command{};
  command.cmd = AMDKFD_IOC_WAIT_EVENTS;
  command.buf = &wait;
  command.buf_size = sizeof(wait); // No inline event array accompanies the header.
  EXPECT_EQ(rocjitsu::execute_kmd_command(&driver, process, &command), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(command.result, -EINVAL);
  EXPECT_EQ(command.shared_handle, -1);
  EXPECT_EQ(wait.events_ptr, 0xdeadbeef);

  kfd_ioctl_get_version_args version{};
  command.cmd = AMDKFD_IOC_GET_VERSION;
  command.buf = &version;
  command.buf_size = sizeof(version);
  EXPECT_EQ(rocjitsu::execute_kmd_command(&driver, process, &command), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(command.result, 0);
  EXPECT_GT(version.major_version, 0);
  EXPECT_EQ(driver.close(process), 0);
}

TEST(KmdCoreTest, SocketServesCommonKfdWithoutSimulator) {
  using namespace rocjitsu;
  test::ScopedTempDirectory directory{"rj-core-rpc-"};
  const auto path = directory.path() + "/server.sock";
  auto backend = std::make_unique<TestDeviceBackend>();
  auto *state = backend.get();
  std::vector<SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  SimulatedKfd driver(std::move(devices));
  Sysfs::GpuInfo gpu{};
  gpu.gpu_id = 7;
  driver.setup_topology(gpu);
  KfdRpcDevice device(driver);
  KmdRpcServer server(device);
  ASSERT_EQ(server.bind(path.c_str()), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(server.start(), ROCJITSU_STATUS_SUCCESS);
  util::UniqueHandle client(socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  ASSERT_GE(client.get(), 0);
  timeval timeout{2, 0};
  ASSERT_EQ(setsockopt(client.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  ASSERT_LT(path.size(), sizeof(address.sun_path));
  std::strcpy(address.sun_path, path.c_str());
  ASSERT_EQ(connect(client.get(), reinterpret_cast<sockaddr *>(&address), sizeof(address)), 0);

  RpcHeader request{};
  request.opcode = RPC_HANDSHAKE;
  ASSERT_TRUE(rpc_send_exact(client.get(), &request, sizeof(request)));
  RpcHeader response{};
  ASSERT_TRUE(rpc_recv_exact(client.get(), &response, sizeof(response)));
  ASSERT_EQ(response.result, 0);
  ASSERT_GE(response.payload_bytes, sizeof(RpcHandshakeResponse));
  ASSERT_LE(response.payload_bytes, kMaxPayloadBytes);
  std::vector<uint8_t> payload(response.payload_bytes);
  ASSERT_TRUE(rpc_recv_exact(client.get(), payload.data(), payload.size()));
  RpcHandshakeResponse handshake{};
  std::memcpy(&handshake, payload.data(), sizeof(handshake));
  EXPECT_EQ(handshake.gpu_id, 7u);
  EXPECT_EQ(handshake.version, kRpcProtocolVersion);

  kfd_ioctl_get_version_args version{};
  RpcIoctlRequest ioctl_request{};
  ioctl_request.ioctl_cmd = AMDKFD_IOC_GET_VERSION;
  ioctl_request.args_bytes = sizeof(version);
  request.opcode = RPC_IOCTL;
  request.request_id = 42;
  request.payload_bytes = sizeof(ioctl_request) + sizeof(version);
  ASSERT_TRUE(rpc_send_exact(client.get(), &request, sizeof(request)));
  ASSERT_TRUE(rpc_send_exact(client.get(), &ioctl_request, sizeof(ioctl_request)));
  ASSERT_TRUE(rpc_send_exact(client.get(), &version, sizeof(version)));
  ASSERT_TRUE(rpc_recv_exact(client.get(), &response, sizeof(response)));
  ASSERT_EQ(response.result, 0);
  EXPECT_EQ(response.request_id, 42u);
  ASSERT_EQ(response.payload_bytes, sizeof(version));
  ASSERT_TRUE(rpc_recv_exact(client.get(), &version, sizeof(version)));
  EXPECT_GT(version.major_version, 0u);
  server.stop();
  EXPECT_EQ(server.status(), RJ_DAEMON_STATUS_STOPPED);
  EXPECT_EQ(state->live_processes, 0);
}

void check_allocation_failure(int resource, int expected_errno) {
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::make_unique<TestDeviceBackend>()});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto pid = driver.open_process();
  if (!pid)
    _exit(1);
  const auto process = driver.find_process(pid);
  const auto initial_va = process->next_gpu_va_;
  const auto initial_handle = process->next_handle_;
  auto fd_count = [] {
    unsigned count = 0;
    for (int fd = 0; fd < 8192; ++fd)
      if (fcntl(fd, F_GETFD) >= 0)
        ++count;
    return count;
  };
  const auto initial_fds = fd_count();
  if (resource == RLIMIT_FSIZE)
    std::signal(SIGXFSZ, SIG_IGN);
  rlimit limit{};
  if (setrlimit(resource, &limit) != 0)
    _exit(2);
  kfd_ioctl_alloc_memory_of_gpu_args args{};
  args.gpu_id = 7;
  args.size = 4096;
  args.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  args.handle = 0xdeadbeef;
  const int result = driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args);
  const bool clean = result == -expected_errno && process->allocations_.empty() &&
                     process->next_gpu_va_ == initial_va &&
                     process->next_handle_ == initial_handle && args.handle == 0xdeadbeef &&
                     fd_count() == initial_fds;
  _exit(clean ? 0 : 3);
}

TEST(KmdCoreTest, AllocationFdExhaustionDoesNotPublishState) {
  EXPECT_EXIT(check_allocation_failure(RLIMIT_NOFILE, EMFILE), testing::ExitedWithCode(0), "");
}

TEST(KmdCoreTest, AllocationSizingFailureClosesBacking) {
  EXPECT_EXIT(check_allocation_failure(RLIMIT_FSIZE, EFBIG), testing::ExitedWithCode(0), "");
}

TEST(KmdCoreTest, AllocationMappingFailureClosesBacking) {
  EXPECT_EXIT(check_allocation_failure(RLIMIT_AS, ENOMEM), testing::ExitedWithCode(0), "");
}

TEST(KmdCoreTest, AllocationRejectsZeroAndOverflowSizes) {
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::make_unique<TestDeviceBackend>()});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto pid = driver.open_process();
  ASSERT_NE(pid, 0);
  kfd_ioctl_alloc_memory_of_gpu_args args{};
  args.gpu_id = 7;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args), -EINVAL);
  args.size = UINT64_MAX;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args), -EINVAL);
  args.size = 4096;
  args.va_addr = UINT64_MAX - 2048;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args), -EINVAL);
  EXPECT_TRUE(driver.find_process(pid)->allocations_.empty());
  EXPECT_EQ(driver.close(pid), 0);
}

} // namespace
