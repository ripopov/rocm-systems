// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/callback_dispatcher.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/kmd_rpc_server.h"
#include "scoped_temp.h"
#include "util/unique_handle.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <gtest/gtest.h>
#include <mutex>
#include <sys/socket.h>
#include <sys/un.h>
#include <vector>

namespace rocjitsu {
namespace {

// Deliberately has no simulator: linking this test verifies the transport's
// independence from the VM, ISA execution, and configuration libraries.
class BlockingDevice : public KmdRpcDevice {
public:
  std::atomic<unsigned> opens{0};
  std::atomic<unsigned> closes{0};
  std::mutex mutex;
  std::condition_variable changed;
  bool entered = false;
  bool closing = false;
  uint32_t gpu_total = 1;
  bool strict = false;

  rj_status_t open(rj_client_pid_t pid, uint32_t *process) override {
    EXPECT_EQ(pid, getpid());
    *process = ++opens;
    return ROCJITSU_STATUS_SUCCESS;
  }
  rj_status_t close(uint32_t) override {
    ++closes;
    return ROCJITSU_STATUS_SUCCESS;
  }
  rj_status_t gpu_id(uint32_t *id) override {
    *id = 42;
    return ROCJITSU_STATUS_SUCCESS;
  }
  rj_status_t gpu_info(RpcGpuInfo *info) override {
    *info = {};
    info->gpu_id = 42;
    return ROCJITSU_STATUS_SUCCESS;
  }
  rj_status_t gpu_count(uint32_t *count) override {
    *count = gpu_total;
    return ROCJITSU_STATUS_SUCCESS;
  }
  rj_status_t gpu_info_at(uint32_t ordinal, RpcGpuInfo *info) override {
    if (ordinal >= gpu_total)
      return ROCJITSU_STATUS_INVALID_ARGUMENT;
    gpu_info(info);
    info->gpu_id += ordinal;
    info->visible_vram_size = uint64_t{ordinal + 1} << 30;
    info->control_stack_size = 4096;
    info->context_save_size = 8192;
    return ROCJITSU_STATUS_SUCCESS;
  }
  bool strict_render_minor() const override { return strict; }
  rj_status_t topology_path(const char **path) override {
    *path = "/test/topology";
    return ROCJITSU_STATUS_SUCCESS;
  }
  rj_status_t drm_path(const char **path) override {
    *path = "/test/drm";
    return ROCJITSU_STATUS_SUCCESS;
  }
  rj_status_t map(uint32_t, rj_vm_map_t *) override { return ROCJITSU_STATUS_UNSUPPORTED; }
  rj_status_t unmap(uint32_t, rj_vm_unmap_t *) override { return ROCJITSU_STATUS_UNSUPPORTED; }
  rj_status_t execute(uint32_t, rj_vm_cmd_t *request) override {
    std::unique_lock lock(mutex);
    entered = true;
    changed.notify_all();
    changed.wait(lock, [&] { return closing; });
    request->result = -ESRCH;
    return ROCJITSU_STATUS_SUCCESS;
  }
  void close_all() override {
    std::lock_guard lock(mutex);
    closing = true;
    changed.notify_all();
  }
};

class KmdRpcServerTest : public testing::Test {
protected:
  test::ScopedTempDirectory directory{"rj-rpc-test-"};
  std::string path = directory.path() + "/server.sock";
  BlockingDevice device;
  KmdRpcServer server{device};

  util::UniqueHandle connect_client() {
    util::UniqueHandle client(socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strcpy(address.sun_path, path.c_str());
    if (connect(client.get(), reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0)
      return {};
    return client;
  }

  void handshake(int client) {
    RpcHeader request{};
    request.opcode = RPC_HANDSHAKE;
    ASSERT_TRUE(rpc_send_exact(client, &request, sizeof(request)));
    RpcHeader response{};
    ASSERT_TRUE(rpc_recv_exact(client, &response, sizeof(response)));
    ASSERT_EQ(response.result, 0);
    ASSERT_GE(response.payload_bytes, sizeof(RpcHandshakeResponse));
    std::vector<uint8_t> payload(response.payload_bytes);
    ASSERT_TRUE(rpc_recv_exact(client, payload.data(), payload.size()));
    RpcHandshakeResponse info{};
    std::memcpy(&info, payload.data(), sizeof(info));
    EXPECT_EQ(info.version, kRpcProtocolVersion);
    EXPECT_EQ(info.gpu_id, 42u);
  }
};

TEST_F(KmdRpcServerTest, HandshakePreservesMultiGpuMetadataAndRouting) {
  device.gpu_total = 3;
  device.strict = true;
  ASSERT_EQ(server.bind(path.c_str()), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(server.start(), ROCJITSU_STATUS_SUCCESS);
  auto client = connect_client();
  ASSERT_GE(client.get(), 0);
  RpcHeader request{};
  request.opcode = RPC_HANDSHAKE;
  ASSERT_TRUE(rpc_send_exact(client.get(), &request, sizeof(request)));
  RpcHeader response{};
  ASSERT_TRUE(rpc_recv_exact(client.get(), &response, sizeof(response)));
  ASSERT_EQ(response.result, 0);
  ASSERT_GE(response.payload_bytes, sizeof(RpcHandshakeResponse));
  std::vector<uint8_t> payload(response.payload_bytes);
  ASSERT_TRUE(rpc_recv_exact(client.get(), payload.data(), payload.size()));
  RpcHandshakeResponse info{};
  std::memcpy(&info, payload.data(), sizeof(info));
  EXPECT_EQ(info.version, 7u);
  EXPECT_EQ(info.extra_gpu_infos, 2u);
  EXPECT_EQ(info.strict_render_minor, 1u);
  size_t offset = sizeof(info) + info.topology_path_len + info.drm_path_len;
  ASSERT_EQ(payload.size(), offset + 2 * sizeof(RpcGpuInfo));
  EXPECT_EQ(
      std::string(reinterpret_cast<char *>(payload.data() + sizeof(info)), info.topology_path_len),
      "/test/topology");
  for (uint32_t ordinal = 1; ordinal < device.gpu_total; ++ordinal) {
    RpcGpuInfo gpu{};
    std::memcpy(&gpu, payload.data() + offset, sizeof(gpu));
    EXPECT_EQ(gpu.gpu_id, 42u + ordinal);
    EXPECT_EQ(gpu.visible_vram_size, uint64_t{ordinal + 1} << 30);
    EXPECT_EQ(gpu.control_stack_size, 4096u);
    EXPECT_EQ(gpu.context_save_size, 8192u);
    offset += sizeof(gpu);
  }
  server.stop();
}

TEST_F(KmdRpcServerTest, StopDrainsBlockedDeviceOperation) {
  ASSERT_EQ(server.bind(path.c_str()), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(server.start(), ROCJITSU_STATUS_SUCCESS);
  auto client = connect_client();
  ASSERT_GE(client.get(), 0);
  handshake(client.get());
  kfd_ioctl_wait_events_args args{};
  args.timeout = 0xFFFFFFFFu;
  RpcIoctlRequest ioctl_request{};
  ioctl_request.ioctl_cmd = AMDKFD_IOC_WAIT_EVENTS;
  ioctl_request.args_bytes = sizeof(args);
  RpcHeader header{};
  header.opcode = RPC_IOCTL;
  header.payload_bytes = sizeof(ioctl_request) + sizeof(args);
  ASSERT_TRUE(rpc_send_exact(client.get(), &header, sizeof(header)));
  ASSERT_TRUE(rpc_send_exact(client.get(), &ioctl_request, sizeof(ioctl_request)));
  ASSERT_TRUE(rpc_send_exact(client.get(), &args, sizeof(args)));
  {
    std::unique_lock lock(device.mutex);
    ASSERT_TRUE(
        device.changed.wait_for(lock, std::chrono::seconds(5), [&] { return device.entered; }));
  }
  EXPECT_EQ(server.statistics().active_clients, 1u);
  EXPECT_EQ(server.statistics().pending_requests, 1u);
  server.stop_accepting();
  EXPECT_EQ(server.status(), RJ_DAEMON_STATUS_RUNNING);
  server.stop();
  EXPECT_EQ(server.status(), RJ_DAEMON_STATUS_STOPPED);
  EXPECT_EQ(device.closes.load(), 1u);
  EXPECT_EQ(server.statistics().active_clients, 0u);
  EXPECT_EQ(server.statistics().pending_requests, 0u);
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(KmdRpcServerTest, CompetingServerCannotRemoveLiveSocket) {
  ASSERT_EQ(server.bind(path.c_str()), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(server.start(), ROCJITSU_STATUS_SUCCESS);
  {
    BlockingDevice other_device;
    KmdRpcServer other{other_device};
    EXPECT_EQ(other.bind(path.c_str()), ROCJITSU_STATUS_ERROR);
  }
  ASSERT_TRUE(std::filesystem::exists(path));
  auto client = connect_client();
  ASSERT_GE(client.get(), 0);
  handshake(client.get());
}

TEST_F(KmdRpcServerTest, StopAcceptingPreservesEstablishedSessions) {
  ASSERT_EQ(server.bind(path.c_str()), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(server.start(), ROCJITSU_STATUS_SUCCESS);
  auto first = connect_client();
  auto second = connect_client();
  ASSERT_GE(first.get(), 0);
  ASSERT_GE(second.get(), 0);
  handshake(first.get());
  handshake(second.get());

  server.stop_accepting();
  server.stop_accepting();
  EXPECT_EQ(server.status(), RJ_DAEMON_STATUS_RUNNING);
  EXPECT_LT(connect_client().get(), 0);
  EXPECT_EQ(device.opens.load(), 2u);
  EXPECT_EQ(device.closes.load(), 0u);
  {
    std::lock_guard lock(device.mutex);
    EXPECT_FALSE(device.closing);
  }
  // Each pre-existing session must still execute requests and reply normally.
  for (int fd : {first.get(), second.get()}) {
    RpcHeader request{};
    request.opcode = RPC_CLOSE;
    request.request_id = 17;
    ASSERT_TRUE(rpc_send_exact(fd, &request, sizeof(request)));
    RpcHeader response{};
    ASSERT_TRUE(rpc_recv_exact(fd, &response, sizeof(response)));
    EXPECT_EQ(response.opcode, RPC_CLOSE);
    EXPECT_EQ(response.request_id, 17u);
    EXPECT_EQ(response.result, 0);
  }
  EXPECT_EQ(device.closes.load(), 2u);
  server.stop();
  EXPECT_EQ(server.status(), RJ_DAEMON_STATUS_STOPPED);
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(KmdRpcServerTest, QuiescedListenerCannotBeStarted) {
  ASSERT_EQ(server.bind(path.c_str()), ROCJITSU_STATUS_SUCCESS);
  server.stop_accepting();
  EXPECT_EQ(server.start(), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_LT(connect_client().get(), 0);
  server.stop();
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(KmdRpcServerTest, TransportWakeReportsRejectedRequestAndDisconnectWithoutDeviceCallback) {
  CallbackDispatcher dispatcher;
  KmdRpcServer observed{device, [&] { dispatcher.wake(); }};
  ASSERT_EQ(observed.bind(path.c_str()), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(observed.start(), ROCJITSU_STATUS_SUCCESS);
  auto client = connect_client();
  ASSERT_GE(client.get(), 0);
  handshake(client.get());
  EXPECT_EQ(observed.statistics().active_clients, 1u);
  dispatcher.dispatch();
  RpcHeader invalid{};
  invalid.reserved = 1;
  ASSERT_TRUE(rpc_send_exact(client.get(), &invalid, sizeof(invalid)));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  do {
    pollfd notification{dispatcher.fd(), POLLIN, 0};
    ASSERT_EQ(::poll(&notification, 1, 5000), 1);
    dispatcher.dispatch();
    const auto state = observed.statistics();
    if (!state.active_clients) {
      EXPECT_EQ(state.pending_requests, 0u);
      EXPECT_EQ(device.closes.load(), 1u);
      return;
    }
  } while (std::chrono::steady_clock::now() < deadline);
  FAIL() << "transport did not publish a drained disconnect";
}

TEST_F(KmdRpcServerTest, OwnerDispatchesCleanupWhileTransportJoinsBlockedClients) {
  CallbackDispatcher dispatcher;
  const auto owner = std::this_thread::get_id();
  class OwnerDevice final : public BlockingDevice {
  public:
    OwnerDevice(CallbackDispatcher &dispatcher, std::thread::id owner)
        : dispatcher_(dispatcher), owner_(owner) {}
    rj_status_t open(rj_client_pid_t pid, uint32_t *process) override {
      return dispatcher_.invoke([&] {
        EXPECT_EQ(std::this_thread::get_id(), owner_);
        return BlockingDevice::open(pid, process);
      });
    }
    rj_status_t close(uint32_t process) override {
      return dispatcher_.invoke([&] {
        EXPECT_EQ(std::this_thread::get_id(), owner_);
        return BlockingDevice::close(process);
      });
    }
    void close_all() override {
      dispatcher_.invoke([&] {
        EXPECT_EQ(std::this_thread::get_id(), owner_);
        BlockingDevice::close_all();
      });
    }

  private:
    CallbackDispatcher &dispatcher_;
    std::thread::id owner_;
  } embedded_device{dispatcher, owner};
  std::atomic<unsigned> notifications{0};
  KmdRpcServer embedded_server{embedded_device, [&] {
                                 ++notifications;
                                 dispatcher.wake();
                               }};
  ASSERT_EQ(embedded_server.bind(path.c_str()), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(embedded_server.start(), ROCJITSU_STATUS_SUCCESS);
  auto client = connect_client();
  ASSERT_GE(client.get(), 0);
  dispatcher.run_while_dispatching([&] { handshake(client.get()); });

  kfd_ioctl_wait_events_args args{};
  args.timeout = 0xFFFFFFFFu;
  RpcIoctlRequest request{};
  request.ioctl_cmd = AMDKFD_IOC_WAIT_EVENTS;
  request.args_bytes = sizeof(args);
  RpcHeader header{};
  header.opcode = RPC_IOCTL;
  header.payload_bytes = sizeof(request) + sizeof(args);
  ASSERT_TRUE(rpc_send_exact(client.get(), &header, sizeof(header)));
  ASSERT_TRUE(rpc_send_exact(client.get(), &request, sizeof(request)));
  ASSERT_TRUE(rpc_send_exact(client.get(), &args, sizeof(args)));
  {
    std::unique_lock lock(embedded_device.mutex);
    ASSERT_TRUE(embedded_device.changed.wait_for(lock, std::chrono::seconds(5),
                                                 [&] { return embedded_device.entered; }));
  }
  embedded_server.stop_accepting();
  dispatcher.run_while_dispatching([&] { embedded_server.stop(); });
  EXPECT_EQ(embedded_server.status(), RJ_DAEMON_STATUS_STOPPED);
  EXPECT_EQ(embedded_device.opens.load(), 1u);
  EXPECT_EQ(embedded_device.closes.load(), 1u);
  EXPECT_GT(notifications.load(), 0u);
  EXPECT_EQ(embedded_server.statistics().active_clients, 0u);
  EXPECT_EQ(embedded_server.statistics().pending_requests, 0u);
}

} // namespace
} // namespace rocjitsu
