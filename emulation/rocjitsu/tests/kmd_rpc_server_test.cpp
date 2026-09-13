// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
class BlockingDevice final : public KmdRpcDevice {
public:
  std::atomic<unsigned> opens{0};
  std::atomic<unsigned> closes{0};
  std::mutex mutex;
  std::condition_variable changed;
  bool entered = false;
  bool closing = false;

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
  rj_status_t shared_mem(uint32_t, uint64_t, rj_handle_t *) override {
    return ROCJITSU_STATUS_UNSUPPORTED;
  }
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
  server.stop();
  EXPECT_EQ(server.status(), RJ_DAEMON_STATUS_STOPPED);
  EXPECT_EQ(device.closes.load(), 1u);
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

} // namespace
} // namespace rocjitsu
