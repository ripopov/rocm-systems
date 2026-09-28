// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/kmd/server.h"
#include "scoped_temp.h"
#include "util/unique_handle.h"

#include <future>
#include <gtest/gtest.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>

namespace rocjitsu {
namespace {
template <typename T> T abi() {
  T value{};
  value.abi_version = RJ_KMD_SERVER_ABI_VERSION;
  value.struct_size = sizeof(T);
  return value;
}

class KmdEmbeddingTest : public testing::TestWithParam<uint32_t> {
protected:
  test::ScopedTempDirectory directory{"rj-c-embedding-"};
  const std::thread::id owner = std::this_thread::get_id();
  std::string socket_path = directory.path() + "/server.sock";
  rj_kmd_server_t *server = nullptr;
  int opens = 0, closes = 0, shutdowns = 0;

  static KmdEmbeddingTest &host(void *context) {
    auto &self = *static_cast<KmdEmbeddingTest *>(context);
    EXPECT_EQ(std::this_thread::get_id(), self.owner);
    return self;
  }
  rj_kmd_backend_ops_t operations() {
    auto ops = abi<rj_kmd_backend_ops_t>();
    ops.context = this;
    ops.open_process = [](void *c, const rj_kmd_process_request_t *) -> int32_t {
      ++host(c).opens;
      return 0;
    };
    ops.close_process = [](void *c, const rj_kmd_process_request_t *) -> int32_t {
      ++host(c).closes;
      return 0;
    };
    ops.set_process_client_pid = [](void *c, const rj_kmd_process_request_t *) -> int32_t {
      host(c);
      return 0;
    };
    ops.set_apertures = [](void *c, const rj_kmd_apertures_t *) -> int32_t {
      host(c);
      return 0;
    };
    ops.allocate = [](void *, const rj_kmd_allocation_request_t *,
                      rj_kmd_backing_slice_t *) -> int32_t { return -ENOTSUP; };
    ops.free = [](void *, const rj_kmd_free_request_t *) -> int32_t { return 0; };
    ops.map = [](void *, const rj_kmd_mapping_t *) -> int32_t { return -ENOTSUP; };
    ops.unmap = [](void *, const rj_kmd_unmap_request_t *) -> int32_t { return 0; };
    ops.create_queue = [](void *, const rj_kmd_queue_t *) -> int32_t { return -ENOTSUP; };
    ops.update_queue = [](void *, const rj_kmd_queue_update_t *) -> int32_t { return -ENOTSUP; };
    ops.destroy_queue = [](void *, const rj_kmd_queue_key_t *) -> int32_t { return 0; };
    ops.read64 = [](void *, const rj_kmd_read64_t *, uint64_t *) -> int32_t { return -ENOTSUP; };
    ops.write64 = [](void *, const rj_kmd_write64_t *) -> int32_t { return -ENOTSUP; };
    ops.shutdown = [](void *c) { ++host(c).shutdowns; };
    return ops;
  }
  void SetUp() override {
    auto info = abi<rj_kmd_device_info_t>();
    info.architecture = 3; // public rj_code_arch_t CDNA4 value
    info.gpu_id = 7;
    info.gfx_target_version = 90500;
    info.num_xcc = GetParam();
    info.drm_render_minor = 128;
    info.wave_front_size = 64;
    info.simd_per_cu = 4;
    info.max_waves_per_simd = 8;
    info.num_shader_engines = 1;
    info.num_shader_arrays_per_engine = 1;
    info.num_cu_per_sh = 1;
    auto config = abi<rj_kmd_server_config_t>();
    config.runtime_root = directory.path().c_str();
    config.socket_path = socket_path.c_str();
    auto ops = operations();
    char error[256]{};
    ASSERT_EQ(rj_kmd_server_create(&info, &config, &ops, &server, error, sizeof(error)), 0)
        << error;
  }
  void TearDown() override {
    rj_kmd_server_destroy(server);
    server = nullptr;
  }
  void pump_until(std::future<void> &task) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (task.wait_for(std::chrono::seconds(0)) != std::future_status::ready &&
           std::chrono::steady_clock::now() < deadline) {
      auto descriptor = abi<rj_kmd_poll_fd_t>();
      size_t count = 1;
      ASSERT_EQ(rj_kmd_server_poll_fds(server, &descriptor, &count), 0);
      pollfd fd{descriptor.fd, descriptor.events, 0};
      ASSERT_GE(::poll(&fd, 1, 25), 0);
      if (fd.revents) {
        char error[256]{};
        ASSERT_EQ(rj_kmd_server_process_fd(server, fd.fd, fd.revents, error, sizeof(error)), 0)
            << error;
      }
    }
    ASSERT_EQ(task.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    task.get();
  }
};

TEST_P(KmdEmbeddingTest, SharedLibraryServesHandshakeAndDestroysLiveClientOnOwner) {
  char error[256]{};
  ASSERT_EQ(rj_kmd_server_listen(server, error, sizeof(error)), 0) << error;
  util::UniqueHandle client(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  ASSERT_GE(client.get(), 0);
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strcpy(address.sun_path, socket_path.c_str());
  ASSERT_EQ(::connect(client.get(), reinterpret_cast<sockaddr *>(&address), sizeof(address)), 0);
  auto handshake = std::async(std::launch::async, [&] {
    RpcHeader request{};
    request.opcode = RPC_HANDSHAKE;
    ASSERT_TRUE(rpc_send_exact(client.get(), &request, sizeof(request)));
    RpcHeader response{};
    ASSERT_TRUE(rpc_recv_exact(client.get(), &response, sizeof(response)));
    ASSERT_EQ(response.result, 0);
    ASSERT_GE(response.payload_bytes, sizeof(RpcHandshakeResponse));
    std::vector<char> payload(response.payload_bytes);
    ASSERT_TRUE(rpc_recv_exact(client.get(), payload.data(), payload.size()));
    RpcHandshakeResponse info{};
    std::memcpy(&info, payload.data(), sizeof(info));
    EXPECT_EQ(info.gpu_id, 7u);
    EXPECT_EQ(info.version, kRpcProtocolVersion);
    const std::string topology(payload.data() + sizeof(info), info.topology_path_len);
    EXPECT_TRUE(std::filesystem::exists(topology));
    EXPECT_EQ(std::filesystem::path(topology).parent_path(), directory.path());
  });
  pump_until(handshake);
  EXPECT_EQ(opens, 1);
  auto stats = abi<rj_kmd_server_stats_t>();
  ASSERT_EQ(rj_kmd_server_get_stats(server, &stats), 0);
  EXPECT_EQ(stats.active_clients, 1u);
  EXPECT_EQ(stats.active_processes, 1u);
  rj_kmd_server_destroy(server);
  server = nullptr;
  EXPECT_EQ(closes, 1);
  EXPECT_EQ(shutdowns, 1);
  EXPECT_FALSE(std::filesystem::exists(socket_path));
}

TEST_P(KmdEmbeddingTest, RejectsWrongThreadAndInvalidDescriptors) {
  auto wrong_thread = std::async(std::launch::async, [&] {
    size_t count = 0;
    EXPECT_EQ(rj_kmd_server_poll_fds(server, nullptr, &count), -EPERM);
  });
  wrong_thread.get();
  char error[32]{};
  EXPECT_EQ(rj_kmd_server_process_fd(server, -1, POLLIN, error, sizeof(error)), -EINVAL);
  EXPECT_NE(error[0], '\0');
  auto stats = abi<rj_kmd_server_stats_t>();
  stats.abi_version = 1;
  EXPECT_EQ(rj_kmd_server_get_stats(server, &stats), -EINVAL);
  size_t count = 0;
  EXPECT_EQ(rj_kmd_server_poll_fds(server, nullptr, &count), 0);
  EXPECT_EQ(count, 1u);
}

INSTANTIATE_TEST_SUITE_P(Topology, KmdEmbeddingTest, testing::Values(1u, 8u));

} // namespace
} // namespace rocjitsu
