// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kmd_rpc_server.h
/// @brief Reusable nonblocking rocjitsu KMD RPC reactor.

#ifndef ROCJITSU_KMD_LINUX_KMD_RPC_SERVER_H_
#define ROCJITSU_KMD_LINUX_KMD_RPC_SERVER_H_

#include "rocjitsu/kmd/linux/rpc.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {

class SimulatedDriver;

class KmdRpcServer {
public:
  struct PollDescriptor {
    int fd = -1;
    int16_t events = 0;
  };

  KmdRpcServer(SimulatedDriver &driver, const RpcGpuInfo &gpu_info);
  ~KmdRpcServer();

  KmdRpcServer(const KmdRpcServer &) = delete;
  KmdRpcServer &operator=(const KmdRpcServer &) = delete;

  bool listen(const std::string &socket_path, std::string *error = nullptr);
  void stopAccepting();
  void close();

  [[nodiscard]] int listenerFd() const;
  [[nodiscard]] std::vector<PollDescriptor> pollDescriptors() const;

  /// @brief Process readiness already reported by the embedding reactor.
  ///
  /// revents uses poll(2) flags. This method never waits for socket I/O.
  bool processFd(int fd, int16_t revents, std::string *error = nullptr);

  [[nodiscard]] size_t activeClients() const;
  [[nodiscard]] size_t activeProcesses() const;
  [[nodiscard]] size_t pendingResponses() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_KMD_RPC_SERVER_H_
