// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/kmd_rpc_server.h"

#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/simulated_driver.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <poll.h>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unordered_map>
#include <unistd.h>
#include <utility>

namespace rocjitsu {
namespace {

constexpr uint32_t kMaxPayloadBytes = 16 * 1024 * 1024;

bool setNonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

pid_t peerPid(int fd) {
  struct ucred credentials {};
  socklen_t length = sizeof(credentials);
  if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials, &length) == 0 &&
      credentials.pid > 0)
    return credentials.pid;
  return 0;
}

void reconstructEmbeddedPointers(uint32_t command, void *argument,
                                  size_t argument_size, size_t total_size) {
  if (total_size <= argument_size)
    return;
  auto *extra = static_cast<uint8_t *>(argument) + argument_size;
  switch (command) {
  case AMDKFD_IOC_WAIT_EVENTS:
    static_cast<kfd_ioctl_wait_events_args *>(argument)->events_ptr =
        reinterpret_cast<uint64_t>(extra);
    break;
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
    static_cast<kfd_ioctl_map_memory_to_gpu_args *>(argument)
        ->device_ids_array_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
    static_cast<kfd_ioctl_get_process_apertures_new_args *>(argument)
        ->kfd_process_device_apertures_ptr = reinterpret_cast<uint64_t>(extra);
    break;
  default:
    break;
  }
}

} // namespace

class KmdRpcServer::Impl {
public:
  struct Response {
    std::vector<uint8_t> bytes;
    size_t offset = 0;
    int fd = -1;
    bool fd_sent = false;
  };

  struct Client {
    int fd = -1;
    pid_t peer_pid = 0;
    uint32_t process_id = 0;
    std::vector<uint8_t> input;
    std::deque<Response> output;
    bool close_after_flush = false;
  };

  Impl(SimulatedDriver &driver, const RpcGpuInfo &gpu_info)
      : driver(driver), gpu_info(gpu_info) {}

  ~Impl() { close(); }

  void setError(std::string *target, const std::string &message) {
    if (target)
      *target = message;
  }

  void closeClient(int fd) {
    auto it = clients.find(fd);
    if (it == clients.end())
      return;
    if (it->second.process_id != 0)
      driver.close(it->second.process_id);
    ::close(fd);
    clients.erase(it);
  }

  void close() {
    std::vector<int> fds;
    fds.reserve(clients.size());
    for (const auto &[fd, client] : clients) {
      (void)client;
      fds.push_back(fd);
    }
    for (int fd : fds)
      closeClient(fd);
    stopAccepting();
    if (!socket_path.empty()) {
      unlink(socket_path.c_str());
      socket_path.clear();
    }
  }

  void stopAccepting() {
    if (listener >= 0) {
      ::close(listener);
      listener = -1;
    }
  }

  bool acceptClients(std::string *error) {
    while (true) {
      int client_fd = accept4(listener, nullptr, nullptr,
                              SOCK_NONBLOCK | SOCK_CLOEXEC);
      if (client_fd < 0) {
        if (errno == EINTR)
          continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK)
          return true;
        setError(error, std::string("accept failed: ") + strerror(errno));
        return false;
      }
      Client client{};
      client.fd = client_fd;
      client.peer_pid = peerPid(client_fd);
      clients.emplace(client_fd, std::move(client));
    }
  }

  void queueResponse(Client &client, const RpcHeader &header,
                     const void *payload = nullptr, size_t payload_size = 0,
                     int fd = -1) {
    Response response{};
    response.bytes.resize(sizeof(header) + payload_size);
    std::memcpy(response.bytes.data(), &header, sizeof(header));
    if (payload_size > 0)
      std::memcpy(response.bytes.data() + sizeof(header), payload, payload_size);
    response.fd = fd;
    client.output.push_back(std::move(response));
  }

  void queueHandshake(Client &client, const RpcHeader &request) {
    if (client.process_id == 0)
      client.process_id = driver.open_process(client.peer_pid);

    RpcHeader response{};
    response.opcode = RPC_HANDSHAKE;
    response.request_id = request.request_id;
    if (client.process_id == 0) {
      response.result = -EIO;
      queueResponse(client, response);
      client.close_after_flush = true;
      return;
    }

    const std::string topology_path = driver.topology_path();
    const std::string drm_path = driver.topology().drm_path();
    if (topology_path.size() > std::numeric_limits<uint32_t>::max() ||
        drm_path.size() > std::numeric_limits<uint32_t>::max()) {
      response.result = -EOVERFLOW;
      queueResponse(client, response);
      client.close_after_flush = true;
      return;
    }

    RpcHandshakeResponse handshake{};
    handshake.version = kRpcProtocolVersion;
    handshake.gpu_id = driver.gpu_id();
    handshake.topology_path_len =
        static_cast<uint32_t>(topology_path.size());
    handshake.drm_path_len = static_cast<uint32_t>(drm_path.size());
    handshake.gpu_info = gpu_info;

    std::vector<uint8_t> payload(sizeof(handshake) + topology_path.size() +
                                 drm_path.size());
    std::memcpy(payload.data(), &handshake, sizeof(handshake));
    std::memcpy(payload.data() + sizeof(handshake), topology_path.data(),
                topology_path.size());
    std::memcpy(payload.data() + sizeof(handshake) + topology_path.size(),
                drm_path.data(), drm_path.size());
    response.payload_bytes = static_cast<uint32_t>(payload.size());
    queueResponse(client, response, payload.data(), payload.size());
  }

  bool dispatch(Client &client, const RpcHeader &request,
                std::vector<uint8_t> payload, std::string *error) {
    RpcHeader response{};
    response.opcode = request.opcode;
    response.request_id = request.request_id;

    switch (request.opcode) {
    case RPC_HANDSHAKE:
      if (!payload.empty() || client.process_id != 0)
        return false;
      queueHandshake(client, request);
      return true;

    case RPC_CLOSE:
      if (!payload.empty())
        return false;
      if (client.process_id != 0) {
        driver.close(client.process_id);
        client.process_id = 0;
      }
      queueResponse(client, response);
      client.close_after_flush = true;
      return true;

    case RPC_MMAP: {
      if (client.process_id == 0 || payload.size() != sizeof(RpcMmapRequest))
        return false;
      RpcMmapRequest mmap_request{};
      std::memcpy(&mmap_request, payload.data(), sizeof(mmap_request));
      void *mapped = driver.mmap(
          client.process_id, reinterpret_cast<void *>(mmap_request.addr),
          static_cast<size_t>(mmap_request.length), mmap_request.prot,
          mmap_request.flags, static_cast<off_t>(mmap_request.offset));
      response.result = mapped == MAP_FAILED ? -errno : 0;
      response.payload_bytes = sizeof(RpcMmapResponse);
      SimulatedDriver::BackingDescriptor backing{};
      bool has_backing =
          response.result == 0 &&
          driver.get_mmap_backing(client.process_id, mmap_request.offset,
                                  backing);
      RpcMmapResponse mmap_response{
          .mapped_addr = reinterpret_cast<uint64_t>(mapped),
          .backing =
              {
                  .backing_offset = has_backing ? backing.offset : 0,
                  .backing_length = has_backing ? backing.length : 0,
              },
      };
      queueResponse(client, response, &mmap_response, sizeof(mmap_response),
                    has_backing ? backing.fd : -1);
      return true;
    }

    case RPC_MUNMAP: {
      if (client.process_id == 0 || payload.size() != sizeof(RpcMunmapRequest))
        return false;
      RpcMunmapRequest munmap_request{};
      std::memcpy(&munmap_request, payload.data(), sizeof(munmap_request));
      response.result = driver.munmap(
          client.process_id, reinterpret_cast<void *>(munmap_request.addr),
          static_cast<size_t>(munmap_request.length));
      queueResponse(client, response);
      return true;
    }

    case RPC_IOCTL: {
      if (client.process_id == 0 ||
          payload.size() < sizeof(RpcIoctlRequest))
        return false;
      RpcIoctlRequest ioctl_request{};
      std::memcpy(&ioctl_request, payload.data(), sizeof(ioctl_request));
      if (ioctl_request.args_bytes !=
          payload.size() - sizeof(RpcIoctlRequest))
        return false;
      auto *argument = payload.data() + sizeof(RpcIoctlRequest);
      size_t fixed_size = ioctl_arg_size(ioctl_request.ioctl_cmd);
      if (ioctl_request.args_bytes < fixed_size)
        return false;
      reconstructEmbeddedPointers(ioctl_request.ioctl_cmd, argument, fixed_size,
                                  ioctl_request.args_bytes);
      // RemoteDriver owns finite and infinite wait pacing. The embedded
      // reactor must never block gem5's event queue on a condition variable,
      // so each wire request is one nonblocking EventState observation.
      if (canonical_ioctl_request(ioctl_request.ioctl_cmd) ==
          AMDKFD_IOC_WAIT_EVENTS) {
        reinterpret_cast<kfd_ioctl_wait_events_args *>(argument)->timeout = 0;
      }
      response.result =
          driver.ioctl(client.process_id, ioctl_request.ioctl_cmd, argument);
      SimulatedDriver::BackingDescriptor backing{};
      bool has_backing = false;
      if (response.result == 0 &&
          ioctl_request.ioctl_cmd == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
        auto *allocation =
            reinterpret_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(argument);
        const off_t backing_key =
            allocation->flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR
                ? static_cast<off_t>(allocation->handle << 12)
                : static_cast<off_t>(allocation->mmap_offset);
        has_backing = driver.get_mmap_backing(
            client.process_id, backing_key, backing);
      } else if (response.result == 0 &&
                 ioctl_request.ioctl_cmd == AMDKFD_IOC_IPC_IMPORT_HANDLE) {
        auto *allocation =
            reinterpret_cast<kfd_ioctl_ipc_import_handle_args *>(argument);
        has_backing = driver.get_mmap_backing(
            client.process_id, static_cast<off_t>(allocation->mmap_offset),
            backing);
      } else if (
          response.result == 0 &&
          canonical_ioctl_request(ioctl_request.ioctl_cmd) ==
              AMDKFD_IOC_SVM) {
        auto *svm =
            reinterpret_cast<kfd_ioctl_svm_args *>(argument);
        if (svm->op == KFD_IOCTL_SVM_OP_SET_ATTR) {
          has_backing = driver.get_svm_backing(
              client.process_id, svm->start_addr, backing);
        }
      }
      if (has_backing) {
        RpcBackingDescriptor descriptor{
            .backing_offset = backing.offset,
            .backing_length = backing.length,
        };
        std::vector<uint8_t> response_payload(
            ioctl_request.args_bytes + sizeof(descriptor));
        std::memcpy(response_payload.data(), argument,
                    ioctl_request.args_bytes);
        std::memcpy(response_payload.data() + ioctl_request.args_bytes,
                    &descriptor, sizeof(descriptor));
        response.payload_bytes =
            static_cast<uint32_t>(response_payload.size());
        queueResponse(client, response, response_payload.data(),
                      response_payload.size(), backing.fd);
      } else {
        response.payload_bytes = ioctl_request.args_bytes;
        queueResponse(client, response, argument, ioctl_request.args_bytes);
      }
      return true;
    }

    default:
      setError(error, "unknown RPC opcode");
      return false;
    }
  }

  bool parseInput(Client &client, std::string *error) {
    while (client.input.size() >= sizeof(RpcHeader)) {
      RpcHeader header{};
      std::memcpy(&header, client.input.data(), sizeof(header));
      if (header.reserved != 0 || header.result != 0 ||
          header.payload_bytes > kMaxPayloadBytes) {
        setError(error, "invalid RPC header");
        return false;
      }
      size_t frame_size = sizeof(RpcHeader) + header.payload_bytes;
      if (client.input.size() < frame_size)
        return true;
      std::vector<uint8_t> payload(header.payload_bytes);
      if (!payload.empty())
        std::memcpy(payload.data(), client.input.data() + sizeof(RpcHeader),
                    payload.size());
      client.input.erase(client.input.begin(),
                         client.input.begin() + frame_size);
      if (!dispatch(client, header, std::move(payload), error)) {
        if (error && error->empty())
          *error = "invalid RPC payload";
        return false;
      }
      if (client.close_after_flush)
        return true;
    }
    return true;
  }

  bool readClient(Client &client, std::string *error,
                  bool *disconnected) {
    *disconnected = false;
    uint8_t buffer[64 * 1024];
    while (true) {
      ssize_t count = recv(client.fd, buffer, sizeof(buffer), 0);
      if (count > 0) {
        client.input.insert(client.input.end(), buffer, buffer + count);
        if (!parseInput(client, error))
          return false;
        continue;
      }
      if (count == 0) {
        *disconnected = true;
        return false;
      }
      if (errno == EINTR)
        continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        return true;
      setError(error, std::string("receive failed: ") + strerror(errno));
      return false;
    }
  }

  bool writeClient(Client &client, std::string *error) {
    while (!client.output.empty()) {
      auto &response = client.output.front();
      const auto *data = response.bytes.data() + response.offset;
      size_t size = response.bytes.size() - response.offset;
      ssize_t count;
      if (response.fd >= 0 && !response.fd_sent) {
        count = rpc_send_msg(client.fd, data, size, &response.fd, 1);
        if (count > 0)
          response.fd_sent = true;
      } else {
        count = send(client.fd, data, size, MSG_NOSIGNAL);
      }
      if (count > 0) {
        response.offset += static_cast<size_t>(count);
        if (response.offset == response.bytes.size())
          client.output.pop_front();
        continue;
      }
      if (count < 0 && errno == EINTR)
        continue;
      if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        return true;
      setError(error, std::string("send failed: ") + strerror(errno));
      return false;
    }
    return true;
  }

  SimulatedDriver &driver;
  RpcGpuInfo gpu_info{};
  int listener = -1;
  std::string socket_path;
  std::unordered_map<int, Client> clients;
};

KmdRpcServer::KmdRpcServer(SimulatedDriver &driver,
                           const RpcGpuInfo &gpu_info)
    : impl_(std::make_unique<Impl>(driver, gpu_info)) {}

KmdRpcServer::~KmdRpcServer() = default;

bool KmdRpcServer::listen(const std::string &socket_path, std::string *error) {
  if (impl_->listener >= 0 || socket_path.empty()) {
    impl_->setError(error, "server is already listening or path is empty");
    return false;
  }
  std::error_code filesystem_error;
  auto parent = std::filesystem::path(socket_path).parent_path();
  if (!parent.empty())
    std::filesystem::create_directories(parent, filesystem_error);
  if (filesystem_error) {
    impl_->setError(error, "cannot create socket directory: " +
                               filesystem_error.message());
    return false;
  }
  unlink(socket_path.c_str());

  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0 || !setNonblocking(fd)) {
    if (fd >= 0)
      ::close(fd);
    impl_->setError(error, std::string("cannot create listener: ") +
                               strerror(errno));
    return false;
  }
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(address.sun_path)) {
    ::close(fd);
    impl_->setError(error, "socket path is too long");
    return false;
  }
  std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
  if (bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
      ::listen(fd, 16) != 0) {
    int saved_errno = errno;
    ::close(fd);
    unlink(socket_path.c_str());
    impl_->setError(error, std::string("cannot bind listener: ") +
                               strerror(saved_errno));
    return false;
  }
  impl_->listener = fd;
  impl_->socket_path = socket_path;
  return true;
}

void KmdRpcServer::stopAccepting() { impl_->stopAccepting(); }

void KmdRpcServer::close() { impl_->close(); }

int KmdRpcServer::listenerFd() const { return impl_->listener; }

std::vector<KmdRpcServer::PollDescriptor>
KmdRpcServer::pollDescriptors() const {
  std::vector<PollDescriptor> descriptors;
  descriptors.reserve(impl_->clients.size() +
                      (impl_->listener >= 0 ? 1 : 0));
  if (impl_->listener >= 0)
    descriptors.push_back({impl_->listener, POLLIN});
  for (const auto &[fd, client] : impl_->clients) {
    int16_t events = POLLIN;
    if (!client.output.empty())
      events |= POLLOUT;
    descriptors.push_back({fd, events});
  }
  return descriptors;
}

bool KmdRpcServer::processFd(int fd, int16_t revents, std::string *error) {
  if (fd == impl_->listener) {
    if (revents & (POLLERR | POLLHUP | POLLNVAL)) {
      impl_->setError(error, "listener failed");
      return false;
    }
    return !(revents & POLLIN) || impl_->acceptClients(error);
  }

  auto it = impl_->clients.find(fd);
  if (it == impl_->clients.end()) {
    impl_->setError(error, "unknown reactor file descriptor");
    return false;
  }

  bool healthy = true;
  bool disconnected = false;
  if (revents & POLLIN)
    healthy = impl_->readClient(it->second, error, &disconnected);
  if (healthy && (revents & POLLOUT))
    healthy = impl_->writeClient(it->second, error);
  if (healthy && (revents & (POLLERR | POLLNVAL))) {
    impl_->setError(error, "client socket failed");
    healthy = false;
  }
  if (healthy && (revents & POLLHUP)) {
    disconnected = true;
    healthy = false;
  }

  bool close_after_flush =
      healthy && it->second.close_after_flush && it->second.output.empty();
  if (!healthy || disconnected || close_after_flush)
    impl_->closeClient(fd);
  return healthy || disconnected || close_after_flush;
}

size_t KmdRpcServer::activeClients() const { return impl_->clients.size(); }

size_t KmdRpcServer::activeProcesses() const {
  return std::count_if(
      impl_->clients.begin(), impl_->clients.end(),
      [](const auto &entry) { return entry.second.process_id != 0; });
}

size_t KmdRpcServer::pendingResponses() const {
  size_t responses = 0;
  for (const auto &[fd, client] : impl_->clients) {
    (void)fd;
    responses += client.output.size();
  }
  return responses;
}

} // namespace rocjitsu
