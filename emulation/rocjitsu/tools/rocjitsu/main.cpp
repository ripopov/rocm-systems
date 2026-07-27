// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file main.cpp
/// @brief rocjitsu CLI — launcher for GPU simulation via LD_PRELOAD interposition.
///
/// @details Supports three usage patterns:
///   rocjitsu --config foo.json -- ./app           (local mode: in-process simulation)
///   rocjitsu --daemon --config foo.json -- ./app  (daemon mode: fork daemon + launch app)
///   rocjitsu --daemon --config foo.json           (daemon-only: run daemon server)

#include "rocjitsu/vm/rj_vm.h"

#include "rocjitsu/kmd/linux/kmd_rpc_server.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/version.h"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <string_view>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace rocjitsu;

static volatile sig_atomic_t g_stop_requested = 0;

static int run_daemon_server(const char *config_path) {
  rj_vm_t *vm = nullptr;
  if (rj_vm_create(config_path, RJ_VM_MODE_DAEMON, &vm) != ROCJITSU_STATUS_SUCCESS) {
    std::cerr << std::format("rocjitsu: failed to create VM from {}\n", config_path);
    return 1;
  }

  std::jthread engine_thread([vm]() { rj_vm_run(vm, nullptr); });

  RpcGpuInfo gpu_info{};
  if (!vm->vm || !vm->vm->driver() ||
      rj_vm_gpu_info(vm, &gpu_info) != ROCJITSU_STATUS_SUCCESS) {
    rj_vm_request_exit(vm, "KMD server initialization failed");
    engine_thread.join();
    rj_vm_destroy(vm);
    return 1;
  }

  KmdRpcServer server(*vm->vm->driver(), gpu_info);
  std::string server_error;
  if (!server.listen(rpc_default_socket_path(), &server_error)) {
    std::cerr << std::format("rocjitsu: {}\n", server_error);
    rj_vm_request_exit(vm, "KMD server listen failed");
    engine_thread.join();
    rj_vm_destroy(vm);
    return 1;
  }

  g_stop_requested = 0;
  std::signal(SIGINT, [](int) { g_stop_requested = 1; });
  std::signal(SIGTERM, [](int) { g_stop_requested = 1; });

  while (!g_stop_requested) {
    auto descriptors = server.pollDescriptors();
    std::vector<pollfd> poll_fds;
    poll_fds.reserve(descriptors.size());
    for (const auto &descriptor : descriptors)
      poll_fds.push_back({descriptor.fd, descriptor.events, 0});

    int ready = poll(poll_fds.data(), poll_fds.size(), 100);
    if (ready < 0 && errno == EINTR)
      continue;
    if (ready < 0)
      break;
    for (const auto &poll_fd : poll_fds) {
      if (poll_fd.revents == 0)
        continue;
      server_error.clear();
      if (!server.processFd(poll_fd.fd, poll_fd.revents, &server_error) &&
          poll_fd.fd == server.listenerFd()) {
        std::cerr << std::format("rocjitsu: KMD listener failed: {}\n",
                                 server_error);
        g_stop_requested = 1;
        break;
      }
    }
  }
  server.close();

  rj_vm_request_exit(vm, "daemon shutdown");
  engine_thread.join();
  rj_vm_destroy(vm);
  return 0;
}

static std::string find_interposer_lib() {
  char self[4096];
  auto n = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (n <= 0)
    return {};
  self[n] = '\0';
  auto bin_dir = std::filesystem::path(self).parent_path();
  // Installed layout: <prefix>/bin/rocjitsu → <prefix>/lib/librocjitsu.so
  //                   or <prefix>/bin/rocjitsu → <prefix>/lib64/librocjitsu.so
  // Build layout: build/tools/rocjitsu/rocjitsu → build/librocjitsu.so
  for (auto &candidate : {
           bin_dir / ".." / "lib" / "librocjitsu.so",
           bin_dir / ".." / "lib64" / "librocjitsu.so",
           bin_dir / ".." / ".." / "librocjitsu.so",
       }) {
    if (std::filesystem::exists(candidate))
      return std::filesystem::canonical(candidate).string();
  }
  return {};
}

static bool write_config_file(const std::string &config_path) {
  auto cfg_file = rpc_default_config_file_path();
  std::filesystem::create_directories(std::filesystem::path(cfg_file).parent_path());
  std::ofstream ofs(cfg_file);
  if (!ofs)
    return false;
  ofs << config_path << '\n';
  return ofs.good();
}

static void cleanup_runtime_files() {
  auto cfg_file = rpc_default_config_file_path();
  unlink(cfg_file.c_str());
  auto sock_file = rpc_default_socket_path();
  unlink(sock_file.c_str());
}

static void print_usage() {
  std::cerr
      << "Usage: rocjitsu --config <config.json> [--daemon|--attach] -- <app> [args...]\n"
         "\n"
         "Modes:\n"
         "  rocjitsu --config foo.json -- ./app          Local mode (in-process simulation)\n"
         "  rocjitsu --daemon --config foo.json -- ./app Daemon mode (fork daemon + launch app)\n"
         "  rocjitsu --daemon --config foo.json          Daemon-only (run server)\n"
         "  rocjitsu --attach -- ./app                 Attach to running daemon\n"
         "\n"
         "Options:\n"
         "  --config <path>   Simulation config JSON (required except for --attach)\n"
         "  --version, -v     Print version and exit\n"
         "  --help, -h        Print this help and exit\n";
}

int main(int argc, char *argv[]) {
  std::signal(SIGPIPE, SIG_IGN);

  const char *config_path = nullptr;
  bool daemon_mode = false;
  bool attach_mode = false;
  int separator_idx = -1;

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--") {
      separator_idx = i;
      break;
    }
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--daemon") {
      daemon_mode = true;
    } else if (arg == "--attach") {
      attach_mode = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else if (arg == "--version" || arg == "-v") {
      std::cout << "rocjitsu " << ROCJITSU_VERSION << "\n";
      return 0;
    } else {
      std::cerr << std::format("rocjitsu: unknown option: {}\n", arg);
      print_usage();
      return 1;
    }
  }

  if (!config_path && !attach_mode) {
    std::cerr << "rocjitsu: --config is required\n";
    print_usage();
    return 1;
  }

  std::string abs_config;
  if (config_path)
    abs_config = std::filesystem::absolute(config_path).string();
  if (!abs_config.empty() && !std::filesystem::exists(abs_config)) {
    std::cerr << std::format("rocjitsu: config file not found: {}\n", abs_config);
    return 1;
  }

  bool has_app = (separator_idx >= 0 && separator_idx + 1 < argc);

  if (daemon_mode && !has_app)
    return run_daemon_server(abs_config.c_str());

  if (!has_app) {
    std::cerr << "rocjitsu: no application specified after --\n";
    print_usage();
    return 1;
  }

  char **app_argv = &argv[separator_idx + 1];

  auto lib_path = find_interposer_lib();
  if (lib_path.empty()) {
    std::cerr << "rocjitsu: could not find librocjitsu.so\n";
    return 1;
  }

  if (attach_mode) {
    auto sock_path = rpc_default_socket_path();
    if (!std::filesystem::exists(sock_path)) {
      std::cerr << std::format("rocjitsu: no daemon socket at {}\n", sock_path);
      return 1;
    }
  } else if (daemon_mode) {
    pid_t daemon_pid = fork();
    if (daemon_pid < 0) {
      std::cerr << std::format("rocjitsu: fork failed: {}\n", strerror(errno));
      return 1;
    }

    if (daemon_pid == 0) {
      prctl(PR_SET_PDEATHSIG, SIGTERM);
      return run_daemon_server(abs_config.c_str());
    }

    auto sock_path = rpc_default_socket_path();
    for (int i = 0; i < 300; ++i) {
      if (std::filesystem::exists(sock_path))
        break;
      usleep(10000);
    }
    if (!std::filesystem::exists(sock_path)) {
      std::cerr << "rocjitsu: daemon socket did not appear\n";
      kill(daemon_pid, SIGTERM);
      waitpid(daemon_pid, nullptr, 0);
      return 1;
    }
  } else {
    if (!write_config_file(abs_config)) {
      std::cerr << "rocjitsu: failed to write config file\n";
      return 1;
    }
  }

  setenv("LD_PRELOAD", lib_path.c_str(), 1);
  execvp(app_argv[0], app_argv);

  std::cerr << std::format("rocjitsu: execvp failed: {}\n", strerror(errno));
  cleanup_runtime_files();
  return 1;
}
