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
#include <map>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

class TestDeviceBackend : public rocjitsu::kmd::DeviceBackend {
public:
  int registration_result = 0;
  int queue_result = 0;
  bool queue_exception = false;
  int update_result = 0;
  unsigned update_calls = 0;
  uint32_t last_percentage = 0;
  int live_processes = 0;
  rocjitsu::kmd::DeviceMemory *memory_allocator = nullptr;
  std::vector<std::string> *lifecycle = nullptr;
  std::function<void(uint32_t)> on_unregister;
  rocjitsu::kmd::DeviceMemory *memory() override { return memory_allocator; }
  rj_code_arch_t architecture() const override { return ROCJITSU_CODE_ARCH_CDNA4; }
  void configure(const rocjitsu::kmd::DeviceApertures &, uint32_t,
                 const rocjitsu::kmd::DeviceCallbacks &) override {}
  void set_memory_fault_reporter(rocjitsu::kmd::MemoryFaultReporter *) override {}
  int register_process(rocjitsu::KfdProcess &, bool) override {
    if (registration_result == 0)
      ++live_processes;
    return registration_result;
  }
  void unregister_process(uint32_t process) override {
    if (lifecycle)
      lifecycle->push_back("unregister");
    if (on_unregister)
      on_unregister(process);
    --live_processes;
  }
  void set_process_client_pid(uint32_t, int32_t) override {}
  void set_process_mem_fd(uint32_t, int) override {}
  std::optional<uint32_t> queue_xcc(uint32_t) const override { return 0; }
  int create_queue(rocjitsu::KfdProcess &, const rocjitsu::kmd::QueueDescriptor &) override {
    if (queue_exception)
      throw std::bad_alloc();
    return queue_result;
  }
  int update_queue(uint32_t, uint32_t, uint64_t, uint32_t, uint32_t percentage) override {
    ++update_calls;
    last_percentage = percentage;
    return update_result;
  }
  void destroy_queue(uint32_t, uint32_t) override {
    if (lifecycle)
      lifecycle->push_back("destroy_queue");
  }
  void set_doorbell_base(uint32_t, void *) override {}
  uint64_t read64(uint32_t, uint64_t) override { return 0; }
  bool write64(uint32_t, uint64_t, uint64_t) override { return true; }
};

class TestDeviceMemory : public rocjitsu::kmd::DeviceMemory {
public:
  struct State {
    int live = 0;
    int released = 0;
    std::vector<std::string> lifecycle;
    std::map<std::pair<uint32_t, uint64_t>, rocjitsu::kmd::DeviceMapping> mappings;
  };
  struct Lease : rocjitsu::kmd::DeviceAllocation {
    std::shared_ptr<State> state;
    explicit Lease(std::shared_ptr<State> state) : state(std::move(state)) { ++this->state->live; }
    ~Lease() override {
      EXPECT_TRUE(
          std::none_of(state->mappings.begin(), state->mappings.end(),
                       [&](const auto &entry) { return entry.second.allocation_id == id; }));
      if (fd >= 0)
        ::close(fd);
      --state->live;
      ++state->released;
      state->lifecycle.push_back("release");
    }
  };
  std::shared_ptr<State> state = std::make_shared<State>();
  int result = 0;
  bool invalid_range = false;
  int map_result = 0;
  int unmap_result = 0;
  int map_calls = 0;
  int unmap_calls = 0;
  int map(const rocjitsu::kmd::DeviceMapping &mapping) override {
    ++map_calls;
    if (map_result != 0)
      return map_result;
    state->mappings[{mapping.process_id, mapping.gpu_va}] = mapping;
    return 0;
  }
  int unmap(uint32_t process, uint64_t va, uint64_t) override {
    ++unmap_calls;
    if (unmap_result != 0)
      return unmap_result;
    state->mappings.erase({process, va});
    return 0;
  }
  rocjitsu::kmd::DeviceAllocationRequest last_request{};
  int allocate(const rocjitsu::kmd::DeviceAllocationRequest &request,
               std::shared_ptr<rocjitsu::kmd::DeviceAllocation> &out) override {
    last_request = request;
    if (result != 0)
      return result;
    auto lease = std::make_shared<Lease>(state);
    lease->fd = static_cast<int>(syscall(SYS_memfd_create, "backend-arena", MFD_CLOEXEC));
    if (lease->fd < 0)
      return -errno;
    if (ftruncate(lease->fd, 16384) != 0)
      return -errno;
    lease->id = 77;
    lease->device_address = 0x80000000;
    lease->file_offset = invalid_range ? 16384 : 8192;
    lease->size = 4096;
    out = std::move(lease);
    return 0;
  }
};

TEST(KmdCoreTest, ExplicitTopologyRootIsIsolatedAndDoesNotChangeEnvironment) {
  rocjitsu::test::ScopedTempDirectory first_root{"rj-embedded-topology-first-"};
  rocjitsu::test::ScopedTempDirectory second_root{"rj-embedded-topology-second-"};
  const char *environment = std::getenv("ROCJITSU_RUNTIME_DIR");
  const std::optional<std::string> before =
      environment ? std::optional<std::string>(environment) : std::nullopt;
  rocjitsu::Sysfs first, second;
  rocjitsu::Sysfs::GpuInfo gpu{};
  gpu.gpu_id = 7;
  gpu.drm_render_minor = 128;
  const auto first_path = first.generate(gpu, first_root.path());
  ASSERT_FALSE(first_path.empty());
  EXPECT_EQ(std::filesystem::path(first_path).parent_path(), first_root.path());
  EXPECT_EQ(std::filesystem::path(first.drm_path()).parent_path(), first_root.path());
  const auto second_path = second.generate(gpu, second_root.path());
  ASSERT_FALSE(second_path.empty());
  EXPECT_EQ(std::filesystem::path(second_path).parent_path(), second_root.path());
  const auto second_drm = second.drm_path();
  first.cleanup();
  EXPECT_FALSE(std::filesystem::exists(first_path));
  EXPECT_TRUE(std::filesystem::exists(first_root.path()));
  EXPECT_TRUE(std::filesystem::exists(second_path));
  EXPECT_TRUE(std::filesystem::exists(second_drm));
  const char *after = std::getenv("ROCJITSU_RUNTIME_DIR");
  EXPECT_EQ(after ? std::optional<std::string>(after) : std::nullopt, before);
}

TEST(KmdCoreTest, ExplicitInvalidTopologyRootDoesNotFallBackToDefaultDirectory) {
  rocjitsu::test::ScopedTempDirectory root{"rj-embedded-topology-invalid-"};
  const std::string blocker = root.path() + "/not-a-directory";
  util::UniqueHandle file(::open(blocker.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600));
  ASSERT_GE(file.get(), 0);
  rocjitsu::Sysfs topology;
  EXPECT_TRUE(topology.generate(rocjitsu::Sysfs::GpuInfo{}, blocker).empty());
  EXPECT_TRUE(topology.path().empty());
  EXPECT_TRUE(topology.drm_path().empty());
  EXPECT_TRUE(std::filesystem::is_regular_file(blocker));
}

TEST(KmdCoreTest, SvmDoesNotClaimMappingsForBackendOwnedStorage) {
  for (bool external : {false, true}) {
    TestDeviceMemory allocator;
    auto backend = std::make_unique<TestDeviceBackend>();
    backend->memory_allocator = external ? &allocator : nullptr;
    std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
    devices.push_back({7, std::move(backend)});
    rocjitsu::SimulatedKfd driver(std::move(devices));
    const auto pid = driver.open_process();
    ASSERT_NE(pid, 0);
    auto process = driver.find_process(pid);
    kfd_ioctl_svm_args args;
    std::memset(&args, 0, sizeof(args));
    args.op = KFD_IOCTL_SVM_OP_SET_ATTR;
    args.start_addr = 0x400000;
    args.size = 4096;
    EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_SVM, &args), external ? -ENOTSUP : 0);
    EXPECT_EQ(process->svm_ranges_.empty(), external);
    EXPECT_TRUE(process->allocations_.empty());
    EXPECT_EQ(allocator.state->live, 0);
    args.op = KFD_IOCTL_SVM_OP_GET_ATTR;
    EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_SVM, &args), external ? -ENOTSUP : 0);
    EXPECT_EQ(driver.close(pid), 0);
  }
}

TEST(KmdCoreTest, BackendAllocationFailureDoesNotFallBackOrPublishHandles) {
  TestDeviceMemory allocator;
  auto backend = std::make_unique<TestDeviceBackend>();
  backend->memory_allocator = &allocator;
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto pid = driver.open_process();
  ASSERT_NE(pid, 0);
  auto process = driver.find_process(pid);
  const auto next_handle = process->next_handle_;
  const auto next_va = process->next_gpu_va_;
  kfd_ioctl_alloc_memory_of_gpu_args args{};
  args.gpu_id = 7;
  args.size = 4096;
  args.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  args.handle = 1234;
  allocator.result = -ENOMEM;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args), -ENOMEM);
  EXPECT_EQ(args.handle, 1234);
  EXPECT_TRUE(process->allocations_.empty());
  EXPECT_EQ(process->next_handle_, next_handle);
  EXPECT_EQ(process->next_gpu_va_, next_va);
  allocator.result = 0;
  allocator.invalid_range = true;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args), -EINVAL);
  EXPECT_EQ(allocator.state->live, 0);
  EXPECT_EQ(allocator.state->released, 1);
  EXPECT_TRUE(process->allocations_.empty());
  EXPECT_EQ(process->next_handle_, next_handle);
  EXPECT_EQ(process->next_gpu_va_, next_va);
  EXPECT_EQ(driver.close(pid), 0);
}

TEST(KmdCoreTest, BackendAllocationLeaseLivesUntilLastIpcImportIsFreed) {
  TestDeviceMemory allocator;
  auto backend = std::make_unique<TestDeviceBackend>();
  backend->memory_allocator = &allocator;
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto source = driver.open_process();
  const auto target = driver.open_process();
  ASSERT_NE(source, 0);
  ASSERT_NE(target, 0);
  kfd_ioctl_alloc_memory_of_gpu_args args{};
  args.gpu_id = 7;
  args.size = 4096;
  args.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  ASSERT_EQ(driver.ioctl(source, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args), 0);
  EXPECT_EQ(allocator.last_request.process_id, source);
  EXPECT_EQ(allocator.last_request.gpu_va, args.va_addr);
  auto process = driver.find_process(source);
  const auto &allocation = process->allocations_.at(args.handle);
  EXPECT_NE(allocation.memfd, allocation.device_allocation->fd);
  EXPECT_EQ(allocation.memfd_offset, 8192);
  kfd_ioctl_export_dmabuf_args dmabuf{};
  dmabuf.handle = args.handle;
  EXPECT_EQ(driver.ioctl(source, AMDKFD_IOC_EXPORT_DMABUF, &dmabuf), -ENOTSUP);
  *static_cast<uint64_t *>(allocation.host_ptr) = 0x12345678;
  kfd_ioctl_ipc_export_handle_args exported{};
  exported.handle = args.handle;
  ASSERT_EQ(driver.ioctl(source, AMDKFD_IOC_IPC_EXPORT_HANDLE, &exported), 0);
  kfd_ioctl_ipc_import_handle_args imported{};
  imported.gpu_id = 7;
  std::memcpy(imported.share_handle, exported.share_handle, sizeof(imported.share_handle));
  ASSERT_EQ(driver.ioctl(target, AMDKFD_IOC_IPC_IMPORT_HANDLE, &imported), 0);
  uint32_t gpu_id = 7;
  kfd_ioctl_map_memory_to_gpu_args import_mapping{};
  import_mapping.handle = imported.handle;
  import_mapping.n_devices = 1;
  import_mapping.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  ASSERT_EQ(driver.ioctl(target, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &import_mapping), 0);
  const auto target_va = driver.find_process(target)->allocations_.at(imported.handle).gpu_va;
  EXPECT_EQ(allocator.state->mappings.at({target, target_va}).memory_type,
            static_cast<uint8_t>(rocjitsu::amdgpu::Mtype::UC));
  ASSERT_EQ(driver.close(source), 0);
  EXPECT_EQ(allocator.state->live, 1);
  EXPECT_EQ(allocator.state->released, 0);
  auto target_process = driver.find_process(target);
  void *imported_mapping = target_process->allocations_.at(imported.handle).host_ptr;
  EXPECT_EQ(*static_cast<uint64_t *>(imported_mapping), 0x12345678);
  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = imported.handle;
  ASSERT_EQ(driver.ioctl(target, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  EXPECT_EQ(allocator.state->live, 0);
  EXPECT_EQ(allocator.state->released, 1);
  unsigned char residency = 0;
  EXPECT_EQ(mincore(imported_mapping, 4096, &residency), -1);
  EXPECT_EQ(errno, ENOMEM);
  EXPECT_EQ(driver.close(target), 0);
}

TEST(KmdCoreTest, ShutdownStopsQueuesBeforeReleasingBackendBacking) {
  TestDeviceMemory allocator;
  auto backend = std::make_unique<TestDeviceBackend>();
  backend->memory_allocator = &allocator;
  backend->lifecycle = &allocator.state->lifecycle;
  backend->on_unregister = [&](uint32_t process) {
    std::erase_if(allocator.state->mappings,
                  [&](const auto &entry) { return entry.first.first == process; });
  };
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto pid = driver.open_process();
  ASSERT_NE(pid, 0);
  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.gpu_id = 7;
  allocation.size = 4096;
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), 0);
  uint32_t gpu_id = 7;
  kfd_ioctl_map_memory_to_gpu_args mapping{};
  mapping.handle = allocation.handle;
  mapping.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  mapping.n_devices = 1;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mapping), 0);
  ASSERT_EQ(allocator.state->mappings.size(), 1);
  kfd_ioctl_create_queue_args queue{};
  queue.gpu_id = 7;
  queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &queue), 0);
  ASSERT_EQ(driver.close(pid), 0);
  EXPECT_EQ(allocator.state->lifecycle,
            (std::vector<std::string>{"destroy_queue", "unregister", "release"}));
}

TEST(KmdCoreTest, DeviceMappingFailuresRetainBackingUntilUnmapSucceeds) {
  TestDeviceMemory memory;
  auto backend = std::make_unique<TestDeviceBackend>();
  backend->memory_allocator = &memory;
  backend->on_unregister = [&](uint32_t process) {
    std::erase_if(memory.state->mappings,
                  [&](const auto &entry) { return entry.first.first == process; });
  };
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto pid = driver.open_process();
  ASSERT_NE(pid, 0);
  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.gpu_id = 7;
  allocation.size = 4096;
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), 0);
  auto process = driver.find_process(pid);
  uint32_t gpu_id = 7;
  kfd_ioctl_map_memory_to_gpu_args mapping{};
  mapping.handle = allocation.handle;
  mapping.n_devices = 1;
  mapping.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  memory.map_result = -ENOMEM;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mapping), -ENOMEM);
  EXPECT_EQ(mapping.n_success, 0);
  EXPECT_TRUE(process->allocations_.at(allocation.handle).mapped_gpu_ids.empty());
  memory.map_result = 0;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mapping), 0);
  EXPECT_EQ(mapping.n_success, 1);
  const auto &installed = memory.state->mappings.at({pid, allocation.va_addr});
  EXPECT_EQ(installed.allocation_id, 77);
  EXPECT_EQ(installed.device_address, 0x80000000);
  EXPECT_EQ(installed.size, 4096);
  kfd_ioctl_unmap_memory_from_gpu_args unmapping{};
  unmapping.handle = allocation.handle;
  unmapping.n_devices = 1;
  unmapping.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  memory.unmap_result = -EBUSY;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmapping), -EBUSY);
  EXPECT_EQ(unmapping.n_success, 0);
  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = allocation.handle;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), -EBUSY);
  EXPECT_EQ(memory.state->live, 1);
  EXPECT_TRUE(process->allocations_.contains(allocation.handle));
  EXPECT_EQ(memory.state->mappings.size(), 1);
  memory.unmap_result = 0;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmapping), 0);
  EXPECT_EQ(unmapping.n_success, 1);
  EXPECT_EQ(memory.state->live, 1);
  EXPECT_TRUE(memory.state->mappings.empty());
  mapping.n_success = 0;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mapping), 0);
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  EXPECT_EQ(memory.state->live, 0);
  EXPECT_EQ(memory.state->released, 1);
  EXPECT_TRUE(memory.state->mappings.empty());
  EXPECT_EQ(driver.close(pid), 0);
}

TEST(KmdCoreTest, DeviceMappingReportsCompletedPrefixAndResumesRetry) {
  TestDeviceMemory first_memory;
  TestDeviceMemory second_memory;
  auto first = std::make_unique<TestDeviceBackend>();
  auto second = std::make_unique<TestDeviceBackend>();
  first->memory_allocator = &first_memory;
  second->memory_allocator = &second_memory;
  first->on_unregister = [&](uint32_t process) {
    std::erase_if(first_memory.state->mappings,
                  [&](const auto &entry) { return entry.first.first == process; });
  };
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(first)});
  devices.push_back({8, std::move(second)});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto pid = driver.open_process();
  ASSERT_NE(pid, 0);
  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.gpu_id = 7;
  allocation.size = 4096;
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), 0);
  uint32_t ids[] = {7, 8};
  kfd_ioctl_map_memory_to_gpu_args mapping{};
  mapping.handle = allocation.handle;
  mapping.n_devices = 2;
  mapping.device_ids_array_ptr = reinterpret_cast<uint64_t>(ids);
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mapping), -EXDEV);
  EXPECT_EQ(mapping.n_success, 1);
  EXPECT_EQ(first_memory.map_calls, 1);
  EXPECT_EQ(second_memory.map_calls, 0);
  EXPECT_EQ(first_memory.state->mappings.size(), 1);
  // The successful prefix is not submitted again on retry.
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mapping), -EXDEV);
  EXPECT_EQ(mapping.n_success, 1);
  EXPECT_EQ(first_memory.map_calls, 1);
  kfd_ioctl_unmap_memory_from_gpu_args unmapping{};
  unmapping.handle = allocation.handle;
  unmapping.n_devices = 2;
  unmapping.device_ids_array_ptr = reinterpret_cast<uint64_t>(ids);
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmapping), 0);
  EXPECT_EQ(unmapping.n_success, 2);
  EXPECT_EQ(first_memory.unmap_calls, 1);
  EXPECT_EQ(second_memory.unmap_calls, 0);
  EXPECT_TRUE(first_memory.state->mappings.empty());
  EXPECT_EQ(driver.close(pid), 0);
}

TEST(KmdCoreTest, IpcExportReconfiguresDeviceCachePolicyAndPreservesItAcrossRemap) {
  TestDeviceMemory memory;
  auto backend = std::make_unique<TestDeviceBackend>();
  backend->memory_allocator = &memory;
  backend->on_unregister = [&](uint32_t process) {
    std::erase_if(memory.state->mappings,
                  [&](const auto &entry) { return entry.first.first == process; });
  };
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto pid = driver.open_process();
  ASSERT_NE(pid, 0);
  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.gpu_id = 7;
  allocation.size = 4096;
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), 0);
  const auto process = driver.find_process(pid);
  uint32_t gpu_id = 7;
  kfd_ioctl_map_memory_to_gpu_args mapping{};
  mapping.handle = allocation.handle;
  mapping.n_devices = 1;
  mapping.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mapping), 0);
  const auto key = std::make_pair(pid, allocation.va_addr);
  EXPECT_EQ(memory.state->mappings.at(key).memory_type,
            static_cast<uint8_t>(rocjitsu::amdgpu::Mtype::RW));

  kfd_ioctl_ipc_export_handle_args exported{};
  exported.handle = allocation.handle;
  memory.map_result = -EIO;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_IPC_EXPORT_HANDLE, &exported), -EIO);
  for (const auto word : exported.share_handle)
    EXPECT_EQ(word, 0);
  EXPECT_FALSE(process->allocations_.at(allocation.handle).memory_type_override.has_value());
  EXPECT_EQ(memory.state->mappings.at(key).memory_type,
            static_cast<uint8_t>(rocjitsu::amdgpu::Mtype::RW));
  memory.map_result = 0;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_IPC_EXPORT_HANDLE, &exported), 0);
  EXPECT_EQ(memory.state->mappings.at(key).memory_type,
            static_cast<uint8_t>(rocjitsu::amdgpu::Mtype::CC));
  EXPECT_EQ(process->allocations_.at(allocation.handle).flags, KFD_IOC_ALLOC_MEM_FLAGS_VRAM);

  kfd_ioctl_unmap_memory_from_gpu_args unmapping{};
  unmapping.handle = allocation.handle;
  unmapping.n_devices = 1;
  unmapping.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmapping), 0);
  mapping.n_success = 0;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mapping), 0);
  EXPECT_EQ(memory.state->mappings.at(key).memory_type,
            static_cast<uint8_t>(rocjitsu::amdgpu::Mtype::CC));
  EXPECT_EQ(driver.close(pid), 0);
}

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
  EXPECT_EQ(args.queue_id, 0u);
  EXPECT_EQ(args.doorbell_offset, 0u);
  auto process = driver.find_process(process_id);
  ASSERT_NE(process, nullptr);
  EXPECT_TRUE(process->active_queue_ids_.empty());
  EXPECT_TRUE(process->queue_doorbell_map_.empty());
  EXPECT_TRUE(process->queue_snapshot_map_.empty());
  EXPECT_TRUE(process->gpu(0).free_doorbell_offsets.empty());
  EXPECT_EQ(process->gpu(0).next_doorbell_offset, 0u);
  state->queue_result = 0;
  ASSERT_EQ(driver.ioctl(process_id, AMDKFD_IOC_CREATE_QUEUE, &args), 0);
  EXPECT_EQ(args.doorbell_offset & 0xfffu, 0u);
  EXPECT_TRUE(process->gpu(0).free_doorbell_offsets.empty());
  EXPECT_EQ(driver.close(process_id), 0);
}

TEST(KmdCoreTest, QueueCreationExceptionDoesNotPublishOrConsumeDoorbell) {
  auto backend = std::make_unique<TestDeviceBackend>();
  auto *state = backend.get();
  state->queue_exception = true;
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(backend)});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto pid = driver.open_process();
  kfd_ioctl_create_queue_args args{};
  args.gpu_id = 7;
  args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &args), -ENOMEM);
  auto process = driver.find_process(pid);
  ASSERT_NE(process, nullptr);
  EXPECT_TRUE(process->active_queue_ids_.empty());
  EXPECT_TRUE(process->queue_doorbell_map_.empty());
  EXPECT_TRUE(process->queue_snapshot_map_.empty());
  EXPECT_EQ(process->gpu(0).next_doorbell_offset, 0u);
  EXPECT_EQ(args.queue_id, 0u);
  EXPECT_EQ(args.doorbell_offset, 0u);
  state->queue_exception = false;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &args), 0);
  EXPECT_EQ(args.doorbell_offset & 0xfffu, 0u);
  EXPECT_EQ(driver.close(pid), 0);
}

TEST(KmdCoreTest, QueueUpdateTargetsOwnerAndPreservesSnapshotOnFailure) {
  auto first = std::make_unique<TestDeviceBackend>();
  auto second = std::make_unique<TestDeviceBackend>();
  auto *other = first.get();
  auto *owner = second.get();
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::move(first)});
  devices.push_back({8, std::move(second)});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto pid = driver.open_process();
  ASSERT_NE(pid, 0);
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = 8;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = 0x10000;
  create.ring_size = 4096;
  ASSERT_EQ(driver.ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &create), 0);
  auto process = driver.find_process(pid);
  kfd_ioctl_update_queue_args update{};
  update.queue_id = create.queue_id;
  update.ring_base_address = 0x20000;
  update.ring_size = 8192;
  update.queue_percentage = 50;
  owner->update_result = -EBUSY;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_UPDATE_QUEUE, &update), -EBUSY);
  EXPECT_EQ(process->queue_snapshot_map_.at(create.queue_id).ring_base_address, 0x10000u);
  EXPECT_EQ(process->queue_snapshot_map_.at(create.queue_id).ring_size, 4096u);
  EXPECT_EQ(other->update_calls, 0u);
  EXPECT_EQ(owner->update_calls, 1u);
  EXPECT_EQ(owner->last_percentage, 50u);
  owner->update_result = 0;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_UPDATE_QUEUE, &update), 0);
  EXPECT_EQ(process->queue_snapshot_map_.at(create.queue_id).ring_base_address, 0x20000u);
  EXPECT_EQ(process->queue_snapshot_map_.at(create.queue_id).ring_size, 8192u);
  update.queue_id = 0xFFFFFFFF;
  EXPECT_EQ(driver.ioctl(pid, AMDKFD_IOC_UPDATE_QUEUE, &update), -EFAULT);
  EXPECT_EQ(owner->update_calls, 2u);
  EXPECT_EQ(other->update_calls, 0u);
  EXPECT_EQ(driver.close(pid), 0);
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

TEST(KmdCoreTest, SharedBackingSliceSurvivesMappingAndIpcImport) {
  std::vector<rocjitsu::SimulatedKfd::BackendDevice> devices;
  devices.push_back({7, std::make_unique<TestDeviceBackend>()});
  rocjitsu::SimulatedKfd driver(std::move(devices));
  const auto source = driver.open_process();
  const auto target = driver.open_process();
  ASSERT_NE(source, 0);
  ASSERT_NE(target, 0);
  const int fd = static_cast<int>(syscall(SYS_memfd_create, "core-backing-slice", MFD_CLOEXEC));
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, 16384), 0);
  const uint64_t marker = 0xabcd12344321dcbaULL;
  ASSERT_EQ(pwrite(fd, &marker, sizeof(marker), 8192), sizeof(marker));
  // Supply a backend-style allocation whose logical range occupies one page
  // in a larger file. The KFD owns this descriptor from here through close().
  auto process = driver.find_process(source);
  auto &allocation = process->allocations_[42];
  allocation.handle = 42;
  allocation.gpu_id = 7;
  allocation.gpu_va = 0x1000000000ULL;
  allocation.size = 4096;
  allocation.memfd = fd;
  allocation.memfd_offset = 8192;

  rocjitsu::KfdRpcDevice device(driver);
  rj_vm_map_t map{};
  map.length = 4096;
  map.offset = 42 << 12;
  map.prot = PROT_READ | PROT_WRITE;
  map.flags = MAP_SHARED;
  ASSERT_EQ(device.map(source, &map), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(map.map_errno, 0);
  EXPECT_EQ(map.shared_handle, fd);
  EXPECT_EQ(map.shared_offset, 8192);
  EXPECT_EQ(map.shared_size, 4096);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(map.mapped_addr), marker);
  auto oversized = map;
  oversized.length = 8192;
  EXPECT_EQ(device.map(source, &oversized), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(oversized.map_errno, EINVAL);
  struct stat backing_stat{};
  ASSERT_EQ(fstat(fd, &backing_stat), 0);
  EXPECT_EQ(backing_stat.st_size, 16384);

  kfd_ioctl_ipc_export_handle_args export_args{};
  export_args.handle = 42;
  ASSERT_EQ(driver.ioctl(source, AMDKFD_IOC_IPC_EXPORT_HANDLE, &export_args), 0);
  kfd_ioctl_ipc_import_handle_args import_args{};
  std::memcpy(import_args.share_handle, export_args.share_handle, sizeof(import_args.share_handle));
  import_args.gpu_id = 7;
  rj_vm_cmd_t command{};
  command.cmd = AMDKFD_IOC_IPC_IMPORT_HANDLE;
  command.buf = &import_args;
  command.buf_size = sizeof(import_args);
  ASSERT_EQ(device.execute(target, &command), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(command.result, 0);
  EXPECT_GE(command.shared_handle, 0);
  EXPECT_EQ(command.shared_offset, 8192);
  EXPECT_EQ(command.shared_size, 4096);
  const auto imported = driver.find_process(target);
  EXPECT_EQ(*static_cast<uint64_t *>(imported->allocations_.at(import_args.handle).host_ptr),
            marker);
  EXPECT_EQ(driver.close(source), 0);
  EXPECT_EQ(*static_cast<uint64_t *>(imported->allocations_.at(import_args.handle).host_ptr),
            marker);
  EXPECT_EQ(driver.close(target), 0);
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

  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.gpu_id = 7;
  allocation.size = 4096;
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  ioctl_request.ioctl_cmd = AMDKFD_IOC_ALLOC_MEMORY_OF_GPU;
  ioctl_request.args_bytes = sizeof(allocation);
  request.request_id = 43;
  request.payload_bytes = sizeof(ioctl_request) + sizeof(allocation);
  ASSERT_TRUE(rpc_send_exact(client.get(), &request, sizeof(request)));
  ASSERT_TRUE(rpc_send_exact(client.get(), &ioctl_request, sizeof(ioctl_request)));
  ASSERT_TRUE(rpc_send_exact(client.get(), &allocation, sizeof(allocation)));
  int received_fd = -1;
  size_t num_fds = 1;
  ASSERT_EQ(rpc_recv_msg(client.get(), &response, sizeof(response), &received_fd, &num_fds),
            sizeof(response));
  util::UniqueHandle allocation_fd(received_fd);
  ASSERT_EQ(response.result, 0);
  EXPECT_EQ(response.request_id, 43);
  ASSERT_EQ(num_fds, 1);
  ASSERT_GE(allocation_fd.get(), 0);
  ASSERT_EQ(response.payload_bytes, sizeof(RpcSharedBacking) + sizeof(allocation));
  RpcSharedBacking backing{};
  ASSERT_TRUE(rpc_recv_exact(client.get(), &backing, sizeof(backing)));
  ASSERT_TRUE(rpc_recv_exact(client.get(), &allocation, sizeof(allocation)));
  EXPECT_EQ(backing.offset, 0);
  EXPECT_EQ(backing.size, 4096);

  RpcMmapRequest map_request{};
  map_request.length = 4096;
  map_request.prot = PROT_READ | PROT_WRITE;
  map_request.flags = MAP_SHARED;
  map_request.offset = allocation.mmap_offset;
  request.opcode = RPC_MMAP;
  request.request_id = 44;
  request.payload_bytes = sizeof(map_request);
  ASSERT_TRUE(rpc_send_exact(client.get(), &request, sizeof(request)));
  ASSERT_TRUE(rpc_send_exact(client.get(), &map_request, sizeof(map_request)));
  received_fd = -1;
  num_fds = 1;
  ASSERT_EQ(rpc_recv_msg(client.get(), &response, sizeof(response), &received_fd, &num_fds),
            sizeof(response));
  util::UniqueHandle mapping_fd(received_fd);
  ASSERT_EQ(response.result, 0);
  EXPECT_EQ(response.request_id, 44);
  ASSERT_EQ(num_fds, 1);
  ASSERT_GE(mapping_fd.get(), 0);
  ASSERT_EQ(response.payload_bytes, sizeof(RpcMmapResponse));
  RpcMmapResponse map_response{};
  ASSERT_TRUE(rpc_recv_exact(client.get(), &map_response, sizeof(map_response)));
  EXPECT_EQ(map_response.backing.offset, backing.offset);
  EXPECT_EQ(map_response.backing.size, backing.size);
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
