/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#if HT_AMD
namespace {

constexpr char kBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr uint64_t kMaliciousOffset = 0xFFFFFFF0ULL;
constexpr uint64_t kMaliciousSize = 0x1000ULL;

void AppendU64(std::vector<uint8_t>& bytes, uint64_t value) {
  for (unsigned int i = 0; i < sizeof(value); ++i) {
    bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
  }
}

void AppendBundleEntry(std::vector<uint8_t>& bytes, const std::string& id, uint64_t offset,
                       uint64_t size) {
  AppendU64(bytes, offset);
  AppendU64(bytes, size);
  AppendU64(bytes, id.size());
  bytes.insert(bytes.end(), id.begin(), id.end());
}

std::vector<uint8_t> BuildBundle(const std::string& arch, uint64_t offset, uint64_t size) {
  const std::string arch_without_features = arch.substr(0, arch.find(':'));
  std::vector<std::string> ids;
  for (const char* kind : {"hip", "hipv4"}) {
    ids.emplace_back(std::string(kind) + "-amdgcn-amd-amdhsa--" + arch);
    if (arch_without_features != arch) {
      ids.emplace_back(std::string(kind) + "-amdgcn-amd-amdhsa--" + arch_without_features);
    }
    ids.emplace_back(std::string(kind) + "-spirv64-amd-amdhsa--amdgcnspirv");
    ids.emplace_back(std::string(kind) + "-spirv64-amd-amdhsa-unknown-amdgcnspirv");
  }

  std::vector<uint8_t> bytes;
  bytes.insert(bytes.end(), kBundleMagic, kBundleMagic + sizeof(kBundleMagic) - 1);
  AppendU64(bytes, ids.size());
  for (const auto& id : ids) {
    AppendBundleEntry(bytes, id, offset, size);
  }
  return bytes;
}

class ScopedBundleFile {
 public:
  explicit ScopedBundleFile(const std::vector<uint8_t>& bytes)
      : path_("fatbin_bounds_" + std::to_string(reinterpret_cast<uintptr_t>(bytes.data())) +
              ".code") {
    std::ofstream file(path_, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    REQUIRE(file.good());
  }

  ~ScopedBundleFile() { std::remove(path_.c_str()); }

  const char* path() const { return path_.c_str(); }

 private:
  std::string path_;
};

void ExpectFileLoadRejected(const std::vector<uint8_t>& bundle) {
  const ScopedBundleFile file(bundle);
  hipModule_t module = nullptr;
  REQUIRE(hipModuleLoad(&module, file.path()) == hipErrorInvalidImage);
  REQUIRE(module == nullptr);
}

}  // namespace

HIP_TEST_CASE(Unit_hipModuleLoad_MalformedFatBinaryBounds) {
  HIP_CHECK(hipFree(nullptr));
  hipDeviceProp_t props = {};
  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  SECTION("code object offset is outside the image") {
    const auto bundle = BuildBundle(props.gcnArchName, kMaliciousOffset, kMaliciousSize);

    hipModule_t module = nullptr;
    REQUIRE(hipModuleLoadData(&module, bundle.data()) == hipErrorInvalidImage);
    REQUIRE(module == nullptr);
    ExpectFileLoadRejected(bundle);
  }

  SECTION("code object size crosses the image boundary") {
    const auto layout = BuildBundle(props.gcnArchName, 0, 0);
    const auto bundle = BuildBundle(props.gcnArchName, layout.size() - 1, 2);
    ExpectFileLoadRejected(bundle);
  }
}
#endif
