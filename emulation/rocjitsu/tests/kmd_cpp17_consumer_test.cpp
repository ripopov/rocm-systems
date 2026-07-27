// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/server.h"

#include <cassert>

int main() {
  static_assert(RJ_KMD_SERVER_ABI_VERSION == 1);
  rj_kmd_server_t *server = nullptr;
  assert(rj_kmd_server_create(nullptr, nullptr, nullptr, &server, nullptr, 0) ==
         RJ_KMD_STATUS_INVALID_ARGUMENT);
  assert(server == nullptr);
  return 0;
}
