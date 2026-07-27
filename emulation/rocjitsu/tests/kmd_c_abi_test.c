// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/server.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ABI_INIT(value)                                                       \
  do {                                                                        \
    memset(&(value), 0, sizeof(value));                                        \
    (value).abi_version = RJ_KMD_SERVER_ABI_VERSION;                           \
    (value).struct_size = sizeof(value);                                       \
  } while (0)

static int32_t process_ok(void *context,
                          const rj_kmd_process_request_t *request) {
  (void)context;
  (void)request;
  return 0;
}

static int32_t allocate_unsupported(
    void *context, const rj_kmd_allocation_request_t *request,
    rj_kmd_backing_slice_t *slice) {
  (void)context;
  (void)request;
  (void)slice;
  return RJ_KMD_STATUS_NOT_SUPPORTED;
}

static int32_t free_ok(void *context,
                       const rj_kmd_free_request_t *request) {
  (void)context;
  (void)request;
  return 0;
}

static int32_t map_ok(void *context, const rj_kmd_mapping_t *mapping) {
  (void)context;
  (void)mapping;
  return 0;
}

static int32_t unmap_ok(void *context,
                        const rj_kmd_unmap_request_t *request) {
  (void)context;
  (void)request;
  return 0;
}

static int32_t apertures_ok(void *context,
                            const rj_kmd_apertures_t *apertures) {
  (void)context;
  (void)apertures;
  return 0;
}

static int32_t queue_ok(void *context, const rj_kmd_queue_t *queue) {
  (void)context;
  (void)queue;
  return 0;
}

static int32_t queue_key_ok(void *context,
                            const rj_kmd_queue_key_t *queue) {
  (void)context;
  (void)queue;
  return 0;
}

int main(void) {
  char runtime_template[] = "/tmp/rocjitsu_c_abi_XXXXXX";
  char *runtime = mkdtemp(runtime_template);
  assert(runtime != NULL);

  char socket_path[512];
  int length =
      snprintf(socket_path, sizeof(socket_path), "%s/kmd.sock", runtime);
  assert(length > 0 && (size_t)length < sizeof(socket_path));

  rj_kmd_device_info_t device;
  ABI_INIT(device);
  device.architecture = 2;
  device.gpu_id = 38144;
  device.gfx_target_version = 90402;
  device.vendor_id = 0x1002;
  device.device_id = 0x74a1;
  device.drm_render_minor = 128;
  device.wave_front_size = 64;
  device.local_mem_size = 1ull << 30;
  strcpy(device.marketing_name, "rocjitsu C ABI test GPU");

  rj_kmd_server_config_t config;
  ABI_INIT(config);
  config.runtime_root = runtime;
  config.socket_path = socket_path;

  rj_kmd_backend_ops_t backend;
  ABI_INIT(backend);
  backend.open_process = process_ok;
  backend.close_process = process_ok;
  backend.allocate = allocate_unsupported;
  backend.free = free_ok;
  backend.map = map_ok;
  backend.unmap = unmap_ok;
  backend.set_apertures = apertures_ok;
  backend.create_queue = queue_ok;
  backend.destroy_queue = queue_key_ok;

  char error[256] = {0};
  rj_kmd_server_t *server = NULL;
  assert(rj_kmd_server_create(&device, &config, &backend, &server, error,
                              sizeof(error)) == RJ_KMD_STATUS_SUCCESS);
  assert(server != NULL);
  assert(rj_kmd_server_listen(server, error, sizeof(error)) ==
         RJ_KMD_STATUS_SUCCESS);

  size_t count = 0;
  assert(rj_kmd_server_poll_fds(server, NULL, &count) ==
         RJ_KMD_STATUS_SUCCESS);
  assert(count == 1);
  rj_kmd_poll_fd_t poll_fd;
  count = 1;
  assert(rj_kmd_server_poll_fds(server, &poll_fd, &count) ==
         RJ_KMD_STATUS_SUCCESS);
  assert(poll_fd.abi_version == RJ_KMD_SERVER_ABI_VERSION);
  assert(poll_fd.struct_size == sizeof(poll_fd));
  assert(poll_fd.fd >= 0);

  rj_kmd_server_stats_t stats;
  ABI_INIT(stats);
  assert(rj_kmd_server_get_stats(server, &stats) ==
         RJ_KMD_STATUS_SUCCESS);
  assert(stats.active_clients == 0);
  assert(stats.active_processes == 0);

  assert(rj_kmd_server_stop_accepting(server) == RJ_KMD_STATUS_SUCCESS);
  rj_kmd_server_destroy(server);

  device.abi_version++;
  server = NULL;
  assert(rj_kmd_server_create(&device, &config, &backend, &server, error,
                              sizeof(error)) ==
         RJ_KMD_STATUS_INVALID_ARGUMENT);
  assert(server == NULL);
  return 0;
}
