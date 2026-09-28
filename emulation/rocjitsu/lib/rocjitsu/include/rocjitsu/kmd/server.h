// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file server.h
/// @brief Stable C ABI for embedding rocjitsu's KMD RPC server.

#ifndef ROCJITSU_KMD_SERVER_H_
#define ROCJITSU_KMD_SERVER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define RJ_KMD_API __declspec(dllexport)
#else
#define RJ_KMD_API __attribute__((visibility("default")))
#endif

#define RJ_KMD_SERVER_ABI_VERSION 2u

/// ABI v2 uses the shared upstream KFD/RPC implementation. All backend
/// callbacks execute on the thread that creates the server; that thread must
/// service poll_fds/process_fd and destroy the server. Callbacks must not throw
/// exceptions, reenter the server, or block waiting for another KMD operation.
/// Return zero on success or a negative Linux errno, never a positive status.
/// Callback request pointers are borrowed only for the duration of the call.

typedef struct rj_kmd_server rj_kmd_server_t;

typedef enum rj_kmd_status {
  RJ_KMD_STATUS_SUCCESS = 0,
  RJ_KMD_STATUS_INVALID_ARGUMENT = -22,
  RJ_KMD_STATUS_NO_MEMORY = -12,
  RJ_KMD_STATUS_NOT_SUPPORTED = -95,
  RJ_KMD_STATUS_IO_ERROR = -5,
  RJ_KMD_STATUS_INTERNAL_ERROR = -255
} rj_kmd_status_t;

typedef struct rj_kmd_device_info {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t architecture;
  uint32_t gpu_id;
  uint32_t gfx_target_version;
  uint32_t vendor_id;
  uint32_t device_id;
  uint32_t family_id;
  uint64_t unique_id;
  uint32_t location_id;
  uint32_t domain;
  uint64_t hive_id;
  uint32_t drm_render_minor;
  uint32_t revision_id;
  uint32_t pci_revision_id;
  uint32_t simd_count;
  uint32_t max_waves_per_simd;
  uint32_t num_shader_engines;
  uint32_t num_shader_arrays_per_engine;
  uint32_t num_cu_per_sh;
  uint32_t simd_per_cu;
  uint32_t wave_front_size;
  uint32_t num_xcc;
  uint32_t max_slots_scratch_cu;
  uint64_t local_mem_size;
  uint32_t vram_type;
  uint32_t lds_size_kb;
  uint32_t mem_width;
  uint32_t mem_clk_max;
  uint32_t l1_size_kb;
  uint32_t l1_line_size;
  uint32_t l1_assoc;
  uint32_t l2_size_kb;
  uint32_t l2_line_size;
  uint32_t l2_assoc;
  uint32_t num_sdma_engines;
  uint32_t num_sdma_xgmi_engines;
  uint32_t num_cp_queues;
  uint32_t max_engine_clk_fcompute;
  uint32_t capability;
  uint32_t capability2;
  uint64_t debug_prop;
  uint32_t fw_version;
  uint32_t sdma_fw_version;
  char marketing_name[128];
} rj_kmd_device_info_t;

typedef void (*rj_kmd_diagnostic_fn)(void *context, int32_t status, const char *message);

typedef struct rj_kmd_server_config {
  uint32_t abi_version;
  uint32_t struct_size;
  const char *runtime_root;
  const char *socket_path;
  rj_kmd_diagnostic_fn diagnostic;
  void *diagnostic_context;
} rj_kmd_server_config_t;

typedef struct rj_kmd_process_request {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  int32_t client_pid;
} rj_kmd_process_request_t;

typedef struct rj_kmd_allocation_request {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  uint32_t gpu_id;
  uint64_t gpu_va;
  uint64_t size;
  uint32_t flags;
  uint32_t reserved;
} rj_kmd_allocation_request_t;

typedef struct rj_kmd_backing_slice {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t allocation_id;
  uint64_t device_paddr;
  uint64_t size;
  int32_t fd;
  uint32_t reserved;
  uint64_t fd_offset;
} rj_kmd_backing_slice_t;

/// On successful allocate, fd ownership transfers to rocjitsu. The backend
/// owns the storage identified by allocation_id until free, which may occur
/// after close_process when IPC imports retain the allocation. close_process
/// must quiesce device accesses and mappings, not release leased storage.
/// Allocation failure leaves no resources or output fd owned by rocjitsu.
/// A failed map/unmap leaves the previous device mapping intact. map also
/// performs atomic cache-policy replacement; it must honor memory_type.
/// A free failure is terminal and reported by the server; shutdown must reclaim
/// any backend resources that failed normal release.

typedef struct rj_kmd_free_request {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  uint32_t reserved;
  uint64_t allocation_id;
} rj_kmd_free_request_t;

/// Semantic policies, not hardware PTE encodings. Backends must translate
/// these values to their device format or reject an unsupported policy.
typedef enum rj_kmd_memory_type {
  RJ_KMD_MEMORY_UC = 0,
  RJ_KMD_MEMORY_CC = 1,
  RJ_KMD_MEMORY_RW = 2,
  RJ_KMD_MEMORY_WB = 3,
  RJ_KMD_MEMORY_NT = 4
} rj_kmd_memory_type_t;

typedef struct rj_kmd_mapping {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  uint32_t memory_type;
  uint64_t allocation_id;
  uint64_t gpu_va;
  uint64_t device_paddr;
  uint64_t size;
} rj_kmd_mapping_t;

typedef struct rj_kmd_unmap_request {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  uint32_t reserved;
  uint64_t gpu_va;
  uint64_t size;
} rj_kmd_unmap_request_t;

typedef struct rj_kmd_apertures {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t lds_base;
  uint64_t lds_limit;
  uint64_t scratch_base;
  uint64_t scratch_limit;
} rj_kmd_apertures_t;

typedef struct rj_kmd_queue {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  uint32_t queue_id;
  uint64_t ring_base_va;
  uint32_t ring_size;
  uint32_t doorbell_offset;
  uint64_t read_pointer_va;
  uint64_t write_pointer_va;
  uint64_t queue_descriptor_va;
  uint32_t host_accessible;
  uint32_t is_sdma;
} rj_kmd_queue_t;

typedef struct rj_kmd_queue_update {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  uint32_t queue_id;
  uint64_t ring_base_va;
  uint32_t ring_size;
  uint32_t queue_percentage;
} rj_kmd_queue_update_t;

typedef struct rj_kmd_queue_key {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  uint32_t queue_id;
} rj_kmd_queue_key_t;

typedef struct rj_kmd_write64 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  uint32_t reserved;
  uint64_t gpu_va;
  uint64_t value;
} rj_kmd_write64_t;

typedef struct rj_kmd_read64 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  uint32_t reserved;
  uint64_t gpu_va;
} rj_kmd_read64_t;

typedef int32_t (*rj_kmd_process_fn)(void *context, const rj_kmd_process_request_t *request);
typedef int32_t (*rj_kmd_allocate_fn)(void *context, const rj_kmd_allocation_request_t *request,
                                      rj_kmd_backing_slice_t *slice);
typedef int32_t (*rj_kmd_free_fn)(void *context, const rj_kmd_free_request_t *request);
typedef int32_t (*rj_kmd_map_fn)(void *context, const rj_kmd_mapping_t *mapping);
typedef int32_t (*rj_kmd_unmap_fn)(void *context, const rj_kmd_unmap_request_t *request);
typedef int32_t (*rj_kmd_apertures_fn)(void *context, const rj_kmd_apertures_t *apertures);
typedef int32_t (*rj_kmd_queue_fn)(void *context, const rj_kmd_queue_t *queue);
typedef int32_t (*rj_kmd_queue_update_fn)(void *context, const rj_kmd_queue_update_t *queue);
typedef int32_t (*rj_kmd_queue_key_fn)(void *context, const rj_kmd_queue_key_t *queue);
typedef int32_t (*rj_kmd_write64_fn)(void *context, const rj_kmd_write64_t *write);
typedef int32_t (*rj_kmd_read64_fn)(void *context, const rj_kmd_read64_t *read, uint64_t *value);
/// Must quiesce all device accesses before returning. Called once on final
/// teardown or earlier on a terminal cleanup failure. Subsequent free calls
/// for already reclaimed allocations must succeed without double release.
typedef void (*rj_kmd_shutdown_fn)(void *context);

typedef struct rj_kmd_backend_ops {
  uint32_t abi_version;
  uint32_t struct_size;
  void *context;
  rj_kmd_process_fn open_process;
  rj_kmd_process_fn set_process_client_pid;
  rj_kmd_process_fn close_process;
  rj_kmd_allocate_fn allocate;
  rj_kmd_free_fn free;
  rj_kmd_map_fn map;
  rj_kmd_unmap_fn unmap;
  rj_kmd_apertures_fn set_apertures;
  rj_kmd_queue_fn create_queue;
  rj_kmd_queue_update_fn update_queue;
  rj_kmd_queue_key_fn destroy_queue;
  rj_kmd_write64_fn write64;
  rj_kmd_read64_fn read64;
  rj_kmd_shutdown_fn shutdown;
} rj_kmd_backend_ops_t;

typedef struct rj_kmd_poll_fd {
  uint32_t abi_version;
  uint32_t struct_size;
  int32_t fd;
  int16_t events;
  int16_t reserved;
} rj_kmd_poll_fd_t;

typedef struct rj_kmd_doorbell_record {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t process_id;
  uint32_t queue_id;
  uint32_t offset;
  uint32_t reserved;
  uint64_t value;
} rj_kmd_doorbell_record_t;

typedef struct rj_kmd_server_stats {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t active_clients;
  uint64_t active_processes;
  uint64_t active_queues;
  /// In-flight RPC requests, including device waits and response transmission.
  uint64_t pending_responses;
  uint64_t pending_doorbells;
} rj_kmd_server_stats_t;

RJ_KMD_API int32_t rj_kmd_server_create(const rj_kmd_device_info_t *device,
                                        const rj_kmd_server_config_t *config,
                                        const rj_kmd_backend_ops_t *backend,
                                        rj_kmd_server_t **server, char *error, size_t error_size);
/// Stops clients and pumps their cleanup callbacks before releasing storage.
/// A wrong-thread call is rejected without destroying the server.
RJ_KMD_API void rj_kmd_server_destroy(rj_kmd_server_t *server);
RJ_KMD_API int32_t rj_kmd_server_listen(rj_kmd_server_t *server, char *error, size_t error_size);
RJ_KMD_API int32_t rj_kmd_server_stop_accepting(rj_kmd_server_t *server);
RJ_KMD_API int32_t rj_kmd_server_poll_fds(const rj_kmd_server_t *server, rj_kmd_poll_fd_t *fds,
                                          size_t *count);
RJ_KMD_API int32_t rj_kmd_server_process_fd(rj_kmd_server_t *server, int32_t fd, int16_t revents,
                                            char *error, size_t error_size);
RJ_KMD_API int32_t rj_kmd_server_notification_fd(const rj_kmd_server_t *server);
RJ_KMD_API int32_t rj_kmd_server_drain_doorbells(rj_kmd_server_t *server,
                                                 rj_kmd_doorbell_record_t *records, size_t *count);
RJ_KMD_API int32_t rj_kmd_server_post_interrupt(rj_kmd_server_t *server, uint32_t process_id,
                                                uint32_t event_id);
RJ_KMD_API int32_t rj_kmd_server_get_stats(const rj_kmd_server_t *server,
                                           rj_kmd_server_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif // ROCJITSU_KMD_SERVER_H_
