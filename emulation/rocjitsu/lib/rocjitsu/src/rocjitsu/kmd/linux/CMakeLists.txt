# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Common KFD policy and transport intentionally compile without simdojo headers.
add_library(rocjitsu_kmd_common OBJECT
    cwsr.cpp
    events.cpp
    libc_passthrough.cpp
    linux_kfd.cpp
    kmd_command.cpp
    kfd_rpc_device.cpp
    kmd_rpc_server.cpp
    simulated_kfd.cpp
    sysfs.cpp
    transport.cpp
)
_rj_configure_object_library(rocjitsu_kmd_common util)
target_link_libraries(rocjitsu_kmd_common PRIVATE ${CMAKE_DL_LIBS})

# Internal reusable core. The embedding library and simulator use the same
# compiled policy objects; neither maintains a separate ioctl implementation.
add_library(rocjitsu_kmd_core STATIC $<TARGET_OBJECTS:rocjitsu_kmd_common>)
target_link_libraries(rocjitsu_kmd_core PUBLIC util ${CMAKE_DL_LIBS} Threads::Threads)

rj_add_object_library(rocjitsu_kmd
    ../device/simdojo_device_backend.cpp
    ../device/simdojo_debug.cpp
    ../device/simdojo_kfd.cpp
    guest_kfd.cpp
    legacy_gpu_vm.cpp
    libc_passthrough.cpp
    linux_kfd.cpp
    kmd_rpc_server.cpp
    remote_driver.cpp
    rj_daemon.cpp
)
target_include_directories(rocjitsu_kmd PRIVATE ${GENERATED_DIR})
target_link_libraries(rocjitsu_kmd PUBLIC rocjitsu_kmd_core ${CMAKE_DL_LIBS} PRIVATE util)
