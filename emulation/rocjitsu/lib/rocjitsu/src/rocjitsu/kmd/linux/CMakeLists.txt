# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Common KFD policy and transport intentionally compile without simdojo headers.
add_library(rocjitsu_kmd_common OBJECT
    ../device/callback_device_backend.cpp
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

add_library(rocjitsu_kmd_server SHARED ../server.cpp)
target_link_libraries(rocjitsu_kmd_server PRIVATE rocjitsu_kmd_core)
rj_configure_target(rocjitsu_kmd_server PRIVATE_INCLUDES WARNINGS HIDDEN)
set_target_properties(rocjitsu_kmd_server PROPERTIES VERSION 2.0.0 SOVERSION 2)
set(_kmd_server_map ${CMAKE_CURRENT_SOURCE_DIR}/../server.map)
target_link_options(rocjitsu_kmd_server PRIVATE "LINKER:--version-script=${_kmd_server_map}" "LINKER:--no-undefined")
set_property(TARGET rocjitsu_kmd_server APPEND PROPERTY LINK_DEPENDS ${_kmd_server_map})
install(TARGETS rocjitsu_kmd_server LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${ROCJITSU_INCLUDE_DIR}/rocjitsu/kmd/server.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocjitsu/kmd)

if(RJ_KMD_SERVER_ONLY)
    return()
endif()

rj_add_object_library(rocjitsu_kmd
    ../device/simdojo_device_backend.cpp
    ../device/simdojo_debug.cpp
    ../device/simdojo_kfd.cpp
    guest_kfd.cpp
    legacy_gpu_vm.cpp
    remote_driver.cpp
    rj_daemon.cpp
)
target_include_directories(rocjitsu_kmd PRIVATE ${GENERATED_DIR})
target_link_libraries(rocjitsu_kmd PUBLIC rocjitsu_kmd_core ${CMAKE_DL_LIBS} PRIVATE util)
