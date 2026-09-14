# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(kmd_core_tests kmd_core_test.cpp callback_device_backend_test.cpp)
target_link_libraries(kmd_core_tests PRIVATE rocjitsu_kmd_core GTest::gtest_main)
rj_configure_target(kmd_core_tests PRIVATE_INCLUDES WARNINGS)
rj_add_gtest_tests(kmd_core_tests)

add_executable(kmd_embedding_tests kmd_embedding_test.cpp)
target_link_libraries(kmd_embedding_tests PRIVATE rocjitsu_kmd_server util Threads::Threads GTest::gtest_main)
rj_configure_target(kmd_embedding_tests PRIVATE_INCLUDES WARNINGS)
if(RJ_INSTALL_TESTS)
    if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(_kmd_embedding_rpath "${CMAKE_INSTALL_LIBDIR}")
    else()
        file(RELATIVE_PATH _kmd_embedding_lib_relative
            "${CMAKE_INSTALL_FULL_DATADIR}/rocjitsu/tests/bin"
            "${CMAKE_INSTALL_FULL_LIBDIR}")
        set(_kmd_embedding_rpath "$ORIGIN/${_kmd_embedding_lib_relative}")
    endif()
    set_property(TARGET kmd_embedding_tests PROPERTY INSTALL_RPATH "${_kmd_embedding_rpath}")
endif()
rj_add_gtest_tests(kmd_embedding_tests)

add_executable(kmd_rpc_server_tests
    kmd_rpc_server_test.cpp
    callback_dispatcher_test.cpp
    callback_memory_test.cpp
    doorbell_monitor_test.cpp
    ${ROCJITSU_SRC_DIR}/rocjitsu/kmd/linux/kmd_rpc_server.cpp
)
target_link_libraries(kmd_rpc_server_tests PRIVATE util Threads::Threads GTest::gtest_main)
rj_configure_target(kmd_rpc_server_tests PRIVATE_INCLUDES WARNINGS)
rj_add_test(
    NAME KmdRpcServerLifecycle
    COMMAND $<TARGET_FILE:kmd_rpc_server_tests>
    TIMEOUT 20
    INSTALL_COMMAND "bin/kmd_rpc_server_tests"
)
if(RJ_INSTALL_TESTS)
    rj_install_test_target(kmd_rpc_server_tests)
endif()
