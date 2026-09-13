# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(kmd_core_tests kmd_core_test.cpp)
target_link_libraries(kmd_core_tests PRIVATE rocjitsu_kmd_core GTest::gtest_main)
rj_configure_target(kmd_core_tests PRIVATE_INCLUDES WARNINGS)
rj_add_gtest_tests(kmd_core_tests)

add_executable(kmd_rpc_server_tests
    kmd_rpc_server_test.cpp
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
