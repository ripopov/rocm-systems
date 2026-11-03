# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The reusable KMD server also allocates event and doorbell backing storage.
if(RJ_KMD_SERVER_ONLY)
    set(_rj_keeper_build_libdir "$<TARGET_FILE_DIR:rocjitsu_kmd_server>")
    set(_rj_keeper_build_bindir "$<TARGET_FILE_DIR:rocjitsu_memory>")
else()
    set(_rj_keeper_build_libdir "$<TARGET_FILE_DIR:rocjitsu_shared>")
    set(_rj_keeper_build_bindir "$<TARGET_FILE_DIR:rocjitsu_bin>")
endif()
set(_rj_keeper_build_testdir "${PROJECT_BINARY_DIR}/tests")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Private shared-RAM keeper. Exec gives it an independent address space, so it
    # cannot retain application mappings inherited when an allocation forks it.
    add_executable(
        rocjitsu_memory
        ${PROJECT_SOURCE_DIR}/tools/rocjitsu_memory.cpp
        ${ROCJITSU_SRC_DIR}/rocjitsu/kmd/linux/memory_backing.cpp
        ${ROCJITSU_SRC_DIR}/rocjitsu/kmd/linux/libc_passthrough.cpp
    )
    target_link_libraries(
        rocjitsu_memory
        PRIVATE util Threads::Threads ${CMAKE_DL_LIBS}
    )
    rj_configure_target(rocjitsu_memory PRIVATE_INCLUDES WARNINGS)
    set_target_properties(
        rocjitsu_memory
        PROPERTIES OUTPUT_NAME rocjitsu-memory
    )
    # Exec deliberately omits application loader configuration. Keep the private
    # helper independent of LD_LIBRARY_PATH for compiler runtime libraries.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(
            rocjitsu_memory
            PRIVATE -static-libstdc++ -static-libgcc
        )
    endif()
    file(
        RELATIVE_PATH
        _rj_keeper_relative
        "${CMAKE_INSTALL_FULL_LIBDIR}"
        "${CMAKE_INSTALL_FULL_LIBEXECDIR}/rocjitsu/rocjitsu-memory"
    )
    file(
        RELATIVE_PATH
        _rj_keeper_bin_relative
        "${CMAKE_INSTALL_FULL_BINDIR}"
        "${CMAKE_INSTALL_FULL_LIBEXECDIR}/rocjitsu/rocjitsu-memory"
    )
    file(
        RELATIVE_PATH
        _rj_keeper_test_relative
        "${CMAKE_INSTALL_FULL_DATADIR}/rocjitsu/tests/bin"
        "${CMAKE_INSTALL_FULL_LIBEXECDIR}/rocjitsu/rocjitsu-memory"
    )
    foreach(_target rocjitsu_memory rocjitsu_kmd_common)
        target_compile_definitions(
            ${_target}
            PRIVATE
                RJ_MEMORY_KEEPER_BUILD_LIBDIR="${_rj_keeper_build_libdir}"
                RJ_MEMORY_KEEPER_BUILD_BINDIR="${_rj_keeper_build_bindir}"
                RJ_MEMORY_KEEPER_BUILD_TESTDIR="${_rj_keeper_build_testdir}"
                RJ_MEMORY_KEEPER_BUILD_PATH="$<TARGET_FILE:rocjitsu_memory>"
                RJ_MEMORY_KEEPER_INSTALL_PATH="${_rj_keeper_relative}"
                RJ_MEMORY_KEEPER_BIN_INSTALL_PATH="${_rj_keeper_bin_relative}"
                RJ_MEMORY_KEEPER_TEST_INSTALL_PATH="${_rj_keeper_test_relative}"
        )
    endforeach()
    add_dependencies(rocjitsu_kmd_common rocjitsu_memory)
    install(
        TARGETS rocjitsu_memory
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocjitsu
    )
endif()
