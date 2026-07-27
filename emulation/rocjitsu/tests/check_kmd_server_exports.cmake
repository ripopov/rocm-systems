# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT LIBRARY OR NOT NM)
    message(FATAL_ERROR "LIBRARY and NM are required")
endif()

execute_process(
    COMMAND "${NM}" -D --defined-only --format=posix "${LIBRARY}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE symbols
    ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "nm failed: ${error}")
endif()

set(
    allowed
    rj_kmd_server_create
    rj_kmd_server_destroy
    rj_kmd_server_listen
    rj_kmd_server_stop_accepting
    rj_kmd_server_poll_fds
    rj_kmd_server_process_fd
    rj_kmd_server_notification_fd
    rj_kmd_server_drain_doorbells
    rj_kmd_server_post_interrupt
    rj_kmd_server_get_stats
)

string(REPLACE "\n" ";" lines "${symbols}")
foreach(line IN LISTS lines)
    if(line MATCHES "^([^ ]+) ")
        set(symbol "${CMAKE_MATCH_1}")
        string(REGEX REPLACE "@.*$" "" symbol "${symbol}")
        if(symbol MATCHES "^rj_")
            list(FIND allowed "${symbol}" index)
            if(index EQUAL -1)
                message(FATAL_ERROR "unexpected exported symbol: ${symbol}")
            endif()
            list(REMOVE_ITEM allowed "${symbol}")
        elseif(NOT symbol MATCHES "^ROCJITSU_KMD_SERVER_")
            message(FATAL_ERROR "unexpected non-ABI export: ${symbol}")
        endif()
    endif()
endforeach()

if(allowed)
    message(FATAL_ERROR "missing exported symbols: ${allowed}")
endif()
