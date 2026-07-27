# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_kmd_device
    simdojo_device_backend.cpp
)
target_include_directories(rocjitsu_kmd_device PRIVATE ${GENERATED_DIR})
