# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Close LTO temporarily for add_symbol_to_partition_1 issue
set(ENABLE_LTO OFF)

add_subdirectory(utils)

add_subdirectory(al)

add_subdirectory(ops)

# Decouple from ENABLE_NPU_PLUGIN_ENGINE to satisfy npu_interpreter_runtime dependencies
add_subdirectory(vm)
if (ENABLE_NPU_PLUGIN_ENGINE)
    message("===> ov include dir: ENABLE_NPU_PLUGIN_ENGINE in src/plugins/intel_npu/src/CMakeLists.txt")
    add_subdirectory(common)
    add_subdirectory(compiler_adapter)
    add_subdirectory(backend)
    add_subdirectory(plugin)
endif()
