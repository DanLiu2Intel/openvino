# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(logger)

# Decouple from ENABLE_NPU_PLUGIN_ENGINE to satisfy npu_interpreter_runtime dependencies
add_subdirectory(zero)

# just for last ov, cherry-pick from OV PR:https://github.com/openvinotoolkit/openvino/pull/35227
# if(ENABLE_NPU_PLUGIN_ENGINE)
#     add_subdirectory(vm)
# endif()
