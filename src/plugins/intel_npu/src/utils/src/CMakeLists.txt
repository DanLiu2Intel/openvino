# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(logger)
message("====>  does this be include? 6 6.1")
if(ENABLE_NPU_PLUGIN_ENGINE)
    message("====>  does this be include? 6.2")
    add_subdirectory(zero)
endif()
