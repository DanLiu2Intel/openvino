// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//

#include "npu_vm_predict_output_shape.hpp"

namespace intel_npu {

void vm_predict_output_shape(npu_vm_runtime_handle_t engine,
                      const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                      IDynamicGraph::GraphArguments& args,
                      std::vector<ze_command_list_handle_t>& commandLists,
                      ze_command_queue_handle_t commandQueue,
                      ze_fence_handle_t fence,
                      ze_event_handle_t event) {

}

}  // namespace intel_npu
