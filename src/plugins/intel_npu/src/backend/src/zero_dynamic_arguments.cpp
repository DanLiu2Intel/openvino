// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_dynamic_arguments.hpp"

#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

void DynamicArguments::setArgumentProperties(uint32_t argi,
                                             const void* argv,
                                             const ov::Shape& sizes,
                                             const std::vector<size_t>& strides) {
    auto assign_slot = [&](VmMemRefDescriptor& slot) {
        slot._basePtr = slot._data = const_cast<void*>(argv);
        if (slot._dimsCount == 0) {
            slot._dimsCount = static_cast<int64_t>(sizes.size());
            slot._sizes.resize(sizes.size());
            slot._strides.resize(strides.size());
        } else if (slot._dimsCount != static_cast<int64_t>(sizes.size())) {
            OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                           slot._dimsCount,
                           ", new dimension count: ",
                           sizes.size());
        } else if (strides.size() != static_cast<size_t>(sizes.size())) {
            OPENVINO_THROW("Stride count mismatch. Current stride count: ",
                           strides.size(),
                           ", new stride count: ",
                           sizes.size());
        }
        for (int64_t i = 0; i < slot._dimsCount; i++) {
            slot._sizes[i] = static_cast<int64_t>(sizes[i]);
            slot._strides[i] = static_cast<int64_t>(strides[i]);
        }
    };

    if (argi < _inputs.size()) {
        assign_slot(_inputs[argi]);
    } else {
        auto idx = argi - _inputs.size();
        if (idx < _outputs.size()) {
            assign_slot(_outputs[idx]);
        }
    }
}

DynamicArguments::~DynamicArguments() {
    if (_executionContext != nullptr) {
        npuVMRuntimeDestroyExecutionContext(_executionContext);
        _executionContext = nullptr;
    }
}

void DynamicArguments::ensureExecutionContext(npu_vm_runtime_handle_t vmRuntime) {
    if (_executionContext != nullptr) {
        return;
    }
    if (npuVMRuntimeCreateExecutionContext(vmRuntime, &_executionContext) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create a VM execution context");
    }
}

}  // namespace intel_npu
