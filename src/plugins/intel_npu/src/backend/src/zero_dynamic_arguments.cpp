// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_dynamic_arguments.hpp"

#include "openvino/core/except.hpp"

namespace intel_npu {

void DynamicPipelineArguments::setArgumentProperties(uint32_t argi,
                                                     const void* argv,
                                                     const ov::Shape& sizes,
                                                     const std::vector<size_t>& strides) {
    if (static_cast<size_t>(argi) < _inputs.size()) {
        _inputs[argi].setProperties(argv, sizes, strides);
    } else {
        auto idx = static_cast<size_t>(argi) - _inputs.size();
        if (idx < _outputs.size()) {
            _outputs[idx].setProperties(argv, sizes, strides);
        }
    }
}

void DynamicPipelineArguments::ensureExecutionContext(npu_vm_runtime_handle_t vmRuntime) {
    _executionContext.create(vmRuntime);
}

npu_vm_runtime_execution_context_handle_t DynamicPipelineArguments::getExecutionContextHandle() const noexcept {
    return _executionContext.get();
}

std::vector<VmMemRefDescriptor>& DynamicPipelineArguments::inputs() noexcept {
    return _inputs;
}

std::vector<VmMemRefDescriptor>& DynamicPipelineArguments::outputs() noexcept {
    return _outputs;
}

const std::vector<VmMemRefDescriptor>& DynamicPipelineArguments::inputs() const noexcept {
    return _inputs;
}

const std::vector<VmMemRefDescriptor>& DynamicPipelineArguments::outputs() const noexcept {
    return _outputs;
}

void DynamicPipelineArguments::resizeInputs(size_t size) {
    _inputs.resize(size);
}

void DynamicPipelineArguments::resizeOutputs(size_t size) {
    _outputs.resize(size);
}

bool DynamicPipelineArguments::updateMemRefHandles() {
    _inputMemRefs.clear();
    _outputMemRefs.clear();
    _inputMemRefs.reserve(_inputs.size());
    _outputMemRefs.reserve(_outputs.size());

    bool hasUpdates = false;
    for (auto& input : _inputs) {
        input.updateMemRefHandleStatus();
        _inputMemRefs.push_back(input.getMemRefHandle());
        hasUpdates = hasUpdates || input.hasUpdates();
    }
    for (auto& output : _outputs) {
        output.updateMemRefHandleStatus();
        _outputMemRefs.push_back(output.getMemRefHandle());
        hasUpdates = hasUpdates || output.hasUpdates();
    }

    return hasUpdates;
}

std::vector<npu_vm_runtime_mem_ref_handle_t>& DynamicPipelineArguments::inputMemRefs() noexcept {
    return _inputMemRefs;
}

std::vector<npu_vm_runtime_mem_ref_handle_t>& DynamicPipelineArguments::outputMemRefs() noexcept {
    return _outputMemRefs;
}

bool DynamicPipelineArguments::executedOnce() const noexcept {
    return _executedOnce;
}

void DynamicPipelineArguments::markExecuted() noexcept {
    _executedOnce = true;
}

}  // namespace intel_npu
