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

void DynamicPipelineArguments::clearMemRefHandles() {
    _inputMemRefs.clear();
    _outputMemRefs.clear();
}

void DynamicPipelineArguments::reserveMemRefHandles() {
    _inputMemRefs.reserve(_inputs.size());
    _outputMemRefs.reserve(_outputs.size());
}

void DynamicPipelineArguments::addInputMemRefHandle(npu_vm_runtime_mem_ref_handle_t memRef) {
    _inputMemRefs.push_back(memRef);
}

void DynamicPipelineArguments::addOutputMemRefHandle(npu_vm_runtime_mem_ref_handle_t memRef) {
    _outputMemRefs.push_back(memRef);
}

npu_vm_runtime_mem_ref_handle_t* DynamicPipelineArguments::inputMemRefHandlesData() noexcept {
    return _inputMemRefs.data();
}

npu_vm_runtime_mem_ref_handle_t* DynamicPipelineArguments::outputMemRefHandlesData() noexcept {
    return _outputMemRefs.data();
}

uint32_t DynamicPipelineArguments::inputMemRefCount() const {
    return static_cast<uint32_t>(_inputMemRefs.size());
}

uint32_t DynamicPipelineArguments::outputMemRefCount() const {
    return static_cast<uint32_t>(_outputMemRefs.size());
}

bool DynamicPipelineArguments::executedOnce() const noexcept {
    return _executedOnce;
}

void DynamicPipelineArguments::markExecuted() noexcept {
    _executedOnce = true;
}

}  // namespace intel_npu
