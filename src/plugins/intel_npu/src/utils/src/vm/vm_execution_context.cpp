// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vm/vm_execution_context.hpp"

#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

VmExecutionContext::VmExecutionContext(VmExecutionContext&& other) noexcept : _handle(other._handle) {
    other._handle = nullptr;
}

VmExecutionContext& VmExecutionContext::operator=(VmExecutionContext&& other) noexcept {
    if (this != &other) {
        reset();
        _handle = other._handle;
        other._handle = nullptr;
    }
    return *this;
}

VmExecutionContext::~VmExecutionContext() {
    reset();
}

VmExecutionContext::operator bool() const noexcept {
    return _handle != nullptr;
}

npu_vm_runtime_execution_context_handle_t VmExecutionContext::get() const noexcept {
    return _handle;
}

void VmExecutionContext::create(npu_vm_runtime_handle_t vmRuntime) {
    if (_handle != nullptr) {
        return;
    }
    if (npuVMRuntimeCreateExecutionContext(vmRuntime, &_handle) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create a VM execution context");
    }
}

void VmExecutionContext::reset() {
    if (_handle != nullptr) {
        npuVMRuntimeDestroyExecutionContext(_handle);
        _handle = nullptr;
    }
}

}  // namespace intel_npu
