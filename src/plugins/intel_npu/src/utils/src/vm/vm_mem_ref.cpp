// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vm/vm_mem_ref.hpp"

#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

VmMemRef::VmMemRef(VmMemRef&& other) noexcept : _handle(other._handle) {
    other._handle = nullptr;
}

VmMemRef& VmMemRef::operator=(VmMemRef&& other) noexcept {
    if (this != &other) {
        reset();
        _handle = other._handle;
        other._handle = nullptr;
    }
    return *this;
}

VmMemRef::~VmMemRef() {
    reset();
}

VmMemRef::operator bool() const noexcept {
    return _handle != nullptr;
}

npu_vm_runtime_mem_ref_handle_t VmMemRef::get() const noexcept {
    return _handle;
}

void VmMemRef::create(int64_t dimsCount) {
    if (_handle != nullptr) {
        return;
    }

    auto result = npuVMRuntimeCreateMemRef(dimsCount, &_handle);
    if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create MemRef handle");
    }
}

void VmMemRef::set(const void* basePtr,
                   const void* data,
                   int64_t offset,
                   int64_t* sizes,
                   int64_t* strides,
                   int64_t dimsCount) {
    create(dimsCount);

    auto result = npuVMRuntimeSetMemRef(_handle, basePtr, data, offset, sizes, strides, dimsCount);
    if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to update MemRef handle");
    }
}

void VmMemRef::parse(const void** basePtr,
                     const void** data,
                     int64_t* offset,
                     int64_t* sizes,
                     int64_t* strides,
                     int64_t* dimsCount) const {
    if (_handle == nullptr) {
        OPENVINO_THROW("Cannot parse an empty MemRef handle");
    }

    if (npuVMRuntimeParseMemRef(_handle, basePtr, data, offset, sizes, strides, dimsCount) !=
        NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to parse MemRef handle");
    }
}

void VmMemRef::reset() {
    if (_handle != nullptr) {
        npuVMRuntimeDestroyMemRef(_handle);
        _handle = nullptr;
    }
}

}  // namespace intel_npu
