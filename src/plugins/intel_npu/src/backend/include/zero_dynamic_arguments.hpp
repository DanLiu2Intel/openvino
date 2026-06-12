// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "intel_npu/runtime/npu_vm_runtime.hpp"
#include "intel_npu/utils/vm/vm_mem_ref.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/runtime/itensor.hpp"

namespace intel_npu {

/**
 * @brief Host-side description of a single dynamic argument plus the VM-runtime
 * MemRef handle that mirrors it on the device.
 * @details Shape, stride, and pointer state belong to the Zero dynamic pipeline;
 * the raw VM handle lifetime is owned by VmMemRef.
 */
struct DynamicMemRefType {
    const void* _basePtr = nullptr;
    const void* _data = nullptr;
    int64_t _offset = 0;
    std::vector<int64_t> _sizes;
    std::vector<int64_t> _strides;
    int64_t _dimsCount = 0;

    VmMemRef _memRef;
    // Set by updateMemRefHandleStatus to report what changed vs. the previous device-side value.
    bool _ptrUpdated = false;
    bool _shapeUpdated = false;
    bool _strideUpdated = false;

    DynamicMemRefType() = default;
    DynamicMemRefType(const DynamicMemRefType&) = delete;
    DynamicMemRefType& operator=(const DynamicMemRefType&) = delete;
    DynamicMemRefType(DynamicMemRefType&& other) noexcept = default;
    DynamicMemRefType& operator=(DynamicMemRefType&& other) noexcept = default;
    ~DynamicMemRefType() = default;

    void setArg(const void* arg);
    void setSize(const ov::Shape& shape);
    void setStrides(const ov::Strides& strides, int32_t elementSize = 1);
    void set(const void* basePtr, int64_t offset, std::shared_ptr<ov::ITensor> tensor);
    void updateStride();
    bool compare(const DynamicMemRefType& memref);

    void updateMemRefHandleStatus();
    void alignWithHandle();
    npu_vm_runtime_mem_ref_handle_t getMemRefHandle() const noexcept;

    friend std::ostream& operator<<(std::ostream& os, const DynamicMemRefType& memRef);
    std::string toString();
};

/**
 * @brief Argument descriptors plus the runtime-side state used to invoke npuVMRuntimeExecute.
 * @details Owns the VM execution context across multiple executes (it caches device-side
 * state and must not be re-created per call). The context is created lazily via
 * ensureExecutionContext on the first execute call and destroyed by the destructor.
 */
struct DynamicArguments {
    std::vector<DynamicMemRefType> _inputs;
    std::vector<DynamicMemRefType> _outputs;
    std::vector<npu_vm_runtime_mem_ref_handle_t> _inputMemRefs;
    std::vector<npu_vm_runtime_mem_ref_handle_t> _outputMemRefs;
    npu_vm_runtime_execution_context_handle_t _executionContext = nullptr;
    // Set by the caller after the first successful @c npuVMRuntimeExecute.
    bool _executedOnce = false;

    DynamicArguments() = default;
    DynamicArguments(const DynamicArguments&) = delete;
    DynamicArguments& operator=(const DynamicArguments&) = delete;
    DynamicArguments(DynamicArguments&&) = delete;
    DynamicArguments& operator=(DynamicArguments&&) = delete;
    ~DynamicArguments();

    /// Create the VM execution context for vmRuntime. No-op if already created.
    void ensureExecutionContext(npu_vm_runtime_handle_t vmRuntime);

    void setArgumentProperties(uint32_t argi,
                               const void* argv,
                               const ov::Shape& shapes,
                               const std::vector<size_t>& strides);
};

}  // namespace intel_npu
