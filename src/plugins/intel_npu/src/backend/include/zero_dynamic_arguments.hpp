// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "intel_npu/runtime/npu_vm_runtime.hpp"
#include "intel_npu/utils/vm/vm_execution_context.hpp"
#include "intel_npu/utils/vm/vm_mem_ref_descriptor.hpp"
#include "openvino/core/shape.hpp"

namespace intel_npu {

/**
 * @brief Argument descriptors plus the runtime-side state used to invoke npuVMRuntimeExecute.
 * @details Owns the pipeline IO descriptors and cached VM execution context across
 * multiple executes. The context is created lazily via ensureExecutionContext on
 * the first execute call.
 */
struct DynamicPipelineArguments {
    DynamicPipelineArguments() = default;
    DynamicPipelineArguments(const DynamicPipelineArguments&) = delete;
    DynamicPipelineArguments& operator=(const DynamicPipelineArguments&) = delete;
    DynamicPipelineArguments(DynamicPipelineArguments&&) = delete;
    DynamicPipelineArguments& operator=(DynamicPipelineArguments&&) = delete;
    ~DynamicPipelineArguments() = default;

    /// Create the VM execution context for vmRuntime. No-op if already created.
    void ensureExecutionContext(npu_vm_runtime_handle_t vmRuntime);
    npu_vm_runtime_execution_context_handle_t getExecutionContextHandle() const noexcept;

    std::vector<VmMemRefDescriptor>& inputs() noexcept;
    std::vector<VmMemRefDescriptor>& outputs() noexcept;
    const std::vector<VmMemRefDescriptor>& inputs() const noexcept;
    const std::vector<VmMemRefDescriptor>& outputs() const noexcept;
    void resizeInputs(size_t size);
    void resizeOutputs(size_t size);

    void clearMemRefHandles();
    void reserveMemRefHandles();
    void addInputMemRefHandle(npu_vm_runtime_mem_ref_handle_t memRef);
    void addOutputMemRefHandle(npu_vm_runtime_mem_ref_handle_t memRef);
    npu_vm_runtime_mem_ref_handle_t* inputMemRefHandlesData() noexcept;
    npu_vm_runtime_mem_ref_handle_t* outputMemRefHandlesData() noexcept;
    uint32_t inputMemRefCount() const;
    uint32_t outputMemRefCount() const;

    bool executedOnce() const noexcept;
    void markExecuted() noexcept;

    void setArgumentProperties(uint32_t argi,
                               const void* argv,
                               const ov::Shape& shapes,
                               const std::vector<size_t>& strides);

private:
    std::vector<VmMemRefDescriptor> _inputs;
    std::vector<VmMemRefDescriptor> _outputs;
    std::vector<npu_vm_runtime_mem_ref_handle_t> _inputMemRefs;
    std::vector<npu_vm_runtime_mem_ref_handle_t> _outputMemRefs;
    VmExecutionContext _executionContext;
    // Set after the first successful @c npuVMRuntimeExecute.
    bool _executedOnce = false;
};

}  // namespace intel_npu
