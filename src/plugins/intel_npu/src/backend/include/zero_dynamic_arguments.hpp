// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "intel_npu/runtime/npu_vm_runtime.hpp"
#include "intel_npu/utils/vm/vm_mem_ref_descriptor.hpp"
#include "openvino/core/shape.hpp"

namespace intel_npu {

/**
 * @brief Argument descriptors plus the runtime-side state used to invoke npuVMRuntimeExecute.
 * @details Owns the VM execution context across multiple executes (it caches device-side
 * state and must not be re-created per call). The context is created lazily via
 * ensureExecutionContext on the first execute call and destroyed by the destructor.
 */
struct DynamicArguments {
    std::vector<VmMemRefDescriptor> _inputs;
    std::vector<VmMemRefDescriptor> _outputs;
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
