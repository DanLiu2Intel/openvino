// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/runtime/npu_vm_runtime.hpp"

namespace intel_npu {

class VmExecutionContext final {
public:
    VmExecutionContext() = default;
    VmExecutionContext(const VmExecutionContext&) = delete;
    VmExecutionContext& operator=(const VmExecutionContext&) = delete;
    VmExecutionContext(VmExecutionContext&& other) noexcept;
    VmExecutionContext& operator=(VmExecutionContext&& other) noexcept;
    ~VmExecutionContext();

    explicit operator bool() const noexcept;
    npu_vm_runtime_execution_context_handle_t get() const noexcept;

    void create(npu_vm_runtime_handle_t vmRuntime);
    void reset();

private:
    npu_vm_runtime_execution_context_handle_t _handle = nullptr;
};

}  // namespace intel_npu
