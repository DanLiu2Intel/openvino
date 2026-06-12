// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "intel_npu/runtime/npu_vm_runtime.hpp"

namespace intel_npu {

class VmMemRef final {
public:
    VmMemRef() = default;
    VmMemRef(const VmMemRef&) = delete;
    VmMemRef& operator=(const VmMemRef&) = delete;
    VmMemRef(VmMemRef&& other) noexcept;
    VmMemRef& operator=(VmMemRef&& other) noexcept;
    ~VmMemRef();

    explicit operator bool() const noexcept;
    npu_vm_runtime_mem_ref_handle_t get() const noexcept;

    void create(int64_t dimsCount);
    void set(const void* basePtr,
             const void* data,
             int64_t offset,
             int64_t* sizes,
             int64_t* strides,
             int64_t dimsCount);
    void parse(const void** basePtr,
               const void** data,
               int64_t* offset,
               int64_t* sizes,
               int64_t* strides,
               int64_t* dimsCount) const;
    void reset();

private:
    npu_vm_runtime_mem_ref_handle_t _handle = nullptr;
};

}  // namespace intel_npu
