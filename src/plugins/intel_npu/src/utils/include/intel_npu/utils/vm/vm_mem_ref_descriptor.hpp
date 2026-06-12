// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
 * @brief Host-side VM MemRef descriptor plus the runtime MemRef handle that mirrors it.
 * @details The descriptor stores pointer, shape, and stride state in VM MemRef terms.
 * It can be populated from OpenVINO tensor metadata and synchronized with the VM runtime handle.
 */
struct VmMemRefDescriptor {
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

    VmMemRefDescriptor() = default;
    VmMemRefDescriptor(const VmMemRefDescriptor&) = delete;
    VmMemRefDescriptor& operator=(const VmMemRefDescriptor&) = delete;
    VmMemRefDescriptor(VmMemRefDescriptor&& other) noexcept = default;
    VmMemRefDescriptor& operator=(VmMemRefDescriptor&& other) noexcept = default;
    ~VmMemRefDescriptor() = default;

    void setArg(const void* arg);
    void setSize(const ov::Shape& shape);
    void setStrides(const ov::Strides& strides, int32_t elementSize = 1);
    void set(const void* basePtr, int64_t offset, std::shared_ptr<ov::ITensor> tensor);
    void updateStride();
    bool compare(const VmMemRefDescriptor& memRef) const;

    void updateMemRefHandleStatus();
    void alignWithHandle();
    npu_vm_runtime_mem_ref_handle_t getMemRefHandle() const noexcept;

    friend std::ostream& operator<<(std::ostream& os, const VmMemRefDescriptor& memRef);
    std::string toString() const;
};

}  // namespace intel_npu
