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
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/runtime/itensor.hpp"

namespace intel_npu {

struct DynamicMemRefType {
    const void* _basePtr;
    const void* _data;
    int64_t _offset;
    std::vector<int64_t> _sizes;
    std::vector<int64_t> _strides;
    int64_t _dimsCount;
    std::shared_ptr<void> _impl;

    DynamicMemRefType() : _basePtr(nullptr), _data(nullptr), _offset(0), _sizes(), _strides(), _dimsCount(0) {}

    DynamicMemRefType(const void* basePtr,
                      const void* data,
                      int64_t offset,
                      const std::vector<int64_t>& sizes,
                      const std::vector<int64_t>& strides,
                      int64_t dimsCount)
        : _basePtr(basePtr),
          _data(data),
          _offset(offset),
          _sizes(sizes),
          _strides(strides),
          _dimsCount(dimsCount) {}

    // Copy intentionally drops the runtime impl: VM MemRef handles must not be aliased
    // across copies (would cause double-destroy / shared device state).
    DynamicMemRefType(const DynamicMemRefType& other)
        : _basePtr(other._basePtr),
          _data(other._data),
          _offset(other._offset),
          _sizes(other._sizes),
          _strides(other._strides),
          _dimsCount(other._dimsCount) {}
    DynamicMemRefType& operator=(const DynamicMemRefType& other) {
        if (this != &other) {
            _basePtr = other._basePtr;
            _data = other._data;
            _offset = other._offset;
            _sizes = other._sizes;
            _strides = other._strides;
            _dimsCount = other._dimsCount;
            _impl.reset();
        }
        return *this;
    }
    DynamicMemRefType(DynamicMemRefType&&) noexcept = default;
    DynamicMemRefType& operator=(DynamicMemRefType&&) noexcept = default;
    ~DynamicMemRefType() = default;

    // /// Return the runtime-side impl, lazily creating it on first access.
    // DynamicMemRefImpl& ensure_impl();

    void setArg(const void* arg);
    void setSize(const ov::Shape& shape);
    void setStrides(const ov::Strides& strides, int32_t elementSize = 1);
    void set(const void* basePtr, int64_t offset, std::shared_ptr<ov::ITensor> tensor);
    void updateStride();
    bool compare(const DynamicMemRefType& memref);
    friend std::ostream& operator<<(std::ostream& os, const DynamicMemRefType& memRef);
    std::string toString();
};

    struct DynamicArguments {
    std::vector<DynamicMemRefType> _inputs;
    std::vector<DynamicMemRefType> _outputs;
    std::shared_ptr<void> _impl;

    DynamicArguments() = default;
    DynamicArguments(const DynamicArguments&) = delete;
    DynamicArguments& operator=(const DynamicArguments&) = delete;
    DynamicArguments(DynamicArguments&&) noexcept = default;
    DynamicArguments& operator=(DynamicArguments&&) noexcept = default;
    ~DynamicArguments() = default;

    // // /// Return the runtime-side impl, lazily creating it on first access.
    // // DynamicArgumentsImpl& ensure_impl();
    // //obtained by igraph
    // void ensureExecutionContext(npu_vm_runtime_handle_t vmRuntime);

    void setArgumentProperties(uint32_t argi,
                               const void* argv,
                               const ov::Shape& shapes,
                               const std::vector<size_t>& strides);
};
}  // namespace intel_npu
