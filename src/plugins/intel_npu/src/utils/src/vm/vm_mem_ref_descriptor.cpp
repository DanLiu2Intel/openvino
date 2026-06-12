// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vm/vm_mem_ref_descriptor.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace intel_npu {

void VmMemRefDescriptor::setArg(const void* arg, int64_t offset) {
    _basePtr = _data = arg;
    _offset = offset;
}

void VmMemRefDescriptor::setSize(const ov::Shape& shape) {
    if (_dimsCount == 0) {
        _dimsCount = static_cast<uint32_t>(shape.size());
        _sizes.resize(shape.size());
        _strides.resize(shape.size());
    } else if (_dimsCount != static_cast<int64_t>(shape.size())) {
        OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                       _dimsCount,
                       ", new dimension count: ",
                       shape.size());
    }

    for (int64_t i = 0; i < _dimsCount; ++i) {
        _sizes[i] = static_cast<int64_t>(shape[i]);
    }
}

void VmMemRefDescriptor::setStrides(const ov::Strides& strides, int32_t elementSize) {
    if (_dimsCount == 0) {
        OPENVINO_THROW("Dimension count is zero, shall call setSize before setStrides");
    } else if (_dimsCount != static_cast<int64_t>(strides.size())) {
        OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                       _dimsCount,
                       ", new dimension count: ",
                       strides.size());
    }

    for (int64_t i = 0; i < _dimsCount; ++i) {
        _strides[i] = static_cast<int64_t>(strides[i] / elementSize);
    }
}

void VmMemRefDescriptor::setProperties(const void* arg, const ov::Shape& shape, const std::vector<size_t>& strides) {
    if (strides.size() != shape.size()) {
        OPENVINO_THROW("Stride count mismatch. Current stride count: ",
                       strides.size(),
                       ", new stride count: ",
                       shape.size());
    }

    _basePtr = _data = arg;
    if (_dimsCount == 0) {
        _dimsCount = static_cast<int64_t>(shape.size());
        _sizes.resize(shape.size());
        _strides.resize(shape.size());
    } else if (_dimsCount != static_cast<int64_t>(shape.size())) {
        OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                       _dimsCount,
                       ", new dimension count: ",
                       shape.size());
    }

    for (int64_t i = 0; i < _dimsCount; i++) {
        _sizes[i] = static_cast<int64_t>(shape[i]);
        _strides[i] = static_cast<int64_t>(strides[i]);
    }
}

void VmMemRefDescriptor::set(const void* arg, int64_t offset, std::shared_ptr<ov::ITensor> tensor) {
    _basePtr = _data = arg;
    _offset = offset;
    if (_dimsCount == 0) {
        _dimsCount = static_cast<uint32_t>(tensor->get_shape().size());
        _sizes.resize(_dimsCount);
        _strides.resize(_dimsCount);
    } else if (_dimsCount != static_cast<int64_t>(tensor->get_shape().size())) {
        OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                       _dimsCount,
                       ", new dimension count: ",
                       tensor->get_shape().size());
    }

    auto& shape = tensor->get_shape();
    for (int64_t j = 0; j < _dimsCount; j++) {
        _sizes[j] = static_cast<int64_t>(shape[j]);
    }

    auto& strides = tensor->get_strides();
    if (_dimsCount != static_cast<int64_t>(strides.size())) {
        OPENVINO_THROW("Stride count mismatch. Current dimension count: ",
                       _dimsCount,
                       ", stride count: ",
                       strides.size());
    }
    size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
    for (int64_t j = 0; j < _dimsCount; j++) {
        OPENVINO_ASSERT(strides[j] % elementSize == 0,
                        "Stride ",
                        strides[j],
                        " bytes is not aligned to element size ",
                        elementSize,
                        " bytes. Strides must be multiples of element size.");
        _strides[j] = static_cast<int64_t>(strides[j] / elementSize);
    }
}

void VmMemRefDescriptor::updateStride() {
    // Note: NCHW layout style
    uint64_t stride = 1;
    for (int64_t i = _dimsCount - 1; i >= 0; --i) {
        _strides[i] = stride;
        stride *= _sizes[i];
    }
}

void VmMemRefDescriptor::copyShapeAndStridesFrom(const VmMemRefDescriptor& memRef) {
    _dimsCount = memRef._dimsCount;
    _sizes = memRef._sizes;
    _strides = memRef._strides;
}

ov::Shape VmMemRefDescriptor::getShape() const {
    ov::Shape shape;
    shape.reserve(_sizes.size());
    for (int64_t size : _sizes) {
        shape.push_back(static_cast<size_t>(size));
    }
    return shape;
}

bool VmMemRefDescriptor::compare(const VmMemRefDescriptor& memRef) const {
    if (memRef._dimsCount != _dimsCount || _sizes.size() != memRef._sizes.size() ||
        _strides.size() != memRef._strides.size())
        return false;
    size_t dimsCount = static_cast<size_t>(_dimsCount);
    if (memRef._sizes.size() != dimsCount || memRef._strides.size() != dimsCount)
        return false;
    for (size_t i = 0; i < dimsCount; i++) {
        if (_sizes[i] != memRef._sizes[i] || _strides[i] != memRef._strides[i]) {
            return false;
        }
    }
    return true;
}

bool VmMemRefDescriptor::hasUpdates() const noexcept {
    return _ptrUpdated || _shapeUpdated || _strideUpdated;
}

std::ostream& operator<<(std::ostream& os, const VmMemRefDescriptor& memRef) {
    os << "BasePtr: " << memRef._basePtr << ", Data: " << memRef._data << ", Offset: " << memRef._offset
       << ", Sizes: [";
    for (int64_t size : memRef._sizes) {
        os << size << " ";
    }
    os << "], Strides: [";
    for (int64_t stride : memRef._strides) {
        os << stride << " ";
    }
    os << "]";

    return os;
}

std::string VmMemRefDescriptor::toString() const {
    std::stringstream stream;
    stream << *this;
    return stream.str();
}

void VmMemRefDescriptor::updateMemRefHandleStatus() {
    if (!_memRef) {
        _memRef.create(_dimsCount);
    } else {
        const void* deviceBasePtr = nullptr;
        const void* deviceData = nullptr;
        int64_t deviceOffset = 0;
        std::vector<int64_t> deviceSizes(_sizes.size());
        std::vector<int64_t> deviceStrides(_strides.size());
        int64_t deviceDimsCount = 0;
        _memRef.parse(&deviceBasePtr,
                      &deviceData,
                      &deviceOffset,
                      deviceSizes.data(),
                      deviceStrides.data(),
                      &deviceDimsCount);
        _ptrUpdated = (_basePtr != deviceBasePtr || _data != deviceData || _offset != deviceOffset);
        _shapeUpdated = (_sizes != deviceSizes);
        _strideUpdated = (_strides != deviceStrides);
    }
    _memRef.set(_basePtr, _data, _offset, _sizes.data(), _strides.data(), _dimsCount);
}

void VmMemRefDescriptor::alignWithHandle() {
    if (!_memRef) {
        return;
    }
    _memRef.parse(&_basePtr, &_data, &_offset, _sizes.data(), _strides.data(), &_dimsCount);
}

npu_vm_runtime_mem_ref_handle_t VmMemRefDescriptor::getMemRefHandle() const noexcept {
    return _memRef.get();
}

}  // namespace intel_npu
