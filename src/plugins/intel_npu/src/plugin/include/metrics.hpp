// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "backends.hpp"
#include "npu.hpp"
#include "npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

class Metrics final {
public:
    Metrics(const std::shared_ptr<const NPUBackends>& backends);

    std::vector<std::string> GetAvailableDevicesNames() const;
    const std::vector<std::string>& SupportedMetrics() const;
    std::string GetFullDeviceName(const std::string& specifiedDeviceName) const;
    IDevice::Uuid GetDeviceUuid(const std::string& specifiedDeviceName) const;
    const std::vector<std::string>& GetSupportedConfigKeys() const;
    static const std::vector<std::string>& GetOptimizationCapabilities() {
        return _optimizationCapabilities;
    }
    static const std::tuple<uint32_t, uint32_t, uint32_t>& GetRangeForAsyncInferRequest() {
        return _rangeForAsyncInferRequests;
    }
    static const std::tuple<uint32_t, uint32_t>& GetRangeForStreams() {
        _rangeForStreams;
    }
    std::string GetDeviceArchitecture(const std::string& specifiedDeviceName) const;
    std::string GetBackendName() const;
    uint64_t GetDeviceAllocMemSize(const std::string& specifiedDeviceName) const;
    uint64_t GetDeviceTotalMemSize(const std::string& specifiedDeviceName) const;
    uint32_t GetDriverVersion() const;
    uint32_t GetDriverExtVersion() const;
    uint32_t GetSteppingNumber(const std::string& specifiedDeviceName) const;
    uint32_t GetMaxTiles(const std::string& specifiedDeviceName) const;
    ov::device::PCIInfo GetPciInfo(const std::string& specifiedDeviceName) const;
    std::map<ov::element::Type, float> GetGops(const std::string& specifiedDeviceName) const;
    ov::device::Type GetDeviceType(const std::string& specifiedDeviceName) const;

    static const std::vector<ov::PropertyName> GetCachingProperties() {
        return _cachingProperties;
    }
    static const std::vector<ov::PropertyName> GetInternalSupportedProperties() {
        return _internalSupportedProperties;
    }

    ~Metrics() = default;

private:
    const std::shared_ptr<const NPUBackends> _backends;
    std::vector<std::string> _supportedMetrics;
    std::vector<std::string> _supportedConfigKeys;
    static std::vector<std::string> _optimizationCapabilities;
    static std::vector<ov::PropertyName> _cachingProperties;

    static std::vector<ov::PropertyName> _internalSupportedProperties;

    // Metric to provide a hint for a range for number of async infer requests. (bottom bound, upper bound, step)
    static std::tuple<uint32_t, uint32_t, uint32_t> _rangeForAsyncInferRequests;

    // Metric to provide information about a range for streams.(bottom bound, upper bound)
    static std::tuple<uint32_t, uint32_t> _rangeForStreams;

    std::string getDeviceName(const std::string& specifiedDeviceName) const;
};

 std::vector<std::string> Metrics::_optimizationCapabilities = {
        ov::device::capability::FP16,
        ov::device::capability::INT8,
        ov::device::capability::EXPORT_IMPORT,
    };
 std::vector<ov::PropertyName> Metrics::_cachingProperties = {ov::device::architecture.name(),
                                                              ov::intel_npu::compilation_mode_params.name(),
                                                              ov::intel_npu::tiles.name(),
                                                              ov::intel_npu::dpu_groups.name(),
                                                              ov::intel_npu::dma_engines.name(),
                                                              ov::intel_npu::compilation_mode.name(),
                                                              ov::intel_npu::driver_version.name(),
                                                              ov::intel_npu::compiler_type.name(),
                                                              ov::intel_npu::use_elf_compiler_backend.name(),
                                                              ov::intel_npu::batch_mode.name(),
                                                              ov::hint::execution_mode.name()};

std::vector<ov::PropertyName> Metrics::_internalSupportedProperties = {ov::internal::caching_properties.name()};

// Metric to provide a hint for a range for number of async infer requests. (bottom bound, upper bound, step)
std::tuple<uint32_t, uint32_t, uint32_t> Metrics::_rangeForAsyncInferRequests = {1u, 10u, 1u};

// Metric to provide information about a range for streams.(bottom bound, upper bound)
std::tuple<uint32_t, uint32_t> Metrics::_rangeForStreams = {1u, 4u};


}  // namespace intel_npu
