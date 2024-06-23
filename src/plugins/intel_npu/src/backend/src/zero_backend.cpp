// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_backend.hpp"

#include <vector>

#include "intel_npu/al/config/common.hpp"
#include "zero_device.hpp"

namespace intel_npu {

ZeroEngineBackend::ZeroEngineBackend(const Config& config) {
    Logger::global().setLevel(config.get<LOG_LEVEL>());
    std::printf("====6.23====ZeroEngineBackend constructor=========config.get<LOG_LEVEL>()=%d\n", static_cast<int>(config.get<LOG_LEVEL>()));
    //sLogger::global().setLevel(config.get<LOG_LEVEL>());//因为在plugin constructor 统一过了，就似乎不会影响这个的内容
    //拿到的log_level, 如果config.get<LOG_LEVEL>()没有设置环境变量，默认值返回的是什么？
    //问题在于会将global()的值修改吗?

    //config的默认值不是NO吗？

    _instance = std::make_shared<ZeroInitStructsHolder>();

    auto device = std::make_shared<ZeroDevice>(_instance);
    _devices.emplace(std::make_pair(device->getName(), device));
}

uint32_t ZeroEngineBackend::getDriverVersion() const {
    return _instance->getDriverVersion();
}

uint32_t ZeroEngineBackend::getDriverExtVersion() const {
    return _instance->getDriverExtVersion();
}

bool ZeroEngineBackend::isBatchingSupported() const {
    return _instance->getDriverExtVersion() >= ZE_GRAPH_EXT_VERSION_1_6;
}

ZeroEngineBackend::~ZeroEngineBackend() = default;

const std::shared_ptr<IDevice> ZeroEngineBackend::getDevice() const {
    if (_devices.empty()) {
        return {};
    } else {
        return _devices.begin()->second;
    }
}

const std::shared_ptr<IDevice> ZeroEngineBackend::getDevice(const std::string& /*name*/) const {
    // TODO Add the search of the device by platform & slice
    return getDevice();
}

const std::vector<std::string> ZeroEngineBackend::getDeviceNames() const {
    std::vector<std::string> devicesNames;
    std::for_each(_devices.cbegin(), _devices.cend(), [&devicesNames](const auto& device) {
        devicesNames.push_back(device.first);
    });
    return devicesNames;
}

}  // namespace intel_npu
