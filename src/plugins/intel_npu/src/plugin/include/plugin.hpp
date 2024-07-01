// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "backends.hpp"
#include "intel_npu/al/config/config.hpp"
#include "intel_npu/al/icompiler.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "metrics.hpp"
#include "npu.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

class Plugin : public ov::IPlugin {
public:
    Plugin();

    Plugin(const Plugin&) = delete;

    Plugin& operator=(const Plugin&) = delete;

    virtual ~Plugin() = default;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& stream, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& stream,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    bool is_backends_empty() const {
        if (_backends == nullptr)
            return true;
        //_backends has been init, but not sure _backend in _backends is inited.
        return _backends->is_empty() ? true : false;
    }
    void update_supplement_properties() const;
    void update_BackendsAndMetrics(Config& config) const;
    const std::shared_ptr<IDevice> update_device(Config config) const {
        if (is_backends_empty()) {
            _logger.error(" no bakend. can not init device!");
        } else {
            return _backends->getDevice(config.get<DEVICE_ID>());
        }
    }

private:
    ov::SoPtr<ICompiler> getCompiler(const Config& config) const;

    mutable std::shared_ptr<NPUBackends> _backends;

    std::map<std::string, std::string> _config;
    std::shared_ptr<OptionsDesc> _options;
    mutable Config _globalConfig;
    Logger _logger;
    mutable std::unique_ptr<Metrics> _metrics;

    // properties map: {name -> [supported, mutable, eval function]}
    mutable std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>>
        _properties;
    mutable std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>>
        supplement_properties;

    mutable std::vector<ov::PropertyName> _supportedProperties;

    static std::atomic<int> _compiledModelLoadCounter;
};

}  // namespace intel_npu
