// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "intel_npu/config/config.hpp"
#include "intel_npu/network_metadata.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu {

#ifndef ICOMPILER_MAKE_VERSION
/// @brief Generates npu compiler (generic 'oneAPI') API version number
#    define ICOMPILER_MAKE_VERSION(_major, _minor) ((_major << 16) | (_minor & 0x0000ffff))
#endif  // ICOMPILER_MAKE_VERSION

/**
 * @struct NetworkDescription
 * @brief The object returned by the compiler
 * to provide such information about a network as description of inputs and outputs,
 * name and compiled network in a format executable by device
 */
struct NetworkDescription final {
    NetworkDescription(std::vector<uint8_t>&& compiledNetwork, NetworkMetadata&& metadata)
        : compiledNetwork(std::move(compiledNetwork)),
          metadata(std::move(metadata)) {}
    // Force move semantics to prevent blob copies
    NetworkDescription(const NetworkDescription&) = delete;
    NetworkDescription(NetworkDescription&&) = default;
    NetworkDescription& operator=(const NetworkDescription&) = delete;
    NetworkDescription& operator=(NetworkDescription&&) = default;
    ~NetworkDescription() = default;

    std::vector<uint8_t> compiledNetwork;

    NetworkMetadata metadata;
};


/// need update weightless part
class VCLCompilerImpl {
public:
    VCLCompilerImpl();
    ~VCLCompilerImpl() override;

    static std::shared_ptr<VCLCompilerImpl> getInstance() {
        static std::shared_ptr<VCLCompilerImpl> compiler = std::make_shared<VCLCompilerImpl>();
        return compiler;
    }

    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    NetworkMetadata parse(const std::vector<uint8_t>& network, const Config& config) const override;

    uint32_t get_version() const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const intel_npu::Config& config) const final override;

    bool get_supported_options(std::vector<char>& options) const;

    bool is_option_supported(const std::string& option) const;

    ////need to be updated weightless part
    /**
     * @brief Compiles the model, weights separation enabled. All init schedules along with the main one are compiled in
     * the same scope.
     * @return A "NetworkDescription" object for each init schedule, followed by another one corresponding to the main
     * part.
     */
    std::vector<std::shared_ptr<NetworkDescription>> compileWsOneShot(
        const std::shared_ptr<ov::Model>& /*model*/,
        const Config& /*config*/) const {
        OPENVINO_NOT_IMPLEMENTED;
    }

    /**
     * @brief Sequential compilation of Init(s) and Main
     *
     * "Stateless compiler" approach
     * We want to get multiple Inits in the case of a large number of weights.
     * This allows us to build pipeline:
     * Allocate W1 -> Init1
     *             Allocate W2 -> Init2
     *                          Allocate W3 -> Init2
     *
     * This is why there is an additional parameter callNumber:
     * Compiler should somehow understand wich Init(or Main) to return
     * Plugin does not know total numbers of Init schedules
     */
    NetworkDescription compileWsIterative(const std::shared_ptr<ov::Model>& /*model*/,
                                                  const Config& /*config*/,
                                                  size_t /*callNumber*/) const {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    std::shared_ptr<VCLApi> _vclApi;
    vcl_log_handle_t _logHandle = nullptr;
    vcl_compiler_handle_t _compilerHandle = nullptr;
    vcl_compiler_properties_t _compilerProperties;
    vcl_version_info_t _vclVersion;
    vcl_version_info_t _vclProfilingVersion;
    Logger _logger;
};

}  // namespace intel_npu
