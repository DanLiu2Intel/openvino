// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include <string_view>

#include "graph.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "mem_usage.hpp"
#include "openvino/core/model.hpp"
#include "vcl_serializer.hpp"
#include "weightless_graph.hpp"
#include "weightless_utils.hpp"

namespace {

/**
 * @brief On-going migration from "use_base_model_serializer" to "model_serializer_version". So we have to check both,
 * depending on which one is supported by the compiler.
 */
bool useBaseModelSerializer(const intel_npu::FilteredConfig& config) {
    if (config.isAvailable(ov::intel_npu::use_base_model_serializer.name())) {
        return config.get<intel_npu::USE_BASE_MODEL_SERIALIZER>();
    }
    if (config.isAvailable(ov::intel_npu::model_serializer_version.name())) {
        return (config.get<intel_npu::MODEL_SERIALIZER_VERSION>() !=
                ov::intel_npu::ModelSerializerVersion::NO_WEIGHTS_COPY);
    }
    return true;
}

}  // namespace

namespace intel_npu {

DriverCompilerAdapter::DriverCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("DriverCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize DriverCompilerAdapter start");

    uint32_t graphExtVersion = _zeroInitStruct->getGraphDdiTable().version();

    _compilerProperties = _zeroInitStruct->getCompilerProperties();

    _logger.info("DriverCompilerAdapter creating adapter using graphExtVersion");

    _zeGraphExt = std::make_shared<ZeGraphExtWrappers>(_zeroInitStruct);

    _logger.info("initialize DriverCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

/**
 * @brief Print basic model information
 */
void print_basic_info(const std::shared_ptr<const ov::Model>& model) {
    std::cout << "=== Model Basic Information ===" << std::endl;
    std::cout << "Name: " << model->get_name() << std::endl;
    std::cout << "Friendly Name: " << model->get_friendly_name() << std::endl;
    std::cout << "Output Size: " << model->get_output_size() << std::endl;
    std::cout << "Graph Size: " << model->get_graph_size() << " bytes" << std::endl;
    std::cout << "Is Dynamic: " << (model->is_dynamic() ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Print model parameters (inputs)
 */
void print_parameters(const std::shared_ptr<const ov::Model>& model) {
    std::cout << "=== Model Parameters (Inputs) ===" << std::endl;
    const auto& parameters = model->get_parameters();
    std::cout << "Total Parameters: " << parameters.size() << std::endl;

    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& param = parameters[i];
        std::cout << "  [" << i << "] " << param->get_friendly_name() << " : " << param->get_element_type() << " "
                  << param->get_partial_shape() << std::endl;

        // Print additional parameter info
        std::cout << "      Type: " << param->get_type_name() << std::endl;
        if (param->get_output_size() > 0) {
            std::cout << "      Output tensor names: ";
            for (size_t j = 0; j < param->get_output_size(); ++j) {
                auto names = param->get_output_tensor(j).get_names();
                for (const auto& name : names) {
                    std::cout << name << " ";
                }
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

/**
 * @brief Print model results (outputs)
 */
void print_results(const std::shared_ptr<const ov::Model>& model) {
    std::cout << "=== Model Results (Outputs) ===" << std::endl;
    const auto& results = model->get_results();
    std::cout << "Total Results: " << results.size() << std::endl;

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        std::cout << "  [" << i << "] " << result->get_friendly_name() << std::endl;
        std::cout << "      Type: " << result->get_type_name() << std::endl;

        if (result->get_input_size() > 0) {
            const auto& input = result->get_input_source_output(0);
            std::cout << "      Element Type: " << input.get_element_type() << std::endl;
            std::cout << "      Shape: " << input.get_partial_shape() << std::endl;

            auto names = result->get_output_tensor(0).get_names();
            if (!names.empty()) {
                std::cout << "      Tensor names: ";
                for (const auto& name : names) {
                    std::cout << name << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

/**
 * @brief Print model variables
 */
void print_variables(const std::shared_ptr<const ov::Model>& model) {
    std::cout << "=== Model Variables ===" << std::endl;
    const auto& variables = model->get_variables();
    std::cout << "Total Variables: " << variables.size() << std::endl;

    for (size_t i = 0; i < variables.size(); ++i) {
        const auto& var = variables[i];
        const auto& info = var->get_info();
        std::cout << "  [" << i << "] ID: " << info.variable_id << std::endl;
        std::cout << "      Shape: " << info.data_shape << std::endl;
        std::cout << "      Type: " << info.data_type << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief Print model sinks
 */
void print_sinks(const std::shared_ptr<const ov::Model>& model) {
    std::cout << "=== Model Sinks ===" << std::endl;
    const auto& sinks = model->get_sinks();
    std::cout << "Total Sinks: " << sinks.size() << std::endl;

    for (size_t i = 0; i < sinks.size(); ++i) {
        const auto& sink = sinks[i];
        std::cout << "  [" << i << "] " << sink->get_friendly_name() << " (" << sink->get_type_name() << ")"
                  << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief Print runtime information
 */
void print_runtime_info(const std::shared_ptr<const ov::Model>& model) {
    std::cout << "=== Model Runtime Information ===" << std::endl;
    const auto& rt_info = model->get_rt_info();
    std::cout << "Runtime Info Entries: " << rt_info.size() << std::endl;

    for (const auto& kv : rt_info) {
        std::cout << "  " << kv.first << " = ";
        try {
            // Try to convert to string
            std::cout << kv.second.as<std::string>();
        } catch (...) {
            try {
                // Try to convert to int
                std::cout << kv.second.as<int>();
            } catch (...) {
                try {
                    // Try to convert to bool
                    std::cout << (kv.second.as<bool>() ? "true" : "false");
                } catch (...) {
                    std::cout << "[complex type]";
                }
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief Print all nodes in the model
 */
void print_all_nodes(const std::shared_ptr<const ov::Model>& model) {
    std::cout << "=== All Nodes (Detailed) ===" << std::endl;
    const auto& nodes = model->get_ordered_ops();
    std::cout << "Total Nodes: " << nodes.size() << std::endl;

    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        std::cout << "  [" << std::setw(3) << i << "] " << std::setw(20) << std::left << node->get_friendly_name()
                  << " (" << node->get_type_name() << ")" << std::endl;

        // Print inputs
        if (node->get_input_size() > 0) {
            std::cout << "       Inputs: ";
            for (size_t j = 0; j < node->get_input_size(); ++j) {
                const auto& input = node->get_input_source_output(j);
                std::cout << input.get_element_type() << input.get_partial_shape();
                if (j < node->get_input_size() - 1)
                    std::cout << ", ";
            }
            std::cout << std::endl;
        }

        // Print outputs
        if (node->get_output_size() > 0) {
            std::cout << "       Outputs: ";
            for (size_t j = 0; j < node->get_output_size(); ++j) {
                const auto& output = node->get_output_tensor(j);
                std::cout << output.get_element_type() << output.get_partial_shape();
                if (j < node->get_output_size() - 1)
                    std::cout << ", ";
            }
            std::cout << std::endl;
        }

        // Print node runtime info if exists
        const auto& node_rt_info = node->get_rt_info();
        if (!node_rt_info.empty()) {
            std::cout << "       RT Info: ";
            for (const auto& kv : node_rt_info) {
                std::cout << kv.first << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

/**
 * @brief Print graph statistics
 */
void print_graph_statistics(const std::shared_ptr<const ov::Model>& model) {
    std::cout << "=== Graph Statistics ===" << std::endl;
    const auto& nodes = model->get_ops();

    // Count nodes by type
    std::map<std::string, int> node_type_count;
    for (const auto& node : nodes) {
        node_type_count[node->get_type_name()]++;
    }

    std::cout << "Node Type Distribution:" << std::endl;
    for (const auto& kv : node_type_count) {
        std::cout << "  " << std::setw(20) << std::left << kv.first << ": " << kv.second << std::endl;
    }
    std::cout << std::endl;
}

void print_all_info(const std::shared_ptr<const ov::Model>& model) {
    if (!model) {
        std::cout << "Model is null!" << std::endl;
        return;
    }

    print_basic_info(model);
    print_parameters(model);
    print_results(model);
    print_variables(model);
    print_sinks(model);
    print_runtime_info(model);

    const char* detail = std::getenv("DETAIL");
    if (detail) {
        print_all_nodes(model);
    }

    print_graph_statistics(model);
}

// Control the indentation format
std::string getIndent(int level) {
    return std::string(level * 2, ' ');
}

// Get IODescriptor string
std::string ioDescriptorToString(const intel_npu::IODescriptor& desc, int index) {
    std::ostringstream ss;

    ss << getIndent(index) << "IODescriptor {\n";
    ss << getIndent(index + 1) << "nameFromCompiler: \"" << desc.nameFromCompiler << "\"\n";
    ss << getIndent(index + 1) << "precision: " << desc.precision.get_type_name() << "\n";
    ss << getIndent(index + 1) << "shapeFromCompiler: " << desc.shapeFromCompiler << "\n";
    ss << getIndent(index + 1) << "isStateInput: " << (desc.isStateInput ? "true" : "false") << "\n";
    ss << getIndent(index + 1) << "isStateOutput: " << (desc.isStateOutput ? "true" : "false") << "\n";
    ss << getIndent(index + 2) << "isShapeTensor: " << (desc.isShapeTensor ? "true" : "false") << "\n";
    ss << getIndent(index + 2) << "isInitInputWeights: " << (desc.isInitInputWeights ? "true" : "false") << "\n";
    ss << getIndent(index + 2) << "isInitOutputWeights: " << (desc.isInitOutputWeights ? "true" : "false") << "\n";
    ss << getIndent(index + 2) << "isMainInputWeights: " << (desc.isMainInputWeights ? "true" : "false") << "\n";

    if (desc.relatedDescriptorIndex.has_value()) {
        ss << getIndent(index + 2) << "relatedDescriptorIndex: " << desc.relatedDescriptorIndex.value() << "\n";
    } else {
        ss << getIndent(index + 2) << "relatedDescriptorIndex: null\n";
    }

    ss << getIndent(index + 2) << "nodeFriendlyName: \"" << desc.nodeFriendlyName << "\"\n";
    ss << getIndent(index + 2) << "outputTensorNames: [";
    bool first = true;
    for (const auto& name : desc.outputTensorNames) {
        if (!first) {
            ss << ", ";
        }
        ss << "\"" << name << "\"";
        first = false;
    }
    ss << "]\n";

    if (desc.shapeFromIRModel.has_value()) {
        ss << getIndent(index + 2) << "shapeFromIRModel: " << desc.shapeFromIRModel.value() << "\n";
    } else {
        ss << getIndent(index + 2) << "shapeFromIRModel: null\n";
    }

    ss << getIndent(index) << "}";

    return ss.str();
}

// Helper function to add indentation to each line of a string
std::string addIndentationToString(const std::string& inputStr, const std::string& baseIndent) {
    std::ostringstream ss;
    std::istringstream stream(inputStr);
    std::string line;

    while (std::getline(stream, line)) {
        ss << baseIndent << line;
        if (!stream.eof()) {
            ss << "\n";
        }
    }

    return ss.str();
}

// Helper function to add IODescriptor vector to string
std::string addIoDescVectorToString(const std::vector<intel_npu::IODescriptor>& ioDescriptorVector) {
    std::ostringstream ss;
    for (size_t i = 0; i < ioDescriptorVector.size(); ++i) {
        std::string inputStr = ioDescriptorToString(ioDescriptorVector[i], 2);
        ss << addIndentationToString(inputStr, "    ");
        if (i < ioDescriptorVector.size() - 1) {
            ss << ",";
        }
        ss << "\n";
    }
    return ss.str();
}

// Get NetworkMetadata string
std::string networkMetadataToString(const intel_npu::NetworkMetadata& netMetadata) {
    std::ostringstream ss;

    ss << "NetworkMetadata {\n";
    ss << "  name: \"" << netMetadata.name << "\"\n";
    ss << "  numStreams: " << netMetadata.numStreams << "\n";
    ss << "  inputs: [\n" << addIoDescVectorToString(netMetadata.inputs) << "  ]\n";
    ss << "  outputs: [\n" << addIoDescVectorToString(netMetadata.outputs) << "  ]\n";

    if (!netMetadata.profilingOutputs.empty()) {
        ss << "  profilingOutputs: [\n" << addIoDescVectorToString(netMetadata.profilingOutputs) << "  ]\n";
    }
    ss << "}";

    return ss.str();
}

std::shared_ptr<IGraph> DriverCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                       const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compile");

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");

    auto serializedIR = driver_compiler_utils::serializeIR(model,
                                                           compilerVersion,
                                                           maxOpsetVersion,
                                                           useBaseModelSerializer(config),
                                                           _zeGraphExt->isPluginModelHashSupported());

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));

    _logger.debug("build flags");
    buildFlags += driver_compiler_utils::serializeIOInfo(model, useIndices);
    buildFlags += " ";
    buildFlags += driver_compiler_utils::serializeConfig(config,
                                                         compilerVersion,
                                                         _zeGraphExt->isTurboOptionSupported(compilerVersion));

    _logger.debug("compileIR Build flags : %s", buildFlags.c_str());
    std::cout << "------adapter1---------" << std::endl;
    print_all_info(model);
    std::cout << "------adapter2---------" << std::endl;

    _logger.debug("compile start");
    // If UMD Caching is requested to be bypassed or if OV cache is enabled, disable driver caching
    const bool bypassCache = !config.get<CACHE_DIR>().empty() || config.get<BYPASS_UMD_CACHING>();
    auto graphDesc = _zeGraphExt->getGraphDescriptor(std::move(serializedIR), buildFlags, bypassCache);
    _logger.debug("compile end");

    OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeGraphExt->getNetworkMeta(graphDesc);
    networkMeta.name = model->get_friendly_name();

    std::cout << "------adapter3---------" << std::endl;
    std::cout << networkMetadataToString(networkMeta) << std::endl;
    std::cout << "------adapter4---------" << std::endl;
    print_all_info(model);
    std::cout << "------adapter5---------" << std::endl;

    return std::make_shared<Graph>(_zeGraphExt,
                                   _zeroInitStruct,
                                   graphDesc,
                                   std::move(networkMeta),
                                   /* blob = */ std::nullopt,
                                   config);
}

std::shared_ptr<IGraph> DriverCompilerAdapter::compileWS(const std::shared_ptr<ov::Model>& model,
                                                         const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compileWS");

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    if ((compilerVersion.major < 6) || (compilerVersion.major == 6 && compilerVersion.minor < 3)) {
        OPENVINO_THROW("Minimum compiler version required for weights separation: 6.3. Found: ",
                       compilerVersion.major,
                       ".",
                       compilerVersion.minor);
    }

    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    if (config.get<SEPARATE_WEIGHTS_VERSION>() != ov::intel_npu::WSVersion::ITERATIVE) {
        OPENVINO_THROW("Invalid \"SEPARATE_WEIGHTS_VERSION\" value found within the \"compileWS\" call:",
                       config.get<SEPARATE_WEIGHTS_VERSION>(),
                       ". \"WSVersion::ITERATIVE\" is the only supported value for the compiler-in-driver path.");
    }

    _logger.debug("serialize IR");
    auto serializedIR = driver_compiler_utils::serializeIR(model,
                                                           compilerVersion,
                                                           maxOpsetVersion,
                                                           useBaseModelSerializer(config),
                                                           true,
                                                           _zeGraphExt->isPluginModelHashSupported());

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));

    const std::string serializedIOInfo = driver_compiler_utils::serializeIOInfo(model, useIndices);
    const FilteredConfig* plgConfig = dynamic_cast<const FilteredConfig*>(&config);
    if (plgConfig == nullptr) {
        OPENVINO_THROW("config is not FilteredConfig");
    }
    FilteredConfig updatedConfig = *plgConfig;

    // WS v3 is based on a stateless compiler. We'll use a separate config entry for informing the compiler the index of
    // the current call iteration.
    std::vector<NetworkMetadata> initNetworkMetadata;
    NetworkMetadata mainNetworkMetadata;
    std::vector<GraphDescriptor> initGraphDescriptors;
    GraphDescriptor mainGraphHandle;
    size_t callNumber = 0;

    // Convention: run until the main schedule has been returned.
    int64_t compile_model_mem_start = 0;
    if (_logger.level() >= ov::log::Level::INFO) {
        compile_model_mem_start = get_peak_memory_usage();
    }
    while (true) {
        _logger.debug("compileWS iteration %d", callNumber);
        updatedConfig.update({{ov::intel_npu::ws_compile_call_number.name(), std::to_string(callNumber++)}});

        _logger.debug("build flags");
        buildFlags = serializedIOInfo;
        buildFlags += " ";
        buildFlags += driver_compiler_utils::serializeConfig(updatedConfig,
                                                             compilerVersion,
                                                             _zeGraphExt->isTurboOptionSupported(compilerVersion));

        _logger.debug("compile start");
        // If UMD Caching is requested to be bypassed or if OV cache is enabled, disable driver caching
        const bool bypassCache = !config.get<CACHE_DIR>().empty() || config.get<BYPASS_UMD_CACHING>();
        auto graphDesc = _zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, bypassCache);
        _logger.debug("compile end");

        OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
        NetworkMetadata networkMetadata = _zeGraphExt->getNetworkMeta(graphDesc);

        if (isInitMetadata(networkMetadata)) {
            networkMetadata.name = model->get_friendly_name() + "_init";
            initNetworkMetadata.push_back(std::move(networkMetadata));
            initGraphDescriptors.push_back(graphDesc);
        } else {
            networkMetadata.name = model->get_friendly_name() + "_main";
            mainNetworkMetadata = std::move(networkMetadata);
            mainGraphHandle = graphDesc;
            serializedIR = SerializedIR();
            // By convention, the main schedule is the last result produced by the compiler
            break;
        }
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        auto compile_model_mem_end = get_peak_memory_usage();
        _logger.debug("Start of compilation memory usage: Peak %lld KB", compile_model_mem_start);
        _logger.debug("End of compilation memory usage: Peak %lld KB", compile_model_mem_end);
        // Note: Following log is parsed by CI. Take care when modifying it.
        _logger.info("Compilation memory usage: Peak %lld KB", compile_model_mem_end - compile_model_mem_start);
    }

    return std::make_shared<WeightlessGraph>(_zeGraphExt,
                                             _zeroInitStruct,
                                             mainGraphHandle,
                                             std::move(mainNetworkMetadata),
                                             /* mainBlob = */ std::nullopt,
                                             initGraphDescriptors,
                                             std::move(initNetworkMetadata),
                                             /* initBlobs = */ std::nullopt,
                                             model,
                                             config);
}

std::shared_ptr<IGraph> DriverCompilerAdapter::parse(
    const ov::Tensor& mainBlob,
    const FilteredConfig& config,
    const std::optional<std::vector<ov::Tensor>>& initBlobs,
    const std::optional<std::shared_ptr<const ov::Model>>& model) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "parse");

    _logger.debug("parse start");
    auto mainGraphDesc = _zeGraphExt->getGraphDescriptor(mainBlob.data(), mainBlob.get_byte_size());
    _logger.debug("parse end");

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeGraphExt->getNetworkMeta(mainGraphDesc);

    // exporting the blob when we get it from cache or ov::hint::compiled_blob property
    // shall be available
    const bool blobIsPersistent = config.has<COMPILED_BLOB>()       ? true
                                  : config.has<LOADED_FROM_CACHE>() ? config.get<LOADED_FROM_CACHE>()
                                                                    : false;

    if (!initBlobs.has_value()) {
        return std::make_shared<Graph>(_zeGraphExt,
                                       _zeroInitStruct,
                                       mainGraphDesc,
                                       std::move(networkMeta),
                                       mainBlob,
                                       config,
                                       blobIsPersistent);
    }

    // The presence of init schedules means weights separation has been enabled at compilation time. Use a specific
    // "Graph" object as wrapper over all L0 handles.
    std::vector<GraphDescriptor> initGraphDescriptors;
    std::vector<NetworkMetadata> initMetadata;

    for (const auto& initBlob : initBlobs.value()) {
        auto initGraphDesc = _zeGraphExt->getGraphDescriptor(initBlob.data(), initBlob.get_byte_size());

        initGraphDescriptors.push_back(initGraphDesc);
        initMetadata.push_back(_zeGraphExt->getNetworkMeta(initGraphDesc));
    }

    return std::make_shared<WeightlessGraph>(_zeGraphExt,
                                             _zeroInitStruct,
                                             mainGraphDesc,
                                             std::move(networkMeta),
                                             mainBlob,
                                             initGraphDescriptors,
                                             std::move(initMetadata),
                                             initBlobs,
                                             model.value(),
                                             config,
                                             blobIsPersistent);
}

ov::SupportedOpsMap DriverCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(query_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "query");

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    auto serializedIR =
        driver_compiler_utils::serializeIR(model, compilerVersion, maxOpsetVersion, useBaseModelSerializer(config));

    std::string buildFlags;
    buildFlags += driver_compiler_utils::serializeConfig(config, compilerVersion);
    _logger.debug("queryImpl build flags : %s", buildFlags.c_str());

    ov::SupportedOpsMap result;
    const std::string deviceName = "NPU";

    try {
        const auto supportedLayers = _zeGraphExt->queryGraph(std::move(serializedIR), buildFlags);
        for (auto&& layerName : supportedLayers) {
            result.emplace(layerName, deviceName);
        }
        _logger.info("For given model, there are %d supported layers", supportedLayers.size());
    } catch (std::exception& e) {
        OPENVINO_THROW("Fail in calling querynetwork : ", e.what());
    }

    _logger.debug("query end");
    return result;
}

uint32_t DriverCompilerAdapter::get_version() const {
    return _zeroInitStruct->getCompilerVersion();
}

std::vector<std::string> DriverCompilerAdapter::get_supported_options() const {
    std::string compilerOptionsStr;
    compilerOptionsStr = _zeGraphExt->getCompilerSupportedOptions();
    // vectorize string
    std::istringstream suppstream(compilerOptionsStr);
    std::vector<std::string> compilerOpts;
    std::string option;
    while (suppstream >> option) {
        compilerOpts.push_back(option);
    }
    return compilerOpts;
}

bool DriverCompilerAdapter::is_option_supported(std::string optName, std::optional<std::string> optValue) const {
    return _zeGraphExt->isOptionSupported(std::move(optName), std::move(optValue));
}

}  // namespace intel_npu
