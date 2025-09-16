// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <fstream>
#include <string_view>

#include "async_infer_request.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/network_metadata.hpp"
#include "intel_npu/prefix.hpp"
#include "metadata.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "transformations/utils/utils.hpp"

namespace {

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};
class StringNotEqualException : public std::runtime_error {
public:
    StringNotEqualException(const std::string& message) : std::runtime_error(message) {}
};

void compareStrings(const std::string& str1, const std::string& str2) {
    if (str1 != str2) {
        throw StringNotEqualException("##metadata are not equal: \"" + str1 + "\" != \"" + str2 + "\"");
    }
}

std::map<std::string, std::size_t> stream{{"3720", 4}, {"4000", 4}, {"5010", 1}, {"5020", 2}, {"6000", 3}};

// for result: std::vector<std::shared_ptr<ov::op::v0::Result>>

std::vector<intel_npu::IODescriptor> convertIODescriptors(
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> parameters,
    intel_npu::FilteredConfig config) {
    std::cout << "------------PROCESS THE INPUT------------" << std::endl;
    std::vector<intel_npu::IODescriptor> convertedIODescriptors;

    for (int i = 0; i < parameters.size(); i++) {
        const auto& param = parameters[i];
        intel_npu::IODescriptor ioDesc;
        ioDesc.nameFromCompiler = param->get_friendly_name();
        ioDesc.precision = param->get_element_type();
        ioDesc.shapeFromCompiler = param->get_shape();
        if (intel_npu::isStateInputName(ioDesc.nameFromCompiler)) {
            ioDesc.nameFromCompiler = ioDesc.nameFromCompiler.substr(intel_npu::READVALUE_PREFIX.length());
            ioDesc.isStateInput = true;
        } else if (intel_npu::isShapeTensorName(ioDesc.nameFromCompiler)) {
            ioDesc.nameFromCompiler = ioDesc.nameFromCompiler.substr(intel_npu::SHAPE_TENSOR_PREFIX.length());
            ioDesc.isShapeTensor = true;
        } else if (intel_npu::isInitInputWeightsName(ioDesc.nameFromCompiler)) {
            ioDesc.nameFromCompiler = ioDesc.nameFromCompiler.substr(intel_npu::INIT_INPUT_WEIGHTS_PREFIX.length());
            ioDesc.isInitInputWeights = true;
        } else if (intel_npu::isMainInputWeightsName(ioDesc.nameFromCompiler)) {
            ioDesc.nameFromCompiler = ioDesc.nameFromCompiler.substr(intel_npu::MAIN_INPUT_WEIGHTS_PREFIX.length());
            ioDesc.isMainInputWeights = true;
        }

        if (param->get_output_size() > 0) {
            std::cout << "      Output tensor names : ";
            for (size_t j = 0; j < param->get_output_size(); ++j) {
                auto names = param->get_output_tensor(j).get_names();
                for (const auto& name : names) {
                    std::cout << name << " ";
                    ioDesc.outputTensorNames.insert(name);
                }
            }
            std::cout << std::endl;
        }

        ioDesc.nodeFriendlyName = param->get_friendly_name();
        
        //  /**
        //  * @brief The shape extracted from the IR model.
        //  * @details The values may differ from the ones found in "shapeFromCompiler" if batching is to be handled by the
        //  * plugin.
        //  *
        //  * This field may be empty if the I/O entry is not found in the original IR model (i.e. the entry was added
        //  * by the compiler).
        //  */
        // std::optional<ov::PartialShape> shapeFromIRModel = std::nullopt;
        if (config.isAvailable(ov::intel_npu::batch_mode.name()) &&
            !config.has(ov::intel_npu::batch_mode.name())) {
            ioDesc.shapeFromIRModel = param->get_shape();
        } else if (config.isAvailable(ov::intel_npu::batch_mode.name()) && !config->get_variables().empty()) {
            if (config.get<intel_npu::BATCH_MODE>() == ov::intel_npu::BatchMode::PLUGIN) {
                OPENVINO_THROW("This model contains states, thus it is not supported when handling batching on the plugin");
            }
            ioDesc.shapeFromIRModel = param->get_shape();
        }

        convertedIODescriptors.push_back(ioDesc);
    }
    std::cout << "------------PROCESS THE INPUT-----end-------" << std::endl;
    return convertedIODescriptors;
}

std::vector<intel_npu::IODescriptor> convertIODescriptors(std::vector<std::shared_ptr<ov::op::v0::Result>> results,
    intel_npu::FilteredConfig config) {
    std::cout << "------------PROCESS THE output------------" << std::endl;
    std::vector<intel_npu::IODescriptor> convertedIODescriptors;

    for (int i = 0; i < results.size(); i++) {
        const auto& result = results[i];
        intel_npu::IODescriptor ioDesc;
        ioDesc.nameFromCompiler = ov::op::util::get_ie_output_name(result->input_value(0));  // output
        ioDesc.precision = result->get_element_type();
        ioDesc.shapeFromCompiler = result->get_shape();
        if (intel_npu::isStateOutputName(ioDesc.nameFromCompiler)) {
            ioDesc.nameFromCompiler = ioDesc.nameFromCompiler.substr(intel_npu::ASSIGN_PREFIX.length());
            ioDesc.isStateOutput = true;
        } else if (intel_npu::isShapeTensorName(ioDesc.nameFromCompiler)) {
            ioDesc.nameFromCompiler = ioDesc.nameFromCompiler.substr(intel_npu::SHAPE_TENSOR_PREFIX.length());
            ioDesc.isShapeTensor = true;
        } else if (intel_npu::isInitOutputWeightsName(ioDesc.nameFromCompiler)) {
            ioDesc.nameFromCompiler = ioDesc.nameFromCompiler.substr(intel_npu::INIT_OUTPUT_WEIGHTS_PREFIX.length());
            ioDesc.isInitOutputWeights = true;
        }

        auto names = result->get_output_tensor(0).get_names();// return std::unordered_set<std::string>&
        if (!names.empty()) {
            std::cout << "      Tensor names (may contains multi names):: ";
            for (const auto& name : names) {
                std::cout << name << " ";
                ioDesc.outputTensorNames.insert(name);
            }
            std::cout << std::endl;
        }
        auto it = names.find(ioDesc.nameFromCompiler);
        if (it != names.end()) {
            std::cout << "Element " << ioDesc.nameFromCompiler << " already exists in the vector." << std::endl;
        } else {
            names.insert(ioDesc.nameFromCompiler);
            std::cout << "Element " << ioDesc.nameFromCompiler << " was not found in the vector and has been inserted." << std::endl;
        }
        ioDesc.nodeFriendlyName = result->get_friendly_name();
    //  /**
    //  * @brief The shape extracted from the IR model.
    //  * @details The values may differ from the ones found in "shapeFromCompiler" if batching is to be handled by the
    //  * plugin.
    //  *
    //  * This field may be empty if the I/O entry is not found in the original IR model (i.e. the entry was added
    //  * by the compiler).
    //  */
    // std::optional<ov::PartialShape> shapeFromIRModel = std::nullopt;
        //need process batch mode
        if (config.isAvailable(ov::intel_npu::batch_mode.name()) &&
            !config.has(ov::intel_npu::batch_mode.name())) {
            ioDesc.shapeFromIRModel = param->get_shape();
        } else if (config.isAvailable(ov::intel_npu::batch_mode.name()) && !config->get_variables().empty()) {
            if (config.get<intel_npu::BATCH_MODE>() == ov::intel_npu::BatchMode::PLUGIN) {
                OPENVINO_THROW("This model contains states, thus it is not supported when handling batching on the plugin");
            }
            ioDesc.shapeFromIRModel = param->get_shape();
        }
        convertedIODescriptors.push_back(ioDesc);
    }
     std::cout << "------------PROCESS THE output------END------" << std::endl;
    return convertedIODescriptors;
}

void getNetworkMetadata(const std::shared_ptr<const ov::Model>& model,
                        intel_npu::NetworkMetadata& networkMetadata,  // return type
                        intel_npu::FilteredConfig config) {
    if (!model) {
        std::cout << "passed model is empty" << std::endl;
    }

    networkMetadata.name = model->get_friendly_name(); //add_abc is not match

    const auto& parameters = model->get_parameters();
    const auto& results = model->get_results();

    networkMetadata.inputs = convertIODescriptors(parameters, config);
    networkMetadata.outputs = convertIODescriptors(results, config, false);
    // profilingOutputs how to get?
    if (config.has<intel_npu::PERF_COUNT>()) {
        networkMetadata.profilingOutputs = convertIODescriptors(results);
        std::cout << "profiling info is not true, need check with IMD" << std::endl;
    } else {
        networkMetadata.profilingOutputs = {};
        std::cout << "profiling info is empty, need check with IMD, may get log warning "
                     "\"inferRequest::get_profiling_info complete with empty\""
                  << std::endl;
    }

    if (!networkMetadata.profilingOutputs.empty()) {
        std::cout << "Metadata structure contains info on profiling outputs" << std::endl;
    } else {
        std::cout << "Metadata structure does not contain info on profiling outputs" << std::endl;
    }

    if (!networkMetadata.outputs.empty()) {
        std::cout << "Metadata structure does not contain info on outputs" << std::endl;
    }
    std::cout << "platform: " << config.get<intel_npu::PLATFORM>()
              << "    deviceID: " << config.get<intel_npu::DEVICE_ID>()
              << "    PERF_COUNT: " << config.get<intel_npu::PERF_COUNT>() << std::endl;
    networkMetadata.numStreams = stream[config.get<intel_npu::PLATFORM>()];

    networkMetadata.bindRelatedDescriptors();
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

}  // namespace

namespace intel_npu {

using intel_npu::envVarStrToBool;

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<IDevice>& device,
                             const std::shared_ptr<IGraph>& graph,
                             const FilteredConfig& config)
    : ICompiledModel(model, plugin),
      _config(config),
      _logger("CompiledModel", config.get<LOG_LEVEL>()),
      _device(device),
      _graph(graph) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::CompiledModel");

    OV_ITT_TASK_CHAIN(COMPILED_MODEL, itt::domains::NPUPlugin, "CompiledModel::CompiledModel", "initialize_properties");
    _properties = std::make_unique<Properties>(PropertiesType::COMPILED_MODEL, _config);
    _properties->registerProperties();

    configure_stream_executors(model);

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

CompiledModel::~CompiledModel() {
    _logger.debug("~CompiledModel()");
    std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(get_task_executor())->cpu_reset();
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::create_infer_request");

    // sanity check
    if (_device == nullptr) {
        OPENVINO_THROW("No available devices. Failed to create infer request!");
    }

    if (!_config.get<CREATE_EXECUTOR>() || _config.get<DEFER_WEIGHTS_LOAD>()) {
        if (_graph == nullptr) {
            OPENVINO_THROW("Invalid graph handle! Failed to create infer request!");
        }
        _graph->initialize(_config);
    }

    const std::shared_ptr<SyncInferRequest>& syncInferRequest =
        _device->createInferRequest(shared_from_this(), _config);
    syncInferRequest->initialize_states();

    return std::make_shared<AsyncInferRequest>(syncInferRequest,
                                               get_task_executor(),
                                               _resultExecutor,
                                               get_callback_executor());
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OPENVINO_THROW_NOT_IMPLEMENTED(
        "The synchronous inference request structure implemented by the NPU plugin does not inherit "
        "the \"ov::ISyncInferRequest\" class");
}

void CompiledModel::export_model(std::ostream& stream) const {
    _logger.debug("CompiledModel::export_model");

    auto [blobSizesBeforeVersioning, initBlobSizes] = _graph->export_blob(stream);

    Metadata<CURRENT_METADATA_VERSION>(blobSizesBeforeVersioning, CURRENT_OPENVINO_VERSION, initBlobSizes)
        .write(stream);
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    ov::ParameterVector parameters;
    ov::ResultVector results;

    for (const IODescriptor& inputDescriptor : _graph->get_metadata().inputs) {
        if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::op::v0::Parameter> parameter =
            std::make_shared<ov::op::v0::Parameter>(inputDescriptor.precision, inputDescriptor.shapeFromCompiler);

        parameter->set_friendly_name(inputDescriptor.nodeFriendlyName);
        parameter->output(0).get_tensor().set_names(inputDescriptor.outputTensorNames);
        parameters.push_back(std::move(parameter));
    }

    // The "result" nodes require a parent node in order to satisfy the API conventions. Additionally, a dummy shape for
    // the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values (a
    // constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (const IODescriptor& outputDescriptor : _graph->get_metadata().outputs) {
        if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::Node> constantDummy = std::make_shared<ov::op::v0::Constant>(
            outputDescriptor.precision,
            outputDescriptor.shapeFromCompiler.to_shape().empty() ? CONSTANT_NODE_DUMMY_SHAPE
                                                                  : outputDescriptor.shapeFromCompiler.to_shape());

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy =
            std::make_shared<ov::descriptor::Tensor>(outputDescriptor.precision,
                                                     outputDescriptor.shapeFromCompiler,
                                                     outputDescriptor.outputTensorNames);

        auto& result = results.emplace_back(std::make_shared<ov::op::v0::Result>(constantDummy));
        result->output(0).set_tensor_ptr(tensorDummy);
        result->set_friendly_name(outputDescriptor.nodeFriendlyName);
    }

    _logger.warning("Returning a dummy ov::Model object that contains only the given parameter and result nodes");

    return std::make_shared<ov::Model>(results, parameters);
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    // 1. Set the property via Properties interface
    _properties->set_property(properties);

    // 2. Extra hooks
    if (properties.count(std::string(WORKLOAD_TYPE::key())) != 0) {
        if (_graph != nullptr) {
            const auto workloadType = properties.at(ov::workload_type.name()).as<ov::WorkloadType>();
            _graph->set_workload_type(workloadType);
        }
    }
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    // special cases
    if (name == ov::model_name.name()) {
        OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
        return _graph->get_metadata().name;
    } else {
        // default behaviour
        return _properties->get_property(name);
    }
}

const std::shared_ptr<IGraph>& CompiledModel::get_graph() const {
    return _graph;
}

const FilteredConfig& CompiledModel::get_config() const {
    return _config;
}

void CompiledModel::configure_stream_executors(const std::shared_ptr<const ov::Model>& model) {
    std::shared_ptr<ov::threading::ITaskExecutor> task_executor;
    if (get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>()) {
        task_executor = ov::threading::executor_manager()->get_executor("NPU");
    } else if (get_property(ov::hint::enable_cpu_pinning.name()).as<bool>()) {
        auto executor_config = ov::threading::IStreamsExecutor::Config{
            /* name = */ "Intel NPU plugin executor",
            /* streams = */ get_plugin()->get_property(ov::num_streams.name(), {}).as<ov::streams::Num>(),
            /* threads_per_stream = */ 1,
            /* thread_preferred_core_type = */ ov::hint::SchedulingCoreType::PCORE_ONLY,
            /* cpu_reservation = */ true};
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(executor_config);
    } else {
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"NPUPlugin executor"});
    }
    auto metadata = _graph->get_metadata();
    std::cout << "[CompiledModel]------10.after compile, init---------" << std::endl;
    std::cout << networkMetadataToString(metadata) << std::endl;
    std::cout << "[CompiledModel]------11.after compile, init----------" << std::endl;

    intel_npu::NetworkMetadata metadataFake;
    getNetworkMetadata(model, metadataFake, _config);

    std::cout << "[CompiledModel]------12.fake metadata, init---------" << std::endl;
    std::cout << networkMetadataToString(metadataFake) << std::endl;
    std::cout << "[CompiledModel]------13.fake metadata, init after---------" << std::endl;

    try {
        std::string string1 = networkMetadataToString(metadata);
        std::string string2 = networkMetadataToString(metadataFake);

        compareStrings(string1, string2);
        std::cout << "metadata and metadataFake are equal." << std::endl;
    } catch (const StringNotEqualException& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    set_task_executor(std::move(task_executor));
    const auto executorId = _graph->get_metadata().name + "_NPUResultExecutor";
    _resultExecutor = ov::threading::executor_manager()->get_executor(executorId);
}

}  // namespace intel_npu
