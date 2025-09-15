// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <fstream>
#include <string_view>

#include "async_infer_request.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "metadata.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "transformations/utils/utils.hpp"

namespace {

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

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

/**
 * @brief Print basic model information
 */
void print_basic_info(const std::shared_ptr<ov::Model>& model) {
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
void print_parameters(const std::shared_ptr<ov::Model>& model) {
    std::cout << "=== Model Parameters (Inputs) ===" << std::endl;
    const auto& parameters = model->get_parameters();
    std::cout << "Total Parameters: " << parameters.size() << std::endl;

    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& param = parameters[i];
        std::cout << "  [" << i << "] " << param->get_friendly_name() << "/(get_name is " << param->get_name()
                  << ") : " << param->get_element_type() << " " << param->get_partial_shape() << std::endl;

        // Print additional parameter info
        std::cout << "      Type: " << param->get_type_name() << std::endl;
        if (param->get_output_size() > 0) {
            std::cout << "      Output tensor names (may contains multi names): ";
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
void print_results(const std::shared_ptr<ov::Model>& model) {
    std::cout << "=== Model Results (Outputs) ===" << std::endl;
    const auto& results = model->get_results();
    std::cout << "Total Results: " << results.size() << std::endl;

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        std::cout << "  [" << i << "] " << result->get_friendly_name() << "/(get_name is " << result->get_name()
                  << ") : " << std::endl;
        std::cout << "      Type: " << result->get_type_name() << std::endl;

        if (result->get_input_size() > 0) {
            const auto& input = result->get_input_source_output(0);
            std::cout << "      Element Type: " << input.get_element_type() << std::endl;
            std::cout << "      Shape: " << input.get_partial_shape() << std::endl;

            auto names = result->get_output_tensor(0).get_names();
            if (!names.empty()) {
                std::cout << "      Tensor names (may contains multi names):: ";
                for (const auto& name : names) {
                    std::cout << name << " ";
                }
                std::cout << std::endl;
            }
        }

        /// new add test line
        std::cout << "     Output tensor name (ov::op::util::get_ie_output_name(result->input_value(0))): "
                  << ov::op::util::get_ie_output_name(result->input_value(0)) << std::endl;
    }
    std::cout << std::endl;
}

void print_all_info(const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        std::cout << "Model is null!" << std::endl;
        return;
    }

    print_basic_info(model);
    print_parameters(model);
    print_results(model);
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

    configure_stream_executors();

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
    const char* pstr = std::getenv("get_runtime_model");
    const char* pstrdetail = std::getenv("get_runtime_Detail");

    if (pstr) {
        int ii1 = 0, io1 = 0;
        for (const IODescriptor& inputDescriptor : _graph->get_metadata().inputs) {
            if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor) {
                continue;
            }

            std::shared_ptr<ov::op::v0::Parameter> parameter =
                std::make_shared<ov::op::v0::Parameter>(inputDescriptor.precision, inputDescriptor.shapeFromCompiler);

            parameter->set_friendly_name(inputDescriptor.nodeFriendlyName);
            parameter->output(0).get_tensor().set_names(inputDescriptor.outputTensorNames);
            parameters.push_back(std::move(parameter));
            if (pstrdetail) {
                // print input info, shape, type, name
                std::cout << "input1 [" << ii1 << "]:" << std::endl;
                std::cout << "input1 from model's set_friendly_name" << parameter->get_friendly_name() << std::endl;
                std::cout << "input1 from model's precision" << parameter->get_element_type() << std::endl;
                bool first = true;
                for (const auto& item : parameter->get_shape()) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << "\"" << item << "\"";
                    first = false;
                }
                std::cout << "]" << std::endl;

                for (const auto& item : parameter->get_shape()) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << "\"" << item << "\"";
                    first = false;
                }
                std::cout << "]" << std::endl;
            }

            std::cout << "---------inputDescriptor---start----" << io1 << "-------" << std::endl;
            std::cout << ioDescriptorToString(inputDescriptor, 1) << std::endl;
            std::cout << "---------inputDescriptor---end------" << io1++ << "-------" << std::endl;
        }

        // The "result" nodes require a parent node in order to satisfy the API conventions. Additionally, a dummy shape
        // for the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values
        // (a constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
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
            if (pstrdetail) {
                // print input info, shape, type, name
                std::cout << "outnput1 [" << ii1++ << "]:" << std::endl;
                std::cout << "output1 from model's set_friendly_name" << result->get_friendly_name() << std::endl;
                std::cout << "output1 from model's precision" << result->get_element_type() << std::endl;
                bool first = true;
                for (const auto& item : result->get_shape()) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << "\"" << item << "\"";
                    first = false;
                }
                std::cout << "]" << std::endl;

                for (const auto& item : result->get_shape()) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << "\"" << item << "\"";
                    first = false;
                }
                std::cout << "]" << std::endl;
                std::cout << "---------outputDescriptor---start----" << io1 << "-------" << std::endl;
                std::cout << ioDescriptorToString(outputDescriptor, 1) << std::endl;
                std::cout << "---------outputDescriptor---end------" << io1++ << "-------" << std::endl;
            }
        }

        _logger.warning("Returning a dummy ov::Model object that contains only the given parameter and result nodes");

        return std::make_shared<ov::Model>(results, parameters);
    } else {
        int ii1 = 0, io1 = 0;
        ov::ParameterVector parameters;
        ov::ResultVector results;

        for (const ov::Output<const ov::Node>& nodeOutput : inputs()) {
            std::shared_ptr<ov::Node> clonedParameter =
                std::dynamic_pointer_cast<const ov::op::v0::Parameter>(nodeOutput.get_node_shared_ptr())
                    ->clone_with_new_inputs({});
            parameters.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(clonedParameter));
            if (pstrdetail) {
                // print input info, shape, type, name
                std::cout << "input1 [" << ii1 << "]:" << std::endl;
                std::cout << "  input1 from model's set_friendly_name: " << clonedParameter->get_friendly_name()
                          << std::endl;
                std::cout << "  input1 from model's precision: " << clonedParameter->get_element_type() << std::endl;
                bool first = true;
                std::cout << "  input1 from model's get_shape [";
                for (const auto& item : clonedParameter->get_shape()) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << "\"" << item << "\"";
                    first = false;
                }
                std::cout << "]" << std::endl;

                std::cout << "  input1 from model's nodeOutput.get_names() [";
                first = true;
                for (const auto& name : nodeOutput.get_names()) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << "\"" << name << "\"";
                    first = false;
                }
                std::cout << "]" << std::endl;

                std::cout << "input1 from model's print_parameters:" << std::endl;
                std::cout << "  [" << ii1++ << "] " << clonedParameter->get_friendly_name() << "/(get_name is "
                          << clonedParameter->get_name() << ") : " << clonedParameter->get_element_type() << " "
                          << std::endl;

                // Print additional parameter info
                std::cout << "input1 from model's additional info:\n Type: " << clonedParameter->get_type_name()
                          << std::endl;
                if (clonedParameter->get_output_size() > 0) {
                    std::cout << "      <input1 from model's additional info> Output tensor names (may contains multi "
                                 "names): ";
                    for (size_t j = 0; j < clonedParameter->get_output_size(); ++j) {
                        auto names = clonedParameter->get_output_tensor(j).get_names();
                        for (const auto& name : names) {
                            std::cout << name << " ";
                        }
                    }
                    std::cout << std::endl;
                }
                ii1++;
            }
        }

        for (const ov::Output<const ov::Node>& nodeOutput : outputs()) {
            const auto resultOriginal =
                std::dynamic_pointer_cast<const ov::op::v0::Result>(nodeOutput.get_node_shared_ptr());
            const ov::element::Type precision = nodeOutput.get_element_type();
            const ov::Shape shape =
                nodeOutput.get_partial_shape().is_dynamic() ? CONSTANT_NODE_DUMMY_SHAPE : nodeOutput.get_shape();

            std::shared_ptr<ov::Node> constantDummy = std::make_shared<ov::op::v0::Constant>(precision, shape);
            const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy =
                std::make_shared<ov::descriptor::Tensor>(precision, shape, nodeOutput.get_names());

            auto& resultCopy = results.emplace_back(std::make_shared<ov::op::v0::Result>(constantDummy));
            resultCopy->output(0).set_tensor_ptr(tensorDummy);
            resultCopy->set_friendly_name(resultOriginal->get_friendly_name());
            if (pstrdetail) {
                // print output info, shape, type, name
                std::cout << "result1 [" << io1 << "]:" << std::endl;
                std::cout << "result1 from model's set_friendly_name" << resultOriginal->get_friendly_name()
                          << std::endl;
                std::cout << "result1 from model's precision" << precision << std::endl;
                std::cout << "result1 from model's shape [";
                bool first = true;
                for (const auto& item : shape) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << "\"" << item << "\"";
                    first = false;
                }
                std::cout << "]" << std::endl;

                std::cout << "result1 from model's nodeOutput.get_names() [";
                first = true;
                for (const auto& name : nodeOutput.get_names()) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << "\"" << name << "\"";
                    first = false;
                }
                std::cout << "]" << std::endl;

                std::cout << "result1 from model's print_results:" << std::endl;
                const auto& res = results[io1];
                std::cout << "  [" << io1++ << "] " << res->get_friendly_name() << "/(get_name is " << res->get_name()
                          << ") : " << std::endl;
                std::cout << "      Type: " << res->get_type_name() << std::endl;

                if (res->get_input_size() > 0) {
                    const auto& input = res->get_input_source_output(0);
                    std::cout << "      Element Type: " << input.get_element_type() << std::endl;
                    std::cout << "      Shape: " << input.get_partial_shape() << std::endl;

                    auto names = res->get_output_tensor(0).get_names();
                    if (!names.empty()) {
                        std::cout << "      Tensor names (may contains multi names):: ";
                        for (const auto& name : names) {
                            std::cout << name << " ";
                        }
                        std::cout << std::endl;
                    }
                }

                /// new add test line
                std::cout << "     Output tensor name (ov::op::util::get_ie_output_name(res->input_value(0))): "
                          << ov::op::util::get_ie_output_name(res->input_value(0)) << std::endl;
            }
        }

        return std::make_shared<ov::Model>(results, parameters);
    }
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

void CompiledModel::configure_stream_executors() {
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

    set_task_executor(std::move(task_executor));
    const auto executorId = _graph->get_metadata().name + "_NPUResultExecutor";
    _resultExecutor = ov::threading::executor_manager()->get_executor(executorId);
}

}  // namespace intel_npu
