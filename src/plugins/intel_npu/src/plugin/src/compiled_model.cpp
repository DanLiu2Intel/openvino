// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <fstream>
#include <stdexcept>
#include <string_view>

#include "async_infer_request.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "metadata.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"

#if defined(__linux__)
#    include "transformations/utils/utils.hpp"
#endif
namespace {

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

/**
 * @brief Print basic model information
 */
std::string print_basic_info(const std::shared_ptr<ov::Model>& model) {
    std::ostringstream ss;
    ss << "=== Model Basic Information ===" << std::endl;
    ss << "Name: " << model->get_name() << std::endl;
    ss << "Friendly Name: " << model->get_friendly_name() << std::endl;
    ss << "Output Size: " << model->get_output_size() << std::endl;
    ss << "Graph Size: " << model->get_graph_size() << " bytes" << std::endl;
    ss << "Is Dynamic: " << (model->is_dynamic() ? "Yes" : "No") << std::endl;
    ss << std::endl;
    return ss.str();
}

/**
 * @brief Print model parameters (inputs)
 */
std::string print_parameters(const std::shared_ptr<ov::Model>& model) {
    std::ostringstream ss;
    ss << "=== Model Parameters (Inputs) ===" << std::endl;
    const auto& parameters = model->get_parameters();
    ss << "Total Parameters: " << parameters.size() << std::endl;

    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& param = parameters[i];
        ss << "  [" << i << "] " << param->get_friendly_name() << "/(get_name is " << param->get_name()
           << ") : " << param->get_element_type() << " " << param->get_partial_shape() << std::endl;

        // Print additional parameter info
        ss << "      Type: " << param->get_type_name() << std::endl;
        if (param->get_output_size() > 0) {
            ss << "      Output tensor names (may contains multi names): ";
            for (size_t j = 0; j < param->get_output_size(); ++j) {
                auto names = param->get_output_tensor(j).get_names();
                for (const auto& name : names) {
                    ss << name << " ";
                }
            }
            ss << std::endl;
        }
    }
    ss << std::endl;
    return ss.str();
}

std::string print_results(const std::shared_ptr<ov::Model>& model) {
    std::ostringstream ss;
    ss << "=== Model Results (Outputs) ===" << std::endl;
    const auto& results = model->get_results();
    ss << "Total Results: " << results.size() << std::endl;

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        ss << "  [" << i << "] " << result->get_friendly_name() << "/(get_name is " << result->get_name()
           << ") : " << std::endl;
        ss << "      Type: " << result->get_type_name() << std::endl;

        if (result->get_input_size() > 0) {
            const auto& input = result->get_input_source_output(0);
            ss << "      Element Type: " << input.get_element_type() << std::endl;
            ss << "      Shape: " << input.get_partial_shape() << std::endl;

            auto names = result->get_output_tensor(0).get_names();
            if (!names.empty()) {
                ss << "      Tensor names (may contains multi names):: ";
                for (const auto& name : names) {
                    ss << name << " ";
                }
                ss << std::endl;
            }
        }
#if defined(__linux__)
        /// new add test line
        ss << "     Output tensor name (ov::op::util::get_ie_output_name(result->input_value(0))): "
           << ov::op::util::get_ie_output_name(result->input_value(0)) << std::endl;
#endif
    }
    ss << std::endl;

    return ss.str();
}

std::string print_all_info(const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        std::cout << "Model is null!" << std::endl;
        return "";
    }
    std::ostringstream ss;
    ss << print_basic_info(model);
    ss << print_parameters(model);
    ss << print_results(model);
    return ss.str();
}

std::string print_parameters2(const std::shared_ptr<ov::Model>& model) {
    std::ostringstream ss;
    ss << "=== Model Parameters (Inputs) ===" << std::endl;
    const auto& parameters = model->get_parameters();
    ss << "Total Parameters: " << parameters.size() << std::endl;

    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& param = parameters[i];
        // ss << "  [" << i << "] " << param->get_friendly_name() << "/(get_name is " << param->get_name()
        //    << ") : "
        ss << param->get_element_type() << " " << param->get_partial_shape() << std::endl;

        // Print additional parameter info
        ss << "      Type: " << param->get_type_name() << std::endl;
        if (param->get_output_size() > 0) {
            ss << "      Output tensor names (may contains multi names): ";
            for (size_t j = 0; j < param->get_output_size(); ++j) {
                auto names = param->get_output_tensor(j).get_names();
                for (const auto& name : names) {
                    ss << name << " ";
                }
            }
            ss << std::endl;
        }
    }
    ss << std::endl;
    return ss.str();
}

std::string print_results2(const std::shared_ptr<ov::Model>& model) {
    std::ostringstream ss;
    ss << "=== Model Results (Outputs) ===" << std::endl;
    const auto& results = model->get_results();
    ss << "Total Results: " << results.size() << std::endl;

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        ss << "  [" << i << "] "
           << result->get_friendly_name()
              // << "/(get_name is " << result->get_name()
              // << ") : " << std::endl;
           << "      Type: " << result->get_type_name() << std::endl;

        if (result->get_input_size() > 0) {
            const auto& input = result->get_input_source_output(0);
            ss << "      Element Type: " << input.get_element_type() << std::endl;
            ss << "      Shape: " << input.get_partial_shape() << std::endl;

            auto names = result->get_output_tensor(0).get_names();
            if (!names.empty()) {
                ss << "      Tensor names (may contains multi names):: ";
                for (const auto& name : names) {
                    ss << name << " ";
                }
                ss << std::endl;
            }
        }
#if defined(__linux__)
        /// new add test line
        ss << "     Output tensor name (ov::op::util::get_ie_output_name(result->input_value(0))): "
           << ov::op::util::get_ie_output_name(result->input_value(0)) << std::endl;
#endif
    }
    ss << std::endl;

    return ss.str();
}

std::string compare_info(const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        std::cout << "Model is null!" << std::endl;
        return "";
    }
    std::ostringstream ss;
    ss << "Output Size: " << model->get_output_size() << std::endl;
    ss << "Graph Size: " << model->get_graph_size() << " bytes" << std::endl;
    ss << "Is Dynamic: " << (model->is_dynamic() ? "Yes" : "No") << std::endl;
    ss << std::endl;
    ss << print_parameters2(model);
    ss << print_results2(model);
    return ss.str();
}

class NotEqualException : public std::runtime_error {
public:
    NotEqualException(const std::string& message) : std::runtime_error(message) {}
};

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

    for (const ov::Output<const ov::Node>& nodeOutput : inputs()) {
        std::shared_ptr<ov::Node> clonedParameter =
            std::dynamic_pointer_cast<const ov::op::v0::Parameter>(nodeOutput.get_node_shared_ptr())
                ->clone_with_new_inputs({});
        parameters.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(clonedParameter));
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
    }

    auto modelNoUseMetadata = std::make_shared<ov::Model>(results, parameters);

    /////need compare the model generated by metadata
    ov::ParameterVector parameterMetatdasMetatda;
    ov::ResultVector resultMetatdasMetatda;
    for (const IODescriptor& inputDescriptor : _graph->get_metadata().inputs) {
        if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::op::v0::Parameter> parameterMetatda =
            std::make_shared<ov::op::v0::Parameter>(inputDescriptor.precision, inputDescriptor.shapeFromCompiler);

        parameterMetatda->set_friendly_name(inputDescriptor.nodeFriendlyName);
        parameterMetatda->output(0).get_tensor().set_names(inputDescriptor.outputTensorNames);
        parameterMetatdasMetatda.push_back(std::move(parameterMetatda));
    }

    // The "resultMetatda" nodes require a parent node in order to satisfy the API conventions. Additionally, a dummy
    // shape for the "Constant" node was required since the specific constructor does not accept "ov::PartialShape"
    // values (a constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (const IODescriptor& outputDescriptor : _graph->get_metadata().outputs) {
        if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::Node> constantDummyMetatda = std::make_shared<ov::op::v0::Constant>(
            outputDescriptor.precision,
            outputDescriptor.shapeFromCompiler.to_shape().empty() ? CONSTANT_NODE_DUMMY_SHAPE
                                                                  : outputDescriptor.shapeFromCompiler.to_shape());

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummyMetatda =
            std::make_shared<ov::descriptor::Tensor>(outputDescriptor.precision,
                                                     outputDescriptor.shapeFromCompiler,
                                                     outputDescriptor.outputTensorNames);

        auto& resultMetatda =
            resultMetatdasMetatda.emplace_back(std::make_shared<ov::op::v0::Result>(constantDummyMetatda));
        resultMetatda->output(0).set_tensor_ptr(tensorDummyMetatda);
        resultMetatda->set_friendly_name(outputDescriptor.nodeFriendlyName);
    }

    _logger.warning(
        "   Returning a dummy ov::Model object that contains only the given parameterMetatda and resultMetatda nodes");

    auto modelUseMetadata = std::make_shared<ov::Model>(resultMetatdasMetatda, parameterMetatdasMetatda);
    try {
        std::cout << "------------modelNoUseMetadata----------------" << std::endl;
        std::cout << print_all_info(modelNoUseMetadata) << std::endl;
        std::cout << "-----------modelNoUseMetadata VS modelUseMetadata-----------------" << std::endl;
        std::cout << print_all_info(modelUseMetadata) << std::endl;
        std::cout << "------------modelUseMetadata----------------" << std::endl;
        if (compare_info(modelNoUseMetadata) != compare_info(modelUseMetadata)) {
            throw NotEqualException(
                "Runtime model using metadata and not using's result are NOT EQUAL in get_runtime_model()");
        } else {
            _logger.warning("Runtime model using metadata and not using's result are equal in get_runtime_model()");
        }
    } catch (const NotEqualException& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    return modelNoUseMetadata;
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
