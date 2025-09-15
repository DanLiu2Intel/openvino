// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_infer_request.hpp"

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "zero_memory.hpp"
#include "zero_variable_state.hpp"

using namespace intel_npu;

namespace {

constexpr std::size_t SINGLE_TENSOR = 0;
constexpr bool INPUT = true;
constexpr bool OUTPUT = false;

/**
 * @brief Checks that the metadata of the provided descriptor corresponds to the values registered in the Level Zero
 * structure.
 * @param ioDescriptor The OpenVINO API specific I/O descriptor which shall be compared.
 * @param zeDescriptor The Level Zero specific structure used for comparison.
 */
void check_level_zero_attributes_match(const IODescriptor& ioDescriptor, const ArgumentDescriptor& zeDescriptor) {
    std::string zeDescriptorName = zeDescriptor.info.name;

    if (isStateInputName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(READVALUE_PREFIX.length());
    } else if (isStateOutputName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(ASSIGN_PREFIX.length());
    } else if (isShapeTensorName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(SHAPE_TENSOR_PREFIX.length());
    } else if (isInitInputWeightsName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(INIT_INPUT_WEIGHTS_PREFIX.length());
    } else if (isInitOutputWeightsName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(INIT_OUTPUT_WEIGHTS_PREFIX.length());
    } else if (isMainInputWeightsName(zeDescriptorName)) {
        zeDescriptorName = zeDescriptorName.substr(MAIN_INPUT_WEIGHTS_PREFIX.length());
    }

    OPENVINO_ASSERT(ioDescriptor.nameFromCompiler == zeDescriptorName,
                    "Name mismatch between the I/O structure used internally and its Level Zero correspondent: ",
                    ioDescriptor.nameFromCompiler,
                    " vs. ",
                    zeDescriptorName,
                    ". The I/O order may have been altered, which could lead to an erroneous behavior.");
    OPENVINO_ASSERT(ioDescriptor.precision == zeroUtils::toOVElementType(zeDescriptor.info.devicePrecision),
                    "Precision mismatch for input/output named " + ioDescriptor.nameFromCompiler);

    const std::vector<size_t>& ovDimensions = ioDescriptor.shapeFromCompiler.get_max_shape();
    OPENVINO_ASSERT(ovDimensions.size() <= ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE,
                    "Maximum number of dimensions supported: " + std::to_string(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) +
                        '\n' + "Given: " + std::to_string(ovDimensions.size()));

    for (size_t index = 0; index < ovDimensions.size(); ++index) {
        OPENVINO_ASSERT(ovDimensions[index] == zeDescriptor.info.dims[index],
                        "Shape mismatch for input/output named " + ioDescriptor.nameFromCompiler);
    }
    for (size_t index = ovDimensions.size(); index < ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE; ++index) {
        OPENVINO_ASSERT(zeDescriptor.info.dims[index] == 0 || zeDescriptor.info.dims[index] == 1,
                        "Shape mismatch for input/output named " + ioDescriptor.nameFromCompiler);
    }
}

// /**
//  * @brief Print basic model information
//  */
// void print_basic_info(const std::shared_ptr<ov::Model>& model) {
//     std::cout << "=== Model Basic Information ===" << std::endl;
//     std::cout << "Name: " << model->get_name() << std::endl;
//     std::cout << "Friendly Name: " << model->get_friendly_name() << std::endl;
//     std::cout << "Output Size: " << model->get_output_size() << std::endl;
//     std::cout << "Graph Size: " << model->get_graph_size() << " bytes" << std::endl;
//     std::cout << "Is Dynamic: " << (model->is_dynamic() ? "Yes" : "No") << std::endl;
//     std::cout << std::endl;
// }

// /**
//  * @brief Print model parameters (inputs)
//  */
// void print_parameters(const std::shared_ptr<ov::Model>& model) {
//     std::cout << "=== Model Parameters (Inputs) ===" << std::endl;
//     const auto& parameters = model->get_parameters();
//     std::cout << "Total Parameters: " << parameters.size() << std::endl;

//     for (size_t i = 0; i < parameters.size(); ++i) {
//         const auto& param = parameters[i];
//         std::cout << "  [" << i << "] " << param->get_friendly_name() << "/(get_name is " << param->get_name()
//                   << ") : " << param->get_element_type() << " " << param->get_partial_shape() << std::endl;

//         // Print additional parameter info
//         std::cout << "      Type: " << param->get_type_name() << std::endl;
//         if (param->get_output_size() > 0) {
//             std::cout << "      Output tensor names (may contains multi names): ";
//             for (size_t j = 0; j < param->get_output_size(); ++j) {
//                 auto names = param->get_output_tensor(j).get_names();
//                 for (const auto& name : names) {
//                     std::cout << name << " ";
//                 }
//             }
//             std::cout << std::endl;
//         }
//     }
//     std::cout << std::endl;
// }

// /**
//  * @brief Print model results (outputs)
//  */
// void print_results(const std::shared_ptr<ov::Model>& model) {
//     std::cout << "=== Model Results (Outputs) ===" << std::endl;
//     const auto& results = model->get_results();
//     std::cout << "Total Results: " << results.size() << std::endl;

//     for (size_t i = 0; i < results.size(); ++i) {
//         const auto& result = results[i];
//         std::cout << "  [" << i << "] " << result->get_friendly_name() << "/(get_name is " << result->get_name()
//                   << ") : " << std::endl;
//         std::cout << "      Type: " << result->get_type_name() << std::endl;

//         if (result->get_input_size() > 0) {
//             const auto& input = result->get_input_source_output(0);
//             std::cout << "      Element Type: " << input.get_element_type() << std::endl;
//             std::cout << "      Shape: " << input.get_partial_shape() << std::endl;

//             auto names = result->get_output_tensor(0).get_names();
//             if (!names.empty()) {
//                 std::cout << "      Tensor names (may contains multi names):: ";
//                 for (const auto& name : names) {
//                     std::cout << name << " ";
//                 }
//                 std::cout << std::endl;
//             }
//         }

//         /// new add test line
//         std::cout << "     Output tensor name (ov::op::util::get_ie_output_name(result->input_value(0))): "
//                   << ov::op::util::get_ie_output_name(result->input_value(0)) << std::endl;
//     }
//     std::cout << std::endl;
// }

// /**
//  * @brief Print model variables
//  */
// void print_variables(const std::shared_ptr<ov::Model>& model) {
//     std::cout << "=== Model Variables ===" << std::endl;
//     const auto& variables = model->get_variables();
//     std::cout << "Total Variables: " << variables.size() << std::endl;

//     for (size_t i = 0; i < variables.size(); ++i) {
//         const auto& var = variables[i];
//         const auto& info = var->get_info();
//         std::cout << "  [" << i << "] ID: " << info.variable_id << std::endl;
//         std::cout << "      Shape: " << info.data_shape << std::endl;
//         std::cout << "      Type: " << info.data_type << std::endl;
//     }
//     std::cout << std::endl;
// }

// /**
//  * @brief Print model sinks
//  */
// void print_sinks(const std::shared_ptr<ov::Model>& model) {
//     std::cout << "=== Model Sinks ===" << std::endl;
//     const auto& sinks = model->get_sinks();
//     std::cout << "Total Sinks: " << sinks.size() << std::endl;

//     for (size_t i = 0; i < sinks.size(); ++i) {
//         const auto& sink = sinks[i];
//         std::cout << "  [" << i << "] " << sink->get_friendly_name() << " (" << sink->get_type_name() << ")"
//                   << std::endl;
//     }
//     std::cout << std::endl;
// }

// /**
//  * @brief Print runtime information
//  */
// void print_runtime_info(const std::shared_ptr<ov::Model>& model) {
//     std::cout << "=== Model Runtime Information ===" << std::endl;
//     const auto& rt_info = model->get_rt_info();
//     std::cout << "Runtime Info Entries: " << rt_info.size() << std::endl;

//     for (const auto& kv : rt_info) {
//         std::cout << "  " << kv.first << " = ";
//         try {
//             // Try to convert to string
//             std::cout << kv.second.as<std::string>();
//         } catch (...) {
//             try {
//                 // Try to convert to int
//                 std::cout << kv.second.as<int>();
//             } catch (...) {
//                 try {
//                     // Try to convert to bool
//                     std::cout << (kv.second.as<bool>() ? "true" : "false");
//                 } catch (...) {
//                     std::cout << "[complex type]";
//                 }
//             }
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;
// }

// /**
//  * @brief Print all nodes in the model
//  */
// void print_all_nodes(const std::shared_ptr<ov::Model>& model) {
//     std::cout << "=== All Nodes (Detailed) ===" << std::endl;
//     const auto& nodes = model->get_ordered_ops();
//     std::cout << "Total Nodes: " << nodes.size() << std::endl;

//     for (size_t i = 0; i < nodes.size(); ++i) {
//         const auto& node = nodes[i];
//         std::cout << "  [" << std::setw(3) << i << "] " << std::setw(20) << std::left << node->get_friendly_name()
//                   << " (" << node->get_type_name() << ")" << std::endl;

//         // Print inputs
//         if (node->get_input_size() > 0) {
//             std::cout << "       Inputs: ";
//             for (size_t j = 0; j < node->get_input_size(); ++j) {
//                 const auto& input = node->get_input_source_output(j);
//                 std::cout << input.get_element_type() << input.get_partial_shape();
//                 if (j < node->get_input_size() - 1)
//                     std::cout << ", ";
//             }
//             std::cout << std::endl;
//         }

//         // Print outputs
//         if (node->get_output_size() > 0) {
//             std::cout << "       Outputs: ";
//             for (size_t j = 0; j < node->get_output_size(); ++j) {
//                 const auto& output = node->get_output_tensor(j);
//                 std::cout << output.get_element_type() << output.get_partial_shape();
//                 if (j < node->get_output_size() - 1)
//                     std::cout << ", ";
//             }
//             std::cout << std::endl;
//         }

//         // Print node runtime info if exists
//         const auto& node_rt_info = node->get_rt_info();
//         if (!node_rt_info.empty()) {
//             std::cout << "       RT Info: ";
//             for (const auto& kv : node_rt_info) {
//                 std::cout << kv.first << " ";
//             }
//             std::cout << std::endl;
//         }
//     }
//     std::cout << std::endl;
// }

// /**
//  * @brief Print graph statistics
//  */
// void print_graph_statistics(const std::shared_ptr<ov::Model>& model) {
//     std::cout << "=== Graph Statistics ===" << std::endl;
//     const auto& nodes = model->get_ops();

//     // Count nodes by type
//     std::map<std::string, int> node_type_count;
//     for (const auto& node : nodes) {
//         node_type_count[node->get_type_name()]++;
//     }

//     std::cout << "Node Type Distribution:" << std::endl;
//     for (const auto& kv : node_type_count) {
//         std::cout << "  " << std::setw(20) << std::left << kv.first << ": " << kv.second << std::endl;
//     }
//     std::cout << std::endl;
// }

// void print_all_info(const std::shared_ptr<ov::Model>& model) {
//     if (!model) {
//         std::cout << "Model is null!" << std::endl;
//         return;
//     }

//     print_basic_info(model);
//     print_parameters(model);
//     print_results(model);
//     print_variables(model);
//     print_sinks(model);
//     print_runtime_info(model);

//     const char* detail = std::getenv("DETAIL");
//     if (detail) {
//         print_all_nodes(model);
//     }

//     print_graph_statistics(model);
// }

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

inline void printArgumentDescriptor(const intel_npu::ArgumentDescriptor& arg) {
    std::cout << "ArgumentDescriptor:" << std::endl;
    std::cout << "  idx: " << arg.idx << std::endl;
    std::cout << "  info:" << std::endl;
    std::cout << "    stype: " << arg.info.stype << std::endl;
    std::cout << "    pNext: " << arg.info.pNext << std::endl;
    std::cout << "    name: " << arg.info.name << std::endl;
    std::cout << "    type: " << arg.info.type << std::endl;
    std::cout << "    dims: ";
    for (size_t i = 0; i < sizeof(arg.info.dims)/sizeof(arg.info.dims[0]); ++i) {
        std::cout << arg.info.dims[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "    networkPrecision: " << arg.info.networkPrecision << std::endl;
    std::cout << "    networkLayout: " << arg.info.networkLayout << std::endl;
    std::cout << "    devicePrecision: " << arg.info.devicePrecision << std::endl;
    std::cout << "    deviceLayout: " << arg.info.deviceLayout << std::endl;
    std::cout << "    quantReverseScale: " << arg.info.quantReverseScale << std::endl;
    std::cout << "    quantZeroPoint: " << static_cast<int>(arg.info.quantZeroPoint) << std::endl;
    std::cout << "    dims_count: " << arg.info.dims_count << std::endl;
    std::cout << "    debug_friendly_name: " << arg.info.debug_friendly_name << std::endl;
    std::cout << "    associated_tensor_names_count: " << arg.info.associated_tensor_names_count << std::endl;
    std::cout << "    associated_tensor_names: ";
    for (uint32_t i = 0; i < arg.info.associated_tensor_names_count; ++i) {
        std::cout << arg.info.associated_tensor_names[i] << " ";
    }
    std::cout << std::endl;
}

}  // namespace

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                                   const std::shared_ptr<const ICompiledModel>& compiledModel,
                                   const Config& config)
    : SyncInferRequest(compiledModel, config),
      _initStructs(initStructs),
      _graph(compiledModel->get_graph()),
      _config(config),
      _logger("ZeroInferRequest", config.get<LOG_LEVEL>()),
      _graphInputDescriptors(_graph->get_input_descriptors()),
      _graphOutputDescriptors(_graph->get_output_descriptors()),
      _levelZeroInputTensors(_metadata.inputs.size(), std::vector<std::shared_ptr<ov::ITensor>>(1, nullptr)),
      _levelZeroOutputTensors(_metadata.outputs.size(), nullptr) {
    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest");

    std::cout << "-------------------------ZeroInferRequest init1----------------------------------------" << std::endl;
    std::cout << networkMetadataToString(_metadata) << std::endl;
    std::cout << "-------------------------ZeroInferRequest init2----------------------------------------" << std::endl;

    _outputAllocator = std::make_shared<const zeroMemory::HostMemAllocator>(_initStructs);
    _inputAllocator =
        std::make_shared<const zeroMemory::HostMemAllocator>(_initStructs, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);

    _logger.debug("ZeroInferRequest::ZeroInferRequest - checking level zero attributes and allocating tensors");

    size_t ioIndex = 0;
    for (const IODescriptor& inputDescriptor : _metadata.inputs) {
        check_level_zero_attributes_match(inputDescriptor, _graphInputDescriptors.at(ioIndex));

        if (!(inputDescriptor.isStateInput || inputDescriptor.isShapeTensor)) {
            ++ioIndex;
            continue;
        }

        get_level_zero_input(ioIndex) =
            allocate_tensor(inputDescriptor, ioIndex, INPUT, *_inputAllocator, _graph->get_batch_size());

        ++ioIndex;
    }

    ioIndex = 0;
    for (const IODescriptor& outputDescriptor : _metadata.outputs) {
        check_level_zero_attributes_match(outputDescriptor, _graphOutputDescriptors.at(ioIndex));

        if (!(outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor)) {
            ++ioIndex;
            continue;
        }

        _levelZeroOutputTensors.at(ioIndex) =
            allocate_tensor(outputDescriptor, ioIndex, OUTPUT, *_outputAllocator, _graph->get_batch_size());

        ++ioIndex;
    }

    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest completed");
}

void ZeroInferRequest::create_pipeline() {
    for (size_t inputIndex = 0; inputIndex < _metadata.inputs.size(); ++inputIndex) {
        if (is_batched_input(inputIndex) && _graph->get_batch_size().has_value()) {
            if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
                _logger.debug("ZeroInferRequest::create_pipeline - tensors %s were already allocated",
                              _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str());
            } else {
                for (size_t i = 0; i < get_user_inputs(inputIndex).size(); i++) {
                    get_level_zero_inputs(inputIndex).resize(get_user_inputs(inputIndex).size());

                    _logger.debug("ZeroInferRequest::create_pipeline - allocate new input tensor for batched input: %s",
                                  _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str());

                    get_level_zero_input(inputIndex, i) =
                        allocate_tensor(_metadata.inputs.at(inputIndex), inputIndex, true, *_inputAllocator);
                }
            }
            continue;
        }

        if (get_level_zero_input(inputIndex)) {
            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated",
                          _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str());
            continue;
        }

        _logger.debug("ZeroInferRequest::create_pipeline - allocate new input tensor %s",
                      _metadata.inputs.at(inputIndex).nodeFriendlyName.c_str());
        get_level_zero_input(inputIndex) = allocate_tensor(_metadata.inputs.at(inputIndex),
                                                           inputIndex,
                                                           INPUT,
                                                           *_inputAllocator,
                                                           _graph->get_batch_size());
    }

    for (size_t outputIndex = 0; outputIndex < _metadata.outputs.size(); ++outputIndex) {
        if (_levelZeroOutputTensors.at(outputIndex)) {
            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated",
                          _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str());
            continue;
        }
        _logger.debug("ZeroInferRequest::create_pipeline - allocate new output tensor %s",
                      _metadata.outputs.at(outputIndex).nodeFriendlyName.c_str());
        _levelZeroOutputTensors.at(outputIndex) = allocate_tensor(_metadata.outputs.at(outputIndex),
                                                                  outputIndex,
                                                                  OUTPUT,
                                                                  *_outputAllocator,
                                                                  _graph->get_batch_size());
    }
    _logger.debug("ZeroInferRequest::create_pipeline - init completed");

    // Set new tensors and reset variable state flag if memory updated before creating the pipeline
    _logger.debug("ZeroInferRequest::create_pipeline - set new tensors and reset variable state flag if memory updated "
                  "before creating the pipeline");
    for (const auto& variableState : _variableStates) {
        auto zeroState = std::dynamic_pointer_cast<ZeroVariableState>(variableState._ptr);

        OPENVINO_ASSERT(zeroState != nullptr, "State is not compatible with NPU plugin");

        if (zeroState->tensor_was_updated()) {
            get_user_input(zeroState->get_tensor_index()) = zeroState->get_state();
            _userOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_state();

            zeroState->reset_tensor_updated_flag();

            if (zeroState->zero_tensor_should_be_updated()) {
                zeroState->reset_zero_tensor_updated_flag();

                get_level_zero_input(zeroState->get_tensor_index()) = zeroState->get_state()._ptr;
                _levelZeroOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_state()._ptr;
            }
        }
    }

    _logger.debug("ZeroInferRequest::create_pipeline - constructing pipeline");

    // Construct pipeline
    _pipeline =
        std::make_unique<Pipeline>(_config, _initStructs, _graph, _levelZeroInputTensors, _levelZeroOutputTensors);

    _logger.debug("ZeroInferRequest::create_pipeline - SyncInferRequest completed");
}

void ZeroInferRequest::set_tensor_data(const std::shared_ptr<ov::ITensor>& tensor,
                                       const size_t index,
                                       const bool isInput) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "set_tensor_data");
    auto& levelZeroTensors = isInput ? get_level_zero_input(index) : _levelZeroOutputTensors.at(index);

    bool updateCommandListArg = false;

    OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "check_data_allocation");
    if (zeroUtils::memory_was_allocated_in_the_same_l0_context(_initStructs->getContext(), tensor->data())) {
        _logger.debug("ZeroInferRequest::set_tensor_data - tensor was created in the same L0 context");
        levelZeroTensors = tensor;
        updateCommandListArg = true;
    } else {
        auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(levelZeroTensors);
        if (zeroTensor == nullptr || (zeroTensor != nullptr && zeroTensor->tensor_was_shared_with_user())) {
            _logger.debug("ZeroInferRequest::set_tensor_data - create locally L0 tensor");
            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "allocate tensor");

            levelZeroTensors = allocate_tensor(isInput ? _metadata.inputs.at(index) : _metadata.outputs.at(index),
                                               index,
                                               isInput,
                                               isInput ? *_inputAllocator : *_outputAllocator,
                                               _graph->get_batch_size());

            updateCommandListArg = true;
        }
    }

    if (_pipelineIsCreated && updateCommandListArg) {
        _logger.debug("ZeroInferRequest::infer_async - update command list");

        OPENVINO_ASSERT(levelZeroTensors->data(), "Empty buffer");

        OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "update_graph_arguments");
        _pipeline->update_graph_arguments(
            isInput ? _graph->get_input_descriptors().at(index).idx : _graph->get_output_descriptors().at(index).idx,
            levelZeroTensors->data(),
            levelZeroTensors->get_byte_size());
    }
}

void ZeroInferRequest::set_remote_tensor_data(const std::shared_ptr<ZeroRemoteTensor>& tensor,
                                              const size_t index,
                                              const bool isInput) {
    OV_ITT_TASK_CHAIN(ZERO_SET_REMOTE_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "set_remote_tensor_data");

    auto l0_context = tensor->get_zero_context_handle();
    if (_initStructs->getContext() != l0_context) {
        OPENVINO_THROW("Using different context for creating the tensor is not supported");
    }

    auto& levelZeroTensors = isInput ? get_level_zero_input(index) : _levelZeroOutputTensors.at(index);
    levelZeroTensors = tensor;

    if (_pipelineIsCreated) {
        _logger.debug("ZeroInferRequest::infer_async - update command list");

        auto data = tensor->get_original_memory();
        OPENVINO_ASSERT(data, "Empty buffer");

        OV_ITT_TASK_NEXT(ZERO_SET_REMOTE_TENSOR, "update_graph_arguments");
        _pipeline->update_graph_arguments(
            isInput ? _graph->get_input_descriptors().at(index).idx : _graph->get_output_descriptors().at(index).idx,
            data,
            tensor->get_byte_size());
    }
}

void ZeroInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "set_tensor");

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find tensor for port ", port);
    try {
        check_tensor(port, tensor);
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to set tensor. ", ex.what());
    }

    if (foundPort.is_input()) {
        if (get_user_input(foundPort.idx)._ptr == tensor._ptr) {
            // Got set_tensor with the same object - do nothing
            _logger.debug("ZeroInferRequest::set_tensor - got the same tensor, do nothing");
            return;
        }
        if (is_batched_input(foundPort.idx)) {
            // resize vector size to 1 if set_tensor is called after set_tensors
            get_level_zero_inputs(foundPort.idx).resize(1);
            get_level_zero_inputs(foundPort.idx).shrink_to_fit();
            get_user_inputs(foundPort.idx).resize(1);
            get_user_inputs(foundPort.idx).shrink_to_fit();
        }

        get_user_input(foundPort.idx) = tensor;
    } else {
        if (_userOutputTensors.at(foundPort.idx)._ptr == tensor._ptr) {
            // Got set_tensor with the same object here too - do nothing
            _logger.debug("ZeroInferRequest::set_tensor - got the same tensor, do nothing");
            return;
        }
        _userOutputTensors.at(foundPort.idx) = tensor;
    }

    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
        auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(tensor._ptr);

        if (remoteTensor == nullptr) {
            _logger.debug("ZeroInferRequest::set_tensor - set new tensor");
            set_tensor_data(tensor._ptr, foundPort.idx, foundPort.is_input());
        } else {
            _logger.debug("ZeroInferRequest::set_tensor - set new remote tensor");
            set_remote_tensor_data(std::move(remoteTensor), foundPort.idx, foundPort.is_input());
        }
    }
}

void ZeroInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                   const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OV_ITT_TASK_CHAIN(SET_TENSORS, itt::domains::LevelZeroBackend, "set_tensors", "set_tensors");
    if (tensors.size() == 1) {
        set_tensor(port, tensors[0]);
        return;
    }

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find input tensor for port ", port);
    if (!foundPort.is_input()) {
        OPENVINO_THROW("set_input_tensors/set_tensors is not supported for output port.");
    }

    check_batched_tensors(port, tensors);

    get_user_inputs(foundPort.idx).resize(tensors.size());
    get_user_inputs(foundPort.idx) = tensors;

    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
        if (_graph->get_batch_size().has_value()) {
            for (size_t i = 0; i < tensors.size(); i++) {
                auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(tensors[i]._ptr);

                get_level_zero_inputs(foundPort.idx).resize(tensors.size());
                void* data = nullptr;

                if (remoteTensor == nullptr) {
                    bool tensorHasSameL0Context = false;

                    OV_ITT_TASK_NEXT(SET_TENSORS, "check_data_allocation");
                    if (zeroUtils::memory_was_allocated_in_the_same_l0_context(_initStructs->getContext(),
                                                                               tensors[i]->data())) {
                        _logger.debug("ZeroInferRequest::set_tensors - tensor was created in the same L0 context");

                        get_level_zero_input(foundPort.idx, i) = tensors.at(i)._ptr;
                        tensorHasSameL0Context = true;
                    }

                    if (!tensorHasSameL0Context) {
                        _logger.debug("ZeroInferRequest::set_tensors - tensor wasn't created in the same L0 context, "
                                      "create a L0 tensor");

                        get_level_zero_input(foundPort.idx, i) =
                            allocate_tensor(_metadata.inputs.at(foundPort.idx), foundPort.idx, true, *_inputAllocator);
                    }

                    data = get_level_zero_input(foundPort.idx, i)->data();
                } else {
                    _logger.debug("ZeroInferRequest::set_tensors - remote tensor is used");

                    data = remoteTensor->get_original_memory();

                    get_level_zero_input(foundPort.idx, i) = tensors.at(i)._ptr;
                }

                if (_pipelineIsCreated) {
                    OPENVINO_ASSERT(data, "Empty buffer");
                    OV_ITT_TASK_NEXT(SET_TENSORS, "updateCommandList");

                    _pipeline->update_graph_arguments_batching(_graph->get_input_descriptors().at(foundPort.idx).idx,
                                                               data,
                                                               i);
                }
            }
        }
    }
}

ov::SoPtr<ov::ITensor> ZeroInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "get_tensor");

    auto foundPort = find_port(port);
    OPENVINO_ASSERT(foundPort.found(), "Cannot find tensor for port ", port);

    const size_t ioIndex = foundPort.idx;
    const bool isInput = foundPort.is_input();

    if (isInput && is_batched_input(ioIndex)) {
        OPENVINO_THROW("Cannot return tensors in a tensor.");
    }

    auto& userTensors = isInput ? get_user_input(ioIndex) : _userOutputTensors.at(ioIndex);

    if (userTensors) {
        auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(userTensors._ptr);
        if (zeroTensor != nullptr) {
            zeroTensor->set_tensor_shared_with_user();
        }

        _logger.debug("ZeroInferRequest::get_tensor - tensor allocated, get the tensor");
        return userTensors;
    }

    auto& metadata = isInput ? _metadata.inputs.at(ioIndex) : _metadata.outputs.at(ioIndex);
    _logger.debug("ZeroInferRequest::get_tensor - tensor is not allocated, create tensor %s",
                  metadata.nodeFriendlyName.c_str());

    auto& levelZeroTensors = isInput ? get_level_zero_input(ioIndex) : _levelZeroOutputTensors.at(ioIndex);

    levelZeroTensors = allocate_tensor(metadata,
                                       ioIndex,
                                       isInput,
                                       isInput ? *_inputAllocator : *_outputAllocator,
                                       _graph->get_batch_size());

    auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(levelZeroTensors);
    if (zeroTensor != nullptr) {
        zeroTensor->set_tensor_shared_with_user();
    }

    return userTensors;
}

void ZeroInferRequest::update_pipeline_if_memory_changed() {
    size_t ioIndex = 0;

    for (const auto& levelZeroTensor : _levelZeroInputTensors) {
        const auto& inputDescriptor = _metadata.inputs.at(ioIndex);
        auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(levelZeroTensor.at(SINGLE_TENSOR));

        if (is_batched_input(ioIndex) || inputDescriptor.isShapeTensor ||
            is_remote_tensor(levelZeroTensor.at(SINGLE_TENSOR)) || zeroTensor == nullptr) {
            ++ioIndex;
            continue;
        }

        if (zeroTensor->memory_address_changed()) {
            _logger.debug("Update input graph descriptor with the new tensor");
            OPENVINO_ASSERT(zeroTensor->data(), "Empty buffer");

            _pipeline->update_graph_arguments(_graph->get_input_descriptors().at(ioIndex).idx,
                                              zeroTensor->data(),
                                              zeroTensor->get_byte_size());

            if (!inputDescriptor.isStateInput) {
                zeroTensor->reset_memory_flag();
            }
        }

        ++ioIndex;
    }

    ioIndex = 0;

    for (const auto& levelZeroTensor : _levelZeroOutputTensors) {
        const auto& outputDescriptor = _metadata.outputs.at(ioIndex);
        auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(levelZeroTensor);

        if (outputDescriptor.isShapeTensor || is_remote_tensor(levelZeroTensor) || zeroTensor == nullptr) {
            ++ioIndex;
            continue;
        }

        if (zeroTensor->memory_address_changed()) {
            _logger.debug("Update output graph descriptor with the new tensor");
            OPENVINO_ASSERT(zeroTensor->data(), "Empty buffer");

            _pipeline->update_graph_arguments(_graph->get_output_descriptors().at(ioIndex).idx,
                                              zeroTensor->data(),
                                              zeroTensor->get_byte_size());

            zeroTensor->reset_memory_flag();
        }

        ++ioIndex;
    }
}

void ZeroInferRequest::update_states_if_memory_changed() {
    for (const auto& variableState : _variableStates) {
        auto zeroState = std::dynamic_pointer_cast<ZeroVariableState>(variableState._ptr);

        OPENVINO_ASSERT(zeroState != nullptr, "State is not compatible with NPU plugin");

        if (zeroState->tensor_was_updated()) {
            get_user_input(zeroState->get_tensor_index()) = zeroState->get_state();
            _userOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_state();

            zeroState->reset_tensor_updated_flag();

            if (zeroState->zero_tensor_should_be_updated()) {
                auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(zeroState->get_state()._ptr);

                void* userBuffer = !remoteTensor ? zeroState->get_state()->data() : remoteTensor->get_original_memory();

                _pipeline->update_graph_arguments(_graphInputDescriptors.at(zeroState->get_tensor_index()).idx,
                                                  userBuffer,
                                                  zeroState->get_state()->get_byte_size());

                _pipeline->update_graph_arguments(_graphOutputDescriptors.at(zeroState->get_related_tensor_index()).idx,
                                                  userBuffer,
                                                  zeroState->get_state()->get_byte_size());

                zeroState->reset_zero_tensor_updated_flag();

                get_level_zero_input(zeroState->get_tensor_index()) = zeroState->get_state()._ptr;
                _levelZeroOutputTensors.at(zeroState->get_related_tensor_index()) = zeroState->get_state()._ptr;
            }
        }
    }
}

void ZeroInferRequest::infer() {
    if (_config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        OPENVINO_THROW("Only start async is supported when RUN_INFERENCES_SEQUENTIALLY is enabled!");
    }

    infer_async();
    get_result();
}

void ZeroInferRequest::infer_async() {
    _logger.debug("InferRequest::infer_async started");
    OV_ITT_TASK_CHAIN(ZERO_INFER, itt::domains::LevelZeroBackend, "infer_async", "start");

    {
        std::lock_guard<std::mutex> lock(_graph->get_mutex());

        if (!_pipelineIsCreated) {
            OV_ITT_TASK_NEXT(ZERO_INFER, "create_pipeline");
            create_pipeline();

            _pipelineIsCreated = true;
        } else {
            if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
                update_pipeline_if_memory_changed();
                update_states_if_memory_changed();
            }
        }
    }

    size_t inputIndex = 0;
    for (const auto& userTensor : _userInputTensors) {
        const IODescriptor inputDescriptor = _metadata.inputs.at(inputIndex);

        OPENVINO_ASSERT(!inputDescriptor.isInitInputWeights,
                        "This path should not be used for running inferences for the \"init\" model");

        if (inputDescriptor.isShapeTensor) {
            OPENVINO_ASSERT(inputDescriptor.relatedDescriptorIndex.has_value(),
                            "The link between the dynamic tensor and its shape tensor is missing, entry name: ",
                            inputDescriptor.nameFromCompiler);
            const auto& inputDims = get_user_input(*inputDescriptor.relatedDescriptorIndex)->get_shape();

            for (size_t i = 0; i < userTensor.at(SINGLE_TENSOR)->get_size(); ++i) {
                const auto reverseIdx = inputDims.size() - 1 - i;
                userTensor.at(SINGLE_TENSOR)->data<uint32_t>()[i] = static_cast<uint32_t>(inputDims[reverseIdx]);
            }
        }

        if (is_batched_input(inputIndex)) {
            if (_graph->get_batch_size().has_value()) {
                for (size_t i = 0; i < userTensor.size(); i++) {
                    if (!is_remote_tensor(get_level_zero_input(inputIndex, i))) {
                        void* levelZeroBuffer = get_level_zero_input(inputIndex, i)->data();

                        auto userBatchRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor.at(i)._ptr);

                        void* userBuffer = !userBatchRemoteTensor ? userTensor.at(i)->data()
                                                                  : userBatchRemoteTensor->get_original_memory();

                        if (userBuffer != levelZeroBuffer) {
                            if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                                OPENVINO_THROW("Empty buffer");
                            }

                            _logger.info("Batched Tensors - Tensor is not allocated in the current Level Zero context");
                            OV_ITT_TASK_NEXT(ZERO_INFER, "memcpy");
                            std::memcpy(levelZeroBuffer, userBuffer, userTensor.at(i)->get_byte_size());
                        }
                    }
                }
            } else {
                void* levelZeroBuffer = get_level_zero_input(inputIndex)->data();

                _logger.info("Batched Tensors - Tensor is not allocated in the current Level Zero context or must be "
                             "in a continued memory space");

                for (size_t i = 0; i < userTensor.size(); i++) {
                    auto userBatchRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor.at(i)._ptr);

                    void* userBuffer = !userBatchRemoteTensor ? userTensor.at(i)->data()
                                                              : userBatchRemoteTensor->get_original_memory();

                    std::memcpy(static_cast<unsigned char*>(levelZeroBuffer) + (i * userTensor.at(i)->get_byte_size()),
                                userBuffer,
                                userTensor.at(i)->get_byte_size());
                }
            }

            ++inputIndex;
            continue;
        }

        if (inputDescriptor.isMainInputWeights) {
            // These values were set while running the "WeightlessGraph::init" method
            continue;
        }

        auto userRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor.at(SINGLE_TENSOR)._ptr);
        void* userBuffer =
            !userRemoteTensor ? userTensor.at(SINGLE_TENSOR)->data() : userRemoteTensor->get_original_memory();

        const auto& levelZeroTensor = get_level_zero_input(inputIndex);
        if (!is_remote_tensor(levelZeroTensor)) {
            void* levelZeroBuffer = levelZeroTensor->data();

            if (userBuffer != levelZeroBuffer) {
                if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                    OPENVINO_THROW("Empty buffer");
                }

                _logger.info("Tensor is not allocated in the current Level Zero context");
                OV_ITT_TASK_NEXT(ZERO_INFER, "memcpy");
                std::memcpy(levelZeroBuffer, userBuffer, userTensor.at(SINGLE_TENSOR)->get_byte_size());
            }
        }

        ++inputIndex;
    }

    OV_ITT_TASK_NEXT(ZERO_INFER, "push");
    _pipeline->push();
}

void ZeroInferRequest::get_result() {
    OV_ITT_TASK_CHAIN(ZERO_RESULT, itt::domains::LevelZeroBackend, "get_result", "pull");
    _logger.debug("InferRequest::get_result start");
    _pipeline->pull();

    size_t outputIndex = 0;
    for (const auto& userTensor : _userOutputTensors) {
        const IODescriptor outputDescriptor = _metadata.outputs.at(outputIndex);
        if (outputDescriptor.isShapeTensor) {
            OPENVINO_ASSERT(outputDescriptor.relatedDescriptorIndex.has_value(),
                            "The link between the dynamic tensor and its shape tensor is missing, entry name: ",
                            outputDescriptor.nameFromCompiler);

            ov::Shape actualDims;
            actualDims.reserve(userTensor->get_size());

            for (size_t i = 0; i < userTensor->get_size(); ++i) {
                const auto reverseIdx = userTensor->get_size() - 1 - i;
                actualDims.push_back(userTensor->data<uint32_t>()[reverseIdx]);
            }
            auto& tensorToBeReshaped = _userOutputTensors.at(*outputDescriptor.relatedDescriptorIndex);
            tensorToBeReshaped->set_shape(actualDims);
        }

        auto userRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(userTensor._ptr);
        void* userBuffer = !userRemoteTensor ? userTensor->data() : userRemoteTensor->get_original_memory();

        const std::shared_ptr<ov::ITensor>& levelZeroTensor = _levelZeroOutputTensors.at(outputIndex);
        if (!is_remote_tensor(levelZeroTensor)) {
            void* levelZeroBuffer = levelZeroTensor->data();

            if (userBuffer != levelZeroBuffer) {
                if (userBuffer == nullptr || levelZeroBuffer == nullptr) {
                    OPENVINO_THROW("Empty buffer");
                }

                _logger.info("Tensor is not allocated in the current Level Zero context");
                OV_ITT_TASK_NEXT(ZERO_RESULT, "memcpy");
                std::memcpy(userBuffer, levelZeroBuffer, userTensor->get_byte_size());
            }
        }

        ++outputIndex;
    }

    OV_ITT_TASK_NEXT(ZERO_RESULT, "reset");
    _pipeline->reset();
    _logger.debug("InferRequest::get_result finished");
}

void ZeroInferRequest::check_network_precision(const ov::element::Type_t precision) const {
    switch (precision) {
    case ov::element::Type_t::f32:
        break;
    case ov::element::Type_t::f16:
        break;
    case ov::element::Type_t::bf16:
        break;
    case ov::element::Type_t::f8e4m3:
        break;
    case ov::element::Type_t::f8e5m2:
        break;
    case ov::element::Type_t::f8e8m0:
        break;
    case ov::element::Type_t::nf4:
        break;
    case ov::element::Type_t::u2:
        break;
    case ov::element::Type_t::u4:
        break;
    case ov::element::Type_t::i4:
        break;
    case ov::element::Type_t::u8:
        break;
    case ov::element::Type_t::i8:
        break;
    case ov::element::Type_t::u16:
        break;
    case ov::element::Type_t::i16:
        break;
    case ov::element::Type_t::u32:
        break;
    case ov::element::Type_t::i32:
        break;
    case ov::element::Type_t::u64:
        break;
    case ov::element::Type_t::i64:
        break;
    case ov::element::Type_t::f64:
        break;
    case ov::element::Type_t::boolean:
        break;
    default:
        OPENVINO_THROW(
            "Unsupported tensor precision: " + ov::element::Type(precision).get_type_name() +
            "! Supported precisions: FP32, FP16, BF16, FP8, NF4, U2, U4, I4, U8, I8, U16, I16, U32, I32, U64, "
            "I64, FP64, BOOLEAN");
    }
}

std::vector<ov::ProfilingInfo> ZeroInferRequest::get_profiling_info() const {
    OPENVINO_ASSERT(_pipeline, "Profiling information isn't available before running an inference!");

    return _pipeline->get_profiling_info();
}

std::shared_ptr<ov::ITensor> ZeroInferRequest::create_tensor(ov::element::Type type,
                                                             const ov::Shape& shape,
                                                             const ov::Allocator& allocator) const {
    OPENVINO_ASSERT(allocator, "Allocator mush be provided when creating a zero tensor!");

    return std::make_shared<ZeroTensor>(_initStructs, _config, type, shape, allocator);
}

void ZeroInferRequest::add_state(const IODescriptor& descriptor, size_t tensorIndex) const {
    OPENVINO_ASSERT(descriptor.relatedDescriptorIndex.has_value(),
                    "The link between state descriptors is missing, state name: ",
                    descriptor.nameFromCompiler);

    _variableStates.push_back(std::make_shared<ZeroVariableState>(_initStructs,
                                                                  descriptor.nameFromCompiler,
                                                                  get_user_input(tensorIndex),
                                                                  tensorIndex,
                                                                  descriptor.relatedDescriptorIndex.value(),
                                                                  _config));
}

std::shared_ptr<ov::ITensor>& ZeroInferRequest::get_level_zero_input(size_t index, size_t tensorNo) const {
    return _levelZeroInputTensors.at(index).at(tensorNo);
}

std::vector<std::shared_ptr<ov::ITensor>>& ZeroInferRequest::get_level_zero_inputs(size_t index) const {
    return _levelZeroInputTensors.at(index);
}
