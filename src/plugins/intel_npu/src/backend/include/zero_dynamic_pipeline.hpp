// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/network_metadata.hpp"
#include "zero_dynamic_arguments.hpp"
#include "zero_pipeline.hpp"

namespace intel_npu {

class DynamicPipeline final : public IPipeline {
    struct PipelinedCommandLists {
        std::shared_ptr<DynamicPipelineArguments> _arguments;

        std::vector<std::unique_ptr<CommandList>> _commandLists;
        // Store command list handles to pass it to ExecutionEngine
        std::vector<ze_command_list_handle_t> _commandListHandles;

        PipelinedCommandLists(size_t numCommandLists,
                              const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                              std::shared_ptr<DynamicPipelineArguments> args) {
            _commandLists.reserve(numCommandLists);
            for (size_t i = 0; i < numCommandLists; i++) {
                _commandLists.emplace_back(std::make_unique<CommandList>(init_structs));
            }

            for (size_t i = 0; i < numCommandLists; i++) {
                _commandListHandles.push_back(_commandLists[i]->handle());
            }

            if (args != nullptr) {
                _arguments = args;

            } else {
                _arguments = std::make_shared<DynamicPipelineArguments>();
            }
        }

        size_t size() const {
            return _commandListHandles.size();
        }

        ze_command_list_handle_t* data() {
            return _commandListHandles.data();
        }

        /// Allocate per-IO MemRef slots driven by the network metadata. The pipeline ctor fills
        /// each slot's data/shape/strides via setArgumentProperties again.
        void initArgumentsInOutParam(const NetworkMetadata& metadata) {
            _arguments->resizeInputs(metadata.inputs.size());
            auto& inputs = _arguments->inputs();
            for (size_t i = 0; i < inputs.size(); ++i) {
                // Use size as placeholder of stride
                // For now, only considering the usage and subsequent comparison of dimcount, shape, and strides
                const auto& shape = metadata.inputs[i].shapeFromCompiler.get_shape();
                inputs[i].setContiguousShape(shape);
            }

            _arguments->resizeOutputs(metadata.outputs.size());
            auto& outputs = _arguments->outputs();
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto& shape = metadata.outputs[i].shapeFromCompiler.get_shape();
                outputs[i].setContiguousShape(shape);
            }
        }

        std::vector<ze_command_list_handle_t>& getHandles() {
            return _commandListHandles;
        }

        DynamicPipelineArguments& getArguments() {
            return *_arguments;
        }

        void updateMutableCommandList(uint32_t arg_index,
                                      const void* arg_value,
                                      const ov::Strides& strides,
                                      const ov::Shape& shapes) {
            // The strides are already divided by element size
            auto& inputs = _arguments->inputs();
            if (arg_index < inputs.size()) {
                inputs[arg_index].setProperties(arg_value, shapes, strides);
            } else {
                auto& outputs = _arguments->outputs();
                size_t output_index = static_cast<size_t>(arg_index) - inputs.size();
                if (output_index < outputs.size()) {
                    outputs[output_index].setProperties(arg_value, shapes, strides);
                }
            }
        }

        void resetCommandList() {
            for (auto& cmd_list : _commandLists) {
                cmd_list->reset();
            }
        }
    };

public:
    DynamicPipeline(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                    const std::shared_ptr<IGraph>& graph,
                    const Config& config,
                    const std::vector<std::vector<std::shared_ptr<ZeroTensor>>>& input_tensors,
                    const std::vector<std::shared_ptr<ZeroTensor>>& output_tensors,
                    std::shared_ptr<DynamicPipelineArguments> arguments,
                    size_t batch_size = 1);

    DynamicPipeline(const DynamicPipeline&) = delete;
    DynamicPipeline& operator=(const DynamicPipeline&) = delete;
    ~DynamicPipeline() override = default;

    void push() override;
    void pull() override;
    void reset() const override;
    void update_graph_arguments(uint32_t index,
                                const std::shared_ptr<ZeroTensor>& tensor,
                                const std::shared_ptr<ov::ITensor>& userTensor = nullptr) override;
    void update_graph_arguments(uint32_t index,
                                const std::shared_ptr<ZeroTensor>& tensor,
                                size_t batch_index,
                                const std::shared_ptr<ov::ITensor>& userTensor = nullptr) override;

    /// Run VM-runtime output shape prediction. Independent of pipeline instance state
    /// (depends only on the graph's VM runtime handle)
    static void predict_output_shape(const IGraph& graph,
                                     DynamicPipelineArguments& args,
                                     std::vector<VmMemRefDescriptor>& inputsMemRef,
                                     std::vector<VmMemRefDescriptor>& outputsMemRef);

private:
    void execute_vm_runtime(npu_vm_runtime_handle_t vmRuntime,
                            DynamicPipelineArguments& args,
                            std::vector<ze_command_list_handle_t>& commandLists,
                            ze_command_queue_handle_t commandQueue,
                            ze_fence_handle_t fence,
                            ze_event_handle_t event);

    std::vector<std::unique_ptr<PipelinedCommandLists>> _command_lists;
};

}  // namespace intel_npu
