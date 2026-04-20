// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>

#include "intel_npu/common/idynamic_graph.hpp"
#include "zero_pipeline.hpp"

namespace intel_npu {

class DynamicPipeline final : public IPipeline {
    struct PipelinedCommandLists {
        mutable IDynamicGraph::GraphArguments _binding;

        std::vector<std::unique_ptr<CommandList>> _commandLists;
        // Store command list handles to pass it to ExecutionEngine
        std::vector<ze_command_list_handle_t> _commandListHandles;

        PipelinedCommandLists(size_t numCommandLists, const std::shared_ptr<ZeroInitStructsHolder>& init_structs) {
            _commandLists.reserve(numCommandLists);
            for (size_t i = 0; i < numCommandLists; i++) {
                _commandLists.emplace_back(std::make_unique<CommandList>(init_structs));
            }

            for (size_t i = 0; i < numCommandLists; i++) {
                _commandListHandles.push_back(_commandLists[i]->handle());
            }
        }

        size_t size() const {
            return _commandListHandles.size();
        }

        ze_command_list_handle_t* data() {
            return _commandListHandles.data();
        }

        void initializeBinding(const NetworkMetadata& metadata) {
            _binding._inputs.resize(metadata.inputs.size());
            for (size_t i = 0; i < metadata.inputs.size(); ++i) {
                const auto& shape = metadata.inputs[i].shapeFromCompiler.get_shape();
                std::vector<int64_t> shapeVec(shape.begin(), shape.end());
                _binding._inputs[i] = IDynamicGraph::MemRefType(nullptr, nullptr, 0, shapeVec, shapeVec, shapeVec.size());
                _binding._inputs[i].updateStride();
            }

            _binding._outputs.resize(metadata.outputs.size());
            for (size_t i = 0; i < metadata.outputs.size(); ++i) {
                const auto& shape = metadata.outputs[i].shapeFromCompiler.get_shape();
                std::vector<int64_t> shapeVec(shape.begin(), shape.end());
                _binding._outputs[i] = IDynamicGraph::MemRefType(nullptr, nullptr, 0, shapeVec, shapeVec, shapeVec.size());
                _binding._outputs[i].updateStride();
            }
        }

        std::vector<ze_command_list_handle_t>& getHandles() {
            return _commandListHandles;
        }

        IDynamicGraph::GraphArguments& getBinding() {
            return _binding;
        }

        void updateMutableCommandList(uint32_t arg_index,
                                      const void* arg_value,
                                      const ov::Strides& strides,
                                      const ov::Shape& shapes) {
            if (arg_index < _binding._inputs.size()) {
                _binding._inputs[arg_index].setArg(arg_value);
                // Only store the valid shape dimensions
                for (int64_t i = 0; i < _binding._inputs[arg_index]._dimsCount; i++) {
                    _binding._inputs[arg_index]._sizes[i] = shapes[i];
                }

                if (!strides.empty()) {
                    for (int64_t i = 0; i < _binding._inputs[arg_index]._dimsCount; i++) {
                        _binding._inputs[arg_index]._strides[i] = strides[i];
                    }
                } else {
                    // Need stride based on element but not byte, calc from shape
                    _binding._inputs[arg_index].updateStride();
                }
            } else {
                size_t output_index = static_cast<size_t>(arg_index) - _binding._inputs.size();
                if (output_index < _binding._outputs.size()) {
                    _binding._outputs[output_index].setArg(arg_value);

                    // Only store the valid shape dimensions
                    for (int64_t i = 0; i < _binding._outputs[output_index]._dimsCount; i++) {
                        _binding._outputs[output_index]._sizes[i] = shapes[i];
                    }

                    if (!strides.empty()) {
                        for (int64_t i = 0; i < _binding._outputs[output_index]._dimsCount; i++) {
                            _binding._outputs[output_index]._strides[i] = strides[i];
                        }
                    } else {
                        // Need stride based on element but not byte, calc from shape
                        _binding._outputs[output_index].updateStride();
                    }
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

    const IDynamicGraph::GraphArguments& get_graph_arguments(size_t batch_index) const;

private:
    void predict_and_update_outputs(IDynamicGraph* dynamicGraph);
    size_t get_output_metadata_index(uint32_t driver_arg_index) const;
    static ov::Shape memref_to_shape(const IDynamicGraph::MemRefType& memref);
    static size_t memref_num_elements(const IDynamicGraph::MemRefType& memref);

    std::vector<std::unique_ptr<PipelinedCommandLists>> _command_lists;
    std::vector<std::shared_ptr<ZeroTensor>> _output_tensors;
    std::unordered_map<uint32_t, size_t> _output_index_by_driver;
    std::vector<bool> _output_uses_user_tensor;
    bool _graph_arguments_changed = true;
};

}  // namespace intel_npu
