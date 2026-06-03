// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include <mutex>

#include "intel_npu/common/idynamic_graph.hpp"
#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/so_ptr.hpp"

#include "dynamic_graph.hpp"
#include "intel_npu/runtime/npu_vm_runtime.hpp"

namespace intel_npu {
class DynamicGraph final : public IDynamicGraph {
public:
    ///这个还有必要吗？删了DynamicGraphImpl就没必要了
    class Impl {
    public:
        virtual void initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata) = 0;
        virtual uint64_t getNumSubgraphs() = 0;
        virtual npu_vm_runtime_handle_t getVmRuntimeHandle() const = 0;
        virtual ~Impl() {};
    };

    DynamicGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                 ov::Tensor blob,
                 bool blobAllocatedByPlugin,
                 const FilteredConfig& config);

    std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream) const override;

    ze_graph_handle_t get_handle() const override;

    ~DynamicGraph() override;

    const NetworkMetadata& get_metadata() const override;
    ///如果要删除DynamicGraphImpl, 那么这部分内容需要找一个地方更新metadata

    void update_network_name(std::string_view name) override;

    CommandQueueDesc get_command_queue_desc() const override;
    void set_workload_type(const ov::WorkloadType workloadType) override;
    void set_model_priority(const ov::hint::Priority modelPriority) override;

    void set_batch_size(std::size_t batch) override;

    const std::optional<std::size_t> get_batch_size() const override;

    uint32_t get_unique_id() override;
    void set_last_submitted_id(uint32_t id_index) override;
    uint32_t get_last_submitted_id() const override;

    npu_vm_runtime_handle_t get_vm_runtime_handle() const override;
    //返回的是DynamicGraphImpl的内容，那这部分也可以删除

    uint64_t get_num_subgraphs() const override;
    ///这个和_engineProperties.numOfSubGraphs;相关，删除DynamicGraphImpl后，
    //需要在engine创建后拿到这个值，或许这样的话，就不需要这个函数了，
    //传入engineProperties,拿到这个值
    //直接在调用地拿到这个值

    std::optional<bool> is_profiling_blob() const override;

    std::optional<std::string_view> get_compatibility_descriptor() const override;

    //需要全局共享runtimeContext
    void getvmRuntimeContext(npu_vm_runtime_execution_context_handle_t args){
        //params->executionContext;
        // std::shared_ptr<DynamicArgumentsImpl> argsImpl =
        // args._impl ? std::static_pointer_cast<DynamicArgumentsImpl>(args._impl)
        //            : std::make_shared<DynamicArgumentsImpl>()
        // npu_vm_runtime_execute_params_t* params = &argsImpl->_executeParams;

        args = _executeParams.executionContext;
        args = _executionContext;
    }

    //这里的engine可以从 DynamicGraphImpl::_engine 中拿到
    void createVmRuntimeContext(npu_vm_runtime_handle_t engine, npu_vm_runtime_execution_context_handle_t executionContext){
        //params->executionContext;
        // std::shared_ptr<DynamicArgumentsImpl> argsImpl =
        // args._impl ? std::static_pointer_cast<DynamicArgumentsImpl>(args._impl)
        //            : std::make_shared<DynamicArgumentsImpl>()
        // npu_vm_runtime_execute_params_t* params = &argsImpl->_executeParams;

        if (executionContext == nullptr) {
            if (npuVMRuntimeCreateExecutionContext(engine, executionContext) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
                OPENVINO_THROW("Failed to create a VM execution context");
            } else {
                _logger.debug("Execution context is created successfully.");
            }
        }
    }

private:
    void initialize_impl(const FilteredConfig& config) override;

    bool release_blob(const FilteredConfig& config);
    std::optional<size_t> determine_batch_size();

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    NetworkMetadata _metadata;

    /**
     * @brief Stores the number of subgraphs for dynamic models
     * @note the number of subgraphs will be one for static models
     */
    uint64_t _num_of_subgraphs = 1;

    mutable std::mutex _commandQueueDescMutex;
    CommandQueueDesc _commandQueueDesc;
    std::vector<std::shared_ptr<Event>> _lastSubmittedEvent;

    std::optional<ov::Tensor> _blob;

    // In the case of the import path, the blob is released after graph initialization so it can not be any longer
    // exported
    bool _blobIsReleased = false;

    uint32_t _uniqueId = 0;
    uint32_t _lastSubmittedId = 0;

    /**
     * @brief The batch size used by the corresponding model.
     * @details The attribute contains a value only if the plugin performs the batches splitting operation.
     */
    std::optional<std::size_t> _batchSize = std::nullopt;

    Logger _logger;

    std::unique_ptr<Impl> _impl;//指向dynamicGraphImpl

    //context 需要的内容
    npu_vm_runtime_execute_params_t _executeParams;
    npu_vm_runtime_execution_context_handle_t _executionContext;
};

}  // namespace intel_npu
