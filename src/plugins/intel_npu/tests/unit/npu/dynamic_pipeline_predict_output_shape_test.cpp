// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <unordered_map>
#include <vector>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "zero_dynamic_pipeline.hpp"

namespace intel_npu {
namespace {

struct FakeGraph final : public IGraph {
    explicit FakeGraph(void* handle) : _handle(handle) {}

    void* get_handle() const override {
        return _handle;
    }

    std::optional<bool> is_profiling_blob() const override {
        return std::nullopt;
    }

private:
    void* _handle = nullptr;
};

struct FakeMemRefState {
    const void* basePtr = nullptr;
    const void* data = nullptr;
    int64_t offset = 0;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    int64_t dimsCount = 0;
};

std::unordered_map<uintptr_t, FakeMemRefState> g_memRefs;
int g_predictShapeCalls = 0;
bool g_forcePredictFailure = false;
int g_createMemRefCalls = 0;
int g_setMemRefCalls = 0;
int g_parseMemRefCalls = 0;
int g_createExecutionContextCalls = 0;

npu_vm_runtime_result_t fakeCreateMemRef(int64_t dimsCount, npu_vm_runtime_mem_ref_handle_t* phMemRef) {
    ++g_createMemRefCalls;
    auto* token = new int(1);
    *phMemRef = reinterpret_cast<npu_vm_runtime_mem_ref_handle_t>(token);

    FakeMemRefState state;
    state.dimsCount = dimsCount;
    state.sizes.assign(static_cast<size_t>(dimsCount), 0);
    state.strides.assign(static_cast<size_t>(dimsCount), 0);
    g_memRefs.emplace(reinterpret_cast<uintptr_t>(*phMemRef), std::move(state));

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

npu_vm_runtime_result_t fakeDestroyMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef) {
    auto key = reinterpret_cast<uintptr_t>(hMemRef);
    g_memRefs.erase(key);
    delete reinterpret_cast<int*>(hMemRef);
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

npu_vm_runtime_result_t fakeSetMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef,
                                      const void* basePtr,
                                      const void* data,
                                      int64_t offset,
                                      int64_t* pSizes,
                                      int64_t* pStrides,
                                      int64_t dimsCount) {
    ++g_setMemRefCalls;
    auto key = reinterpret_cast<uintptr_t>(hMemRef);
    auto& state = g_memRefs[key];

    state.basePtr = basePtr;
    state.data = data;
    state.offset = offset;
    state.dimsCount = dimsCount;
    state.sizes.assign(pSizes, pSizes + dimsCount);
    state.strides.assign(pStrides, pStrides + dimsCount);

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

npu_vm_runtime_result_t fakeParseMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef,
                                        const void** pBasePtr,
                                        const void** pData,
                                        int64_t* pOffset,
                                        int64_t* pSizes,
                                        int64_t* pStrides,
                                        int64_t* pDimsCount) {
    ++g_parseMemRefCalls;
    auto key = reinterpret_cast<uintptr_t>(hMemRef);
    const auto& state = g_memRefs.at(key);

    *pBasePtr = state.basePtr;
    *pData = state.data;
    *pOffset = state.offset;
    *pDimsCount = state.dimsCount;

    for (int64_t i = 0; i < state.dimsCount; ++i) {
        pSizes[i] = state.sizes[static_cast<size_t>(i)];
        pStrides[i] = state.strides[static_cast<size_t>(i)];
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

npu_vm_runtime_result_t fakeCreateExecutionContext(npu_vm_runtime_handle_t,
                                                   npu_vm_runtime_execution_context_handle_t* phExecutionHandle) {
    ++g_createExecutionContextCalls;
    *phExecutionHandle = reinterpret_cast<npu_vm_runtime_execution_context_handle_t>(0x9999);
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

npu_vm_runtime_result_t fakeDestroyExecutionContext(npu_vm_runtime_execution_context_handle_t) {
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

npu_vm_runtime_result_t fakePredictOutputShape(npu_vm_runtime_handle_t,
                                               npu_vm_runtime_predict_output_shape_params_t* pParams) {
    ++g_predictShapeCalls;

    if (g_forcePredictFailure) {
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    std::vector<int64_t> inSizes;
    int64_t dimsCount = 0;
    if (pParams->numOfInputs > 0) {
        const auto inKey = reinterpret_cast<uintptr_t>(pParams->pInputs[0]);
        const auto& inState = g_memRefs.at(inKey);
        dimsCount = inState.dimsCount;
        inSizes = inState.sizes;
    }

    for (uint32_t i = 0; i < pParams->numOfOutputs; ++i) {
        auto outKey = reinterpret_cast<uintptr_t>(pParams->pOutputs[i]);
        auto& outState = g_memRefs[outKey];

        if (dimsCount > 0) {
            outState.dimsCount = dimsCount;
            outState.sizes = inSizes;
            // Make prediction observable for assertions.
            outState.sizes[0] += 1;

            outState.strides.assign(static_cast<size_t>(dimsCount), 0);
            int64_t stride = 1;
            for (int64_t d = dimsCount - 1; d >= 0; --d) {
                outState.strides[static_cast<size_t>(d)] = stride;
                stride *= outState.sizes[static_cast<size_t>(d)];
            }
        }
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

struct RuntimeFunctionGuard {
    decltype(intel_npu::npuVMRuntimeCreateMemRef) createMemRef = intel_npu::npuVMRuntimeCreateMemRef;
    decltype(intel_npu::npuVMRuntimeDestroyMemRef) destroyMemRef = intel_npu::npuVMRuntimeDestroyMemRef;
    decltype(intel_npu::npuVMRuntimeSetMemRef) setMemRef = intel_npu::npuVMRuntimeSetMemRef;
    decltype(intel_npu::npuVMRuntimeParseMemRef) parseMemRef = intel_npu::npuVMRuntimeParseMemRef;
    decltype(intel_npu::npuVMRuntimeCreateExecutionContext) createExecutionContext =
        intel_npu::npuVMRuntimeCreateExecutionContext;
    decltype(intel_npu::npuVMRuntimeDestroyExecutionContext) destroyExecutionContext =
        intel_npu::npuVMRuntimeDestroyExecutionContext;
    decltype(intel_npu::npuVMRuntimePredictOutputShape) predictOutputShape = intel_npu::npuVMRuntimePredictOutputShape;

    ~RuntimeFunctionGuard() {
        intel_npu::npuVMRuntimeCreateMemRef = createMemRef;
        intel_npu::npuVMRuntimeDestroyMemRef = destroyMemRef;
        intel_npu::npuVMRuntimeSetMemRef = setMemRef;
        intel_npu::npuVMRuntimeParseMemRef = parseMemRef;
        intel_npu::npuVMRuntimeCreateExecutionContext = createExecutionContext;
        intel_npu::npuVMRuntimeDestroyExecutionContext = destroyExecutionContext;
        intel_npu::npuVMRuntimePredictOutputShape = predictOutputShape;
    }
};

void installRuntimeMocks() {
    intel_npu::npuVMRuntimeCreateMemRef = fakeCreateMemRef;
    intel_npu::npuVMRuntimeDestroyMemRef = fakeDestroyMemRef;
    intel_npu::npuVMRuntimeSetMemRef = fakeSetMemRef;
    intel_npu::npuVMRuntimeParseMemRef = fakeParseMemRef;
    intel_npu::npuVMRuntimeCreateExecutionContext = fakeCreateExecutionContext;
    intel_npu::npuVMRuntimeDestroyExecutionContext = fakeDestroyExecutionContext;
    intel_npu::npuVMRuntimePredictOutputShape = fakePredictOutputShape;
}

}  // namespace

TEST(DynamicPipelinePredictShapeTest, PredictShapeUpdatesOutputsAndClearsHandleVectors) {
    RuntimeFunctionGuard guard;
    installRuntimeMocks();

    g_memRefs.clear();
    g_predictShapeCalls = 0;
    g_forcePredictFailure = false;

    FakeGraph graph(reinterpret_cast<void*>(0x1111));
    DynamicArguments args;

    std::vector<MemRefType> inputs(1);
    std::vector<MemRefType> outputs(1);

    int inputData = 1;
    int outputData = 2;

    inputs[0].setArg(&inputData);
    inputs[0].setSize(ov::Shape{2, 3});
    inputs[0].setStrides(ov::Strides{3, 1});

    outputs[0].setArg(&outputData);
    outputs[0].setSize(ov::Shape{1, 1});
    outputs[0].setStrides(ov::Strides{1, 1});

    DynamicPipeline::predict_output_shape(graph, args, inputs, outputs);

    EXPECT_EQ(g_predictShapeCalls, 1);
    EXPECT_EQ(outputs[0]._dimsCount, 2);
    EXPECT_EQ(outputs[0]._sizes, (std::vector<int64_t>{3, 3}));
    EXPECT_EQ(outputs[0]._strides, (std::vector<int64_t>{3, 1}));

    EXPECT_TRUE(args._inputMemRefHandles.empty());
    EXPECT_TRUE(args._outputMemRefHandles.empty());
}

TEST(DynamicPipelinePredictShapeTest, PredictShapeCallsExpectedVmRuntimeApiFlow) {
    RuntimeFunctionGuard guard;
    installRuntimeMocks();

    g_memRefs.clear();
    g_predictShapeCalls = 0;
    g_forcePredictFailure = false;
    g_createMemRefCalls = 0;
    g_setMemRefCalls = 0;
    g_parseMemRefCalls = 0;
    g_createExecutionContextCalls = 0;

    FakeGraph graph(reinterpret_cast<void*>(0x3333));
    DynamicArguments args;

    std::vector<MemRefType> inputs(1);
    std::vector<MemRefType> outputs(1);

    int inputData = 11;
    int outputData = 22;

    inputs[0].setArg(&inputData);
    inputs[0].setSize(ov::Shape{2, 3});
    inputs[0].setStrides(ov::Strides{3, 1});

    outputs[0].setArg(&outputData);
    outputs[0].setSize(ov::Shape{1, 1});
    outputs[0].setStrides(ov::Strides{1, 1});

    DynamicPipeline::predict_output_shape(graph, args, inputs, outputs);

    // input+output each create one memref
    EXPECT_EQ(g_createMemRefCalls, 2);
    // input+output each set one memref
    EXPECT_EQ(g_setMemRefCalls, 2);
    // context created once for this call
    EXPECT_EQ(g_createExecutionContextCalls, 1);
    // VM prediction called exactly once
    EXPECT_EQ(g_predictShapeCalls, 1);
    // output alignWithHandle() parses one memref
    EXPECT_EQ(g_parseMemRefCalls, 1);

    // API clears temporary vectors after prediction
    EXPECT_TRUE(args._inputMemRefHandles.empty());
    EXPECT_TRUE(args._outputMemRefHandles.empty());
}

TEST(DynamicPipelinePredictShapeTest, PredictShapeThrowsWhenRuntimePredictionFails) {
    RuntimeFunctionGuard guard;
    installRuntimeMocks();

    g_memRefs.clear();
    g_predictShapeCalls = 0;
    g_forcePredictFailure = true;

    FakeGraph graph(reinterpret_cast<void*>(0x2222));
    DynamicArguments args;

    std::vector<MemRefType> inputs(1);
    std::vector<MemRefType> outputs(1);

    int inputData = 1;
    int outputData = 2;

    inputs[0].setArg(&inputData);
    inputs[0].setSize(ov::Shape{2, 3});
    inputs[0].setStrides(ov::Strides{3, 1});

    outputs[0].setArg(&outputData);
    outputs[0].setSize(ov::Shape{1, 1});
    outputs[0].setStrides(ov::Strides{1, 1});

    EXPECT_THROW(DynamicPipeline::predict_output_shape(graph, args, inputs, outputs), ov::Exception);
    EXPECT_EQ(g_predictShapeCalls, 1);
}

TEST(DynamicPipelinePredictShapeTest, PredictShapeThrowsOnNullGraphHandle) {
    RuntimeFunctionGuard guard;
    installRuntimeMocks();

    g_memRefs.clear();
    g_predictShapeCalls = 0;
    g_forcePredictFailure = false;

    FakeGraph graph(nullptr);
    DynamicArguments args;
    std::vector<MemRefType> inputs;
    std::vector<MemRefType> outputs;

    EXPECT_THROW(DynamicPipeline::predict_output_shape(graph, args, inputs, outputs), ov::Exception);
}

}  // namespace intel_npu
