// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_npu/utils/vm/dynamic_arguments.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "zero_dynamic_pipeline.hpp"

namespace intel_npu {
namespace {

namespace {
struct RuntimeFunctionGuard {
    decltype(intel_npu::npuVMRuntimeCreateExecutionContext) originalCreate =
        intel_npu::npuVMRuntimeCreateExecutionContext;
    decltype(intel_npu::npuVMRuntimeDestroyExecutionContext) originalDestroy =
        intel_npu::npuVMRuntimeDestroyExecutionContext;

    ~RuntimeFunctionGuard() {
        intel_npu::npuVMRuntimeCreateExecutionContext = originalCreate;
        intel_npu::npuVMRuntimeDestroyExecutionContext = originalDestroy;
    }
};

int g_createExecutionContextCalls = 0;
int g_destroyExecutionContextCalls = 0;
npu_vm_runtime_execution_context_handle_t g_lastDestroyedContext = nullptr;

npu_vm_runtime_result_t fakeCreateExecutionContext(
    npu_vm_runtime_handle_t,
    npu_vm_runtime_execution_context_handle_t* phExecutionHandle) {
    ++g_createExecutionContextCalls;
    *phExecutionHandle = reinterpret_cast<npu_vm_runtime_execution_context_handle_t>(0x1234);
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

npu_vm_runtime_result_t fakeDestroyExecutionContext(npu_vm_runtime_execution_context_handle_t phExecutionHandle) {
    ++g_destroyExecutionContextCalls;
    g_lastDestroyedContext = phExecutionHandle;
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

npu_vm_runtime_result_t fakeCreateExecutionContextFail(
    npu_vm_runtime_handle_t,
    npu_vm_runtime_execution_context_handle_t*) {
    ++g_createExecutionContextCalls;
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}
}  // namespace

TEST(MemRefTypeTest, SetStridesBeforeSetSizeThrows) {
    MemRefType memRef;

    EXPECT_THROW(memRef.setStrides(ov::Strides{1, 1}), ov::Exception);
}

TEST(MemRefTypeTest, SetSizeAndSetStridesSuccess) {
    MemRefType memRef;

    memRef.setSize(ov::Shape{2, 3, 4});
    memRef.setStrides(ov::Strides{12, 4, 1});

    ASSERT_EQ(memRef._dimsCount, 3);
    EXPECT_EQ(memRef._sizes, (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(memRef._strides, (std::vector<int64_t>{12, 4, 1}));
}

TEST(MemRefTypeTest, SetSizeWithDifferentRankThrows) {
    MemRefType memRef;

    memRef.setSize(ov::Shape{2, 3});

    EXPECT_THROW(memRef.setSize(ov::Shape{2, 3, 4}), ov::Exception);
}

TEST(MemRefTypeTest, UpdateStrideComputesNchwLikeOrder) {
    MemRefType memRef;

    memRef.setSize(ov::Shape{2, 3, 4});
    memRef.updateStride();

    EXPECT_EQ(memRef._strides, (std::vector<int64_t>{12, 4, 1}));
}

TEST(MemRefTypeTest, CompareChecksOnlyShapeAndStride) {
    MemRefType left;
    MemRefType right;

    left._basePtr = reinterpret_cast<void*>(0x1111);
    right._basePtr = reinterpret_cast<void*>(0x2222);
    left._data = reinterpret_cast<void*>(0x3333);
    right._data = reinterpret_cast<void*>(0x4444);

    left.setSize(ov::Shape{2, 3});
    left.setStrides(ov::Strides{3, 1});

    right.setSize(ov::Shape{2, 3});
    right.setStrides(ov::Strides{3, 1});

    EXPECT_TRUE(left.compare(right));

    right.setStrides(ov::Strides{4, 1});
    EXPECT_FALSE(left.compare(right));
}

TEST(MemRefTypeTest, CopyConstructorDropsRuntimeImpl) {
    MemRefType original;
    original._impl = std::make_shared<int>(7);
    original.setSize(ov::Shape{2, 3});
    original.setStrides(ov::Strides{3, 1});

    MemRefType copied(original);

    EXPECT_EQ(copied._impl, nullptr);
    EXPECT_EQ(copied._sizes, original._sizes);
    EXPECT_EQ(copied._strides, original._strides);
}

TEST(MemRefTypeTest, CopyAssignmentDropsRuntimeImpl) {
    MemRefType original;
    original._impl = std::make_shared<int>(11);
    original.setSize(ov::Shape{3, 2});
    original.setStrides(ov::Strides{2, 1});

    MemRefType assigned;
    assigned._impl = std::make_shared<int>(22);
    assigned = original;

    EXPECT_EQ(assigned._impl, nullptr);
    EXPECT_EQ(assigned._sizes, original._sizes);
    EXPECT_EQ(assigned._strides, original._strides);
}

TEST(MemRefTypeTest, SetFromTensorUsesTensorShapeAndStrides) {
    MemRefType memRef;

    auto tensor = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{2, 3, 4}));
    float* rawPtr = reinterpret_cast<float*>(tensor->data());
    memRef.set(rawPtr, 0, tensor);

    ASSERT_EQ(memRef._dimsCount, 3);
    EXPECT_EQ(memRef._basePtr, rawPtr);
    EXPECT_EQ(memRef._data, rawPtr);
    EXPECT_EQ(memRef._sizes, (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(memRef._strides, (std::vector<int64_t>{12, 4, 1}));
}

TEST(DynamicArgumentsTest, SetArgumentPropertiesInitializesInputSlot) {
    DynamicArguments args;
    args._inputs.resize(1);
    args._outputs.resize(1);

    int inputData = 1;
    args.setArgumentProperties(0, &inputData, ov::Shape{2, 3}, std::vector<size_t>{3, 1});

    ASSERT_EQ(args._inputs[0]._dimsCount, 2);
    EXPECT_EQ(args._inputs[0]._basePtr, &inputData);
    EXPECT_EQ(args._inputs[0]._data, &inputData);
    EXPECT_EQ(args._inputs[0]._sizes, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(args._inputs[0]._strides, (std::vector<int64_t>{3, 1}));
}

TEST(DynamicArgumentsTest, SetArgumentPropertiesInitializesOutputSlot) {
    DynamicArguments args;
    args._inputs.resize(1);
    args._outputs.resize(1);

    int outputData = 2;
    args.setArgumentProperties(1, &outputData, ov::Shape{4, 5}, std::vector<size_t>{5, 1});

    ASSERT_EQ(args._outputs[0]._dimsCount, 2);
    EXPECT_EQ(args._outputs[0]._basePtr, &outputData);
    EXPECT_EQ(args._outputs[0]._data, &outputData);
    EXPECT_EQ(args._outputs[0]._sizes, (std::vector<int64_t>{4, 5}));
    EXPECT_EQ(args._outputs[0]._strides, (std::vector<int64_t>{5, 1}));
}

TEST(DynamicArgumentsTest, SetArgumentPropertiesWithMismatchedRankThrows) {
    DynamicArguments args;
    args._inputs.resize(1);

    int data = 3;
    args.setArgumentProperties(0, &data, ov::Shape{2, 3}, std::vector<size_t>{3, 1});

    EXPECT_THROW(args.setArgumentProperties(0, &data, ov::Shape{2, 3, 4}, std::vector<size_t>{12, 4, 1}),
                 ov::Exception);
}

TEST(DynamicArgumentsTest, SetArgumentPropertiesWithMismatchedStrideCountThrows) {
    DynamicArguments args;
    args._inputs.resize(1);

    int data = 5;
    EXPECT_THROW(args.setArgumentProperties(0, &data, ov::Shape{2, 3}, std::vector<size_t>{3}), ov::Exception);
}

TEST(DynamicArgumentsTest, SetArgumentPropertiesUpdatesSameSlotWithSameRank) {
    DynamicArguments args;
    args._inputs.resize(1);

    int dataA = 7;
    args.setArgumentProperties(0, &dataA, ov::Shape{2, 3}, std::vector<size_t>{3, 1});

    int dataB = 9;
    args.setArgumentProperties(0, &dataB, ov::Shape{4, 5}, std::vector<size_t>{5, 1});

    ASSERT_EQ(args._inputs[0]._dimsCount, 2);
    EXPECT_EQ(args._inputs[0]._basePtr, &dataB);
    EXPECT_EQ(args._inputs[0]._data, &dataB);
    EXPECT_EQ(args._inputs[0]._sizes, (std::vector<int64_t>{4, 5}));
    EXPECT_EQ(args._inputs[0]._strides, (std::vector<int64_t>{5, 1}));
}

TEST(DynamicArgumentsTest, SetArgumentPropertiesOutOfRangeDoesNotModifySlots) {
    DynamicArguments args;
    args._inputs.resize(1);
    args._outputs.resize(1);

    args._inputs[0]._dimsCount = 2;
    args._inputs[0]._sizes = {2, 3};
    args._inputs[0]._strides = {3, 1};

    args._outputs[0]._dimsCount = 2;
    args._outputs[0]._sizes = {4, 5};
    args._outputs[0]._strides = {5, 1};

    int data = 10;
    args.setArgumentProperties(100, &data, ov::Shape{8, 9}, std::vector<size_t>{9, 1});

    EXPECT_EQ(args._inputs[0]._sizes, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(args._outputs[0]._sizes, (std::vector<int64_t>{4, 5}));
}

TEST(DynamicArgumentsTest, EnsureExecutionContextCreatesOnlyOnce) {
    RuntimeFunctionGuard guard;
    g_createExecutionContextCalls = 0;

    intel_npu::npuVMRuntimeCreateExecutionContext = fakeCreateExecutionContext;

    DynamicArguments args;
    const auto vmRuntime = reinterpret_cast<npu_vm_runtime_handle_t>(0x5678);

    args.ensureExecutionContext(vmRuntime);
    args.ensureExecutionContext(vmRuntime);

    EXPECT_EQ(g_createExecutionContextCalls, 1);
    EXPECT_NE(args._executionContext, nullptr);
}

TEST(DynamicArgumentsTest, DestructorDestroysExecutionContextWhenCreated) {
    RuntimeFunctionGuard guard;
    g_createExecutionContextCalls = 0;
    g_destroyExecutionContextCalls = 0;
    g_lastDestroyedContext = nullptr;

    intel_npu::npuVMRuntimeCreateExecutionContext = fakeCreateExecutionContext;
    intel_npu::npuVMRuntimeDestroyExecutionContext = fakeDestroyExecutionContext;

    npu_vm_runtime_execution_context_handle_t createdContext = nullptr;
    {
        DynamicArguments args;
        const auto vmRuntime = reinterpret_cast<npu_vm_runtime_handle_t>(0x9abc);
        args.ensureExecutionContext(vmRuntime);
        createdContext = args._executionContext;
    }

    EXPECT_EQ(g_createExecutionContextCalls, 1);
    EXPECT_EQ(g_destroyExecutionContextCalls, 1);
    EXPECT_EQ(g_lastDestroyedContext, createdContext);
}

TEST(DynamicArgumentsTest, EnsureExecutionContextThrowsOnCreationFailure) {
    RuntimeFunctionGuard guard;
    g_createExecutionContextCalls = 0;

    intel_npu::npuVMRuntimeCreateExecutionContext = fakeCreateExecutionContextFail;

    DynamicArguments args;
    const auto vmRuntime = reinterpret_cast<npu_vm_runtime_handle_t>(0x3456);

    EXPECT_THROW(args.ensureExecutionContext(vmRuntime), ov::Exception);
    EXPECT_EQ(g_createExecutionContextCalls, 1);
    EXPECT_EQ(args._executionContext, nullptr);
}

}  // namespace
}  // namespace intel_npu
