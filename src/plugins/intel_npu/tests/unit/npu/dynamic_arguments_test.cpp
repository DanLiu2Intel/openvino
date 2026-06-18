// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_npu/utils/vm/dynamic_arguments.hpp"
#include "zero_dynamic_pipeline.hpp"

namespace intel_npu {
namespace {

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

}  // namespace
}  // namespace intel_npu
