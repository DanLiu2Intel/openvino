// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    bool,
    bool> TransposeAfterMatMulTransformationParams;

class TransposeAfterMatMulTransformation :
    public testing::WithParamInterface<TransposeAfterMatMulTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeAfterMatMulTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
