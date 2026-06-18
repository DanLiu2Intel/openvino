// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"

namespace intel_npu {
namespace {

TEST(NPUVMRuntimeApiTest, InitializeAcceptsLibraryName) {
    EXPECT_NO_THROW(NPUVMRuntimeApi::initialize("npu_mlir_runtime"));
}

TEST(NPUVMRuntimeApiTest, InitializeAllowsChangingLibraryBeforeInstanceCreation) {
    EXPECT_NO_THROW(NPUVMRuntimeApi::initialize("npu_mlir_runtime"));
    EXPECT_NO_THROW(NPUVMRuntimeApi::initialize("npu_interpreter_runtime"));
}

TEST(NPUVMRuntimeApiTest, InitializeFromBlobAcceptsNPUByteHeader) {
    const char blobHeader[] = {'N', 'P', 'U', 'B', 'y', 't', 'e', '\0', 'x', 'x', 'x', 'x'};
    EXPECT_NO_THROW(NPUVMRuntimeApi::initializeFromBlob(blobHeader, sizeof(blobHeader)));
}

TEST(NPUVMRuntimeApiTest, InitializeFromBlobAcceptsNonNPUByteHeader) {
    const char blobHeader[] = {'l', 'l', 'v', 'm', 'x', 'x', 'x', 'x'};
    EXPECT_NO_THROW(NPUVMRuntimeApi::initializeFromBlob(blobHeader, sizeof(blobHeader)));
}

TEST(NPUVMRuntimeApiTest, InitializeFromBlobAcceptsTinyBlob) {
    const char tinyBlob[] = {'A'};
    EXPECT_NO_THROW(NPUVMRuntimeApi::initializeFromBlob(tinyBlob, sizeof(tinyBlob)));
}

}  // namespace
}  // namespace intel_npu
