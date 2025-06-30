// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compile_and_infer_different_element_type.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

/**
 * Reads configuration environment variables
 */
class VpuTestEnvConfig {
public:
    std::string IE_NPU_TESTS_DEVICE_NAME;
    std::string IE_NPU_TESTS_DUMP_PATH;
    std::string IE_NPU_TESTS_LOG_LEVEL;
    std::string IE_NPU_TESTS_PLATFORM;
    std::string OV_NPU_TESTS_SKIP_CONFIG_FILE = "npu_skip_func_tests.xml";

    bool IE_NPU_TESTS_RUN_COMPILER = true;
    bool IE_NPU_TESTS_RUN_EXPORT = false;
    bool IE_NPU_TESTS_RUN_IMPORT = false;
    bool IE_NPU_TESTS_RUN_INFER = true;
    bool IE_NPU_TESTS_EXPORT_INPUT = false;
    bool IE_NPU_TESTS_EXPORT_OUTPUT = false;
    bool IE_NPU_TESTS_EXPORT_REF = false;
    bool IE_NPU_TESTS_IMPORT_INPUT = false;
    bool IE_NPU_TESTS_IMPORT_REF = false;
    bool IE_NPU_SINGLE_CLUSTER_MODE = false;

    bool IE_NPU_TESTS_RAW_EXPORT = false;
    bool IE_NPU_TESTS_LONG_FILE_NAME = false;

public:
    static const VpuTestEnvConfig& getInstance();

private:
    explicit VpuTestEnvConfig();
};

std::string getTestsPlatformFromEnvironmentOr(const std::string& instead) {
    return (!VpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM.empty())
                   ? VpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM
                   : instead;
}

std::string getTestsPlatformCompilerInPlugin() {
    return getTestsPlatformFromEnvironmentOr(
            getTestsDeviceNameFromEnvironmentOr(std::string(ov::intel_npu::Platform::AUTO_DETECT)));
}

// auto configs = []() {
//     return std::vector<ov::AnyMap>{{{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
//                                      ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin()),
//                                      {"NPU_COMPILATION_MODE", "DefaultHW"}}},
//                                    {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
//                                      ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin()),
//                                      {"NPU_COMPILATION_MODE", "DefaultHW"}}}};
// };


/// input param 
// using OVInferRequestDynamicParams = std::tuple<
//         std::shared_ptr<Model>,                                         // ov Model
//         std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>,  // input/expected output shapes per inference
//         std::string,                                                       // Device name
//         ov::AnyMap                                                  // Config
// >;


auto configs = []() {
    return std::vector<ov::AnyMap>{{{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
                                     ov::intel_npu::platform(ov::intel_npu::Platform::NPU3720),
                                     {"NPU_COMPILATION_MODE", "DefaultHW"}}},
                                   {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                                     ov::intel_npu::platform(ov::intel_npu::Platform::NPU3720),
                                     {"NPU_COMPILATION_MODE", "DefaultHW"}}}};
};

auto configs2 = []() {
    return std::vector<ov::AnyMap>{{{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
                                     ov::intel_npu::platform(ov::intel_npu::Platform::NPU4000),
                                     {"NPU_COMPILATION_MODE", "DefaultHW"}}},
                                   {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                                     ov::intel_npu::platform(ov::intel_npu::Platform::NPU4000),
                                     {"NPU_COMPILATION_MODE", "DefaultHW"}}}};
};

INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTests, NPUInferRequestElementTypeTests,
        ::testing::Combine(::testing::Values(getFunction()),
                           ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                   {{1, 1, 128}, {1, 1, 128}}, {{128}, {128}}}),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::ValuesIn(configs())),
        ov::test::utils::appendPlatformTypeTestName<OVInferRequestDynamicTests>);

INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTests, NPUInferRequestElementTypeTests,
        ::testing::Combine(::testing::Values(getFunction()),
                           ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                   {{1, 1, 128}, {1, 1, 128}}, {{128}, {128}}}),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::ValuesIn(configs2())),
        ov::test::utils::appendPlatformTypeTestName<OVInferRequestDynamicTests>);