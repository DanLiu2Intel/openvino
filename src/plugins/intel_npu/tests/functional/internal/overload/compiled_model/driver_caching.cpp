#include "overload/driver_caching.hpp"

#include "iostream"

namespace {

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_firstCompilation,
                         CompileAndModelCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                                {ov::intel_npu::bypass_umd_caching(false)},
                                                {ov::intel_npu::bypass_umd_caching(true)}})),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndModelCaching>);

INSTANTIATE_TEST_SUITE_P(smoke_secondCompilation,
                         CompileAndModelCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                                {ov::intel_npu::bypass_umd_caching(false)},
                                                {ov::intel_npu::bypass_umd_caching(true)}})),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndModelCaching>);

INSTANTIATE_TEST_SUITE_P(smoke_secondCompilationBypassCache,
                         CompileAndModelCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                                {ov::intel_npu::bypass_umd_caching(false)}})),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndModelCaching>);


INSTANTIATE_TEST_SUITE_P(smoke_secondCompilationBypassCache,
                         CompileAndModelCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                                {ov::intel_npu::bypass_umd_caching(true)}})),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndModelCaching>);

}  // namespace