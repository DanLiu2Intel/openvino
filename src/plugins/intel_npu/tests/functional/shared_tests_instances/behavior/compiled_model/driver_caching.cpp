#include "overload/driver_caching.hpp"

#include "iostream"

namespace {

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_CompilationCacheFlag,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{{}})),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

INSTANTIATE_TEST_SUITE_P(smoke_CompilationCacheFlag,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                                {ov::intel_npu::bypass_umd_caching(false)},
                                                {ov::cache_dir("path/to/cacheDir")}})),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

#ifdef WIN32
INSTANTIATE_TEST_SUITE_P(smoke_CompilationTwiceOnWindwos,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                                {ov::intel_npu::bypass_umd_caching(false)},
                                                {ov::cache_dir("path/to/cacheDir")}})),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

INSTANTIATE_TEST_SUITE_P(smoke_CompilationTwiceOnWindwos,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{})),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

#else

INSTANTIATE_TEST_SUITE_P(smoke_CompilationTwiceOnLinux,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                                {ov::intel_npu::bypass_umd_caching(false)},
                                                {ov::cache_dir("path/to/cacheDir")}})),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

INSTANTIATE_TEST_SUITE_P(smoke_CompilationTwiceOnLinux,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{})),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

#endif

}  // namespace