#include <gtest/gtest.h>

#include <common_test_utils/test_assertions.hpp>
#include <sstream>

// #include "shared_test_classes/base/ov_subgraph.hpp"s
#include "base/ov_behavior_test_utils.hpp"

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset8.hpp"
#include <openvino/opsets/opset1.hpp>

#include "stdio.h" //
#include <stdlib.h>// env setting

#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/config/config.hpp"

// #include "openvino/runtime/properties.hpp"  //include by config files
#include "openvino/runtime/intel_npu/properties.hpp"

#include <filesystem>
#include <chrono> // cal time

//mkdir folder
#ifdef WIN32
#include "Shlobj.h"
#include "shlobj_core.h"
#include "objbase.h"
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace ov {
namespace test {
namespace behavior {

typedef std::tuple<std::string,                 // Device name
                   ov::AnyMap                   // Config
                   >
    CompileAndModelCachingParams;

inline std::shared_ptr<ov::Model> createModel1() {
    ResultVector results;
    ParameterVector params;
    auto op = std::make_shared<ov::op::v1::Add>(opset8::Constant::create(ov::element::f16, {1}, {1}),
                                                opset8::Constant::create(ov::element::f16, {1}, {1}));
    op->set_friendly_name("Add");
    auto res = std::make_shared<ov::op::v0::Result>(op);
    res->set_friendly_name("Result");
    res->get_output_tensor(0).set_names({"tensor_output"});
    results.push_back(res);
    return std::make_shared<Model>(results, params);
}

inline std::shared_ptr<ov::Model> createModel2() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add");
    return std::make_shared<ov::Model>(ov::OutputVector{add->output(0)}, ov::ParameterVector{param});
}

inline std::shared_ptr<ov::Model> createModel3() {
    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 6, 224});
    auto weights = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 6, 224}, { 1 });
    auto conv = std::make_shared<ov::opset1::GroupConvolution>(param,
                                        weights,
                                        ov::Strides{1},
                                        ov::CoordinateDiff{0},
                                        ov::CoordinateDiff{0},
                                        ov::Strides{1});
    return std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ param });
}

inline std::shared_ptr<ov::Model> createModel4() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add");
    auto subtract = std::make_shared<ov::op::v1::Subtract>(add, const_value);
    subtract->set_friendly_name("sub");
    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(subtract, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::op::v0::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

inline std::shared_ptr<ov::Model> createModel5() {
        const auto param = std::make_shared<ov::opset8::Parameter>(ov::element::f16, ov::PartialShape{2, 2});
    const auto convert = std::make_shared<ov::opset8::Convert>(param, ov::element::f32);
    const auto result = std::make_shared<ov::opset8::Result>(convert);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "StaticFunction");
}


bool containsCacheStatus(const std::string& str, const std::string cmpstr) {  
    return str.find(cmpstr) != std::string::npos;  
}

void checkCacheDirectory() {
    std::filesystem::path path{};
#ifdef WIN32
    wchar_t* local = nullptr;
    auto result = SHGetKnownFolderPath( FOLDERID_LocalAppData, 0, NULL, &local );

    if(SUCCEEDED(result)) {
        // prepend to enable long path name support
        path = std::filesystem::path( L"\\\\?\\" + std::wstring( local ) + +L"\\Intel\\NPU" );

        CoTaskMemFree( local );
    }
#else
    const char *env = getenv("ZE_INTEL_NPU_CACHE_DIR");
    if (env) {
        path = std::filesystem::path(env);
    } else {
        env = getenv("HOME");
        if (env) {
            path = std::filesystem::path(env) / ".cache/ze_intel_npu_cache";
        } else {
            path = std::filesystem::current_path() / ".cache/ze_intel_npu_cache";
        }
    }
#endif

    std::printf(">>>>check cache psth: #%s#\n", path.c_str());
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            std::printf("  >>>>>content: #%s# \n", entry.path().c_str());
        }
    }
}

class CompileAndDriverCaching : public testing::WithParamInterface<CompileAndModelCachingParams>,
                                public OVPluginTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileAndModelCachingParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::printf("--------SetUp--------\n");
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        initStruct = std::make_shared<::intel_npu::ZeroInitStructsHolder>();
        if (!initStruct) {
            GTEST_SKIP() << "ZeroInitStructsHolder init failed, ZeroInitStructsHolder is a nullptr";
        }
        APIBaseTest::SetUp();
    }

    void TearDown() override {
        std::printf("--------TearDown--------\n");
        if (!m_cachedir.empty()) {
            std::printf("            printf m_cachedir:#%s# \n", m_cachedir.c_str());
            core->set_property({ov::cache_dir()});
            core.reset();
            ov::test::utils::removeFilesWithExt(m_cachedir, "blob");
            ov::test::utils::removeDir(m_cachedir);
        }
        if(core) {
            std::printf("  core is not empty\n");
        }
        ov::test::utils::PluginCache::get().reset();
        if(core) {
            std::printf("  core is not empty\n");
        }

        APIBaseTest::TearDown();
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct;
    std::string m_cachedir;
};

TEST_P(CompileAndDriverCaching, CompilationCacheWithEmptyConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();
    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[1.1][EmptyConfig] driver log content1 : #%s#\n", driverLogContent.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent, ""));

    ov::CompiledModel execNet;
    function = createModel1();

    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);

    std::printf("==[1.2][EmptyConfig] driver log content2 : #%s#\n", driverLogContent2.c_str());
#ifdef WIN32
    EXPECT_TRUE(containsCacheStatus(driverLogContent2, "cache_status_t::found"));
#else
    EXPECT_TRUE(containsCacheStatus(driverLogContent2, "cache_status_t::stored"));
#endif

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[1.3][EmptyConfig] driver log content3 : #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent3, "cache_status_t::found"));

    std::printf("==[1.4][EmptyConfig] time: (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithOVCacheConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.1][OVCacheConfig] driver log content1 : #%s#\n", driverLogContent.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent, "cache_status_t::found")));

    configuration[ov::cache_dir.name()] = "./testCacheDir";
    m_cachedir = configuration[ov::cache_dir.name()].as<std::string>();

    ov::CompiledModel execNet;
    function = createModel2();

    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.2][OVCacheConfig] driver log content2 : #%s#\n", driverLogContent2.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent2, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent2, "cache_status_t::found")));

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.2][OVCacheConfig] driver log content3 : #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent3, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent3, "cache_status_t::found")));

    std::printf("==[2.4][EmptyConfig] time: (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithBypassConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3.1][BypassCacheConfig] driver log content1 : #%s#\n", driverLogContent.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent, "cache_status_t::found")));

    configuration[ov::intel_npu::bypass_umd_caching.name()] = true;
    ov::CompiledModel execNet;
    function = createModel3();

    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3.2][BypassCacheConfig] driver log content2 : #%s#\n", driverLogContent2.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent2, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent2, "cache_status_t::found")));

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3.3][BypassCacheConfig] driver log content3 : #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent3, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent3, "cache_status_t::found")));

    std::printf("==[3.4][EmptyConfig] time: (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
}

//test class2
class CompileAndDriverCachingOVcacheDIR : public testing::WithParamInterface<CompileAndModelCachingParams>,
                                public OVPluginTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileAndModelCachingParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            // using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void SetUp() override {
        std::printf("----setup2----\n");
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        initStruct = std::make_shared<::intel_npu::ZeroInitStructsHolder>();
        if (!initStruct) {
            GTEST_SKIP() << "ZeroInitStructsHolder init failed, ZeroInitStructsHolder is a nullptr";
        }
        APIBaseTest::SetUp();
    }

    void TearDown() override {
        std::printf("----teardown2----\n");
        if (!m_cachedir.empty()) {
            std::printf("            printf m_cachedir2:#%s# \n", m_cachedir.c_str());
            core->set_property({ov::cache_dir()});
            core.reset();
            ov::test::utils::removeFilesWithExt(m_cachedir, "blob");
            ov::test::utils::removeDir(m_cachedir);
        }
        if(core) {
            std::printf("  core is not empty2\n");
        }
        ov::test::utils::PluginCache::get().reset();
        if(core) {
            std::printf("  core is not empty2\n");
        }

        APIBaseTest::TearDown();
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct;
    std::string m_cachedir;
};

TEST_P(CompileAndDriverCachingOVcacheDIR, CompilationCacheWithBypassConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[]OVcacheDIR[]][Class ovCache] driver log content1 #%s#\n", driverLogContent.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent, "cache_status_t::found")));

    configuration[ov::cache_dir.name()] = "./testCacheDir";
    m_cachedir = configuration[ov::cache_dir.name()].as<std::string>();
    ov::CompiledModel execNet;
    function = createModel4();

    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[]OVcacheDIR[]][Class ovCache] driver log content2 #%s#\n", driverLogContent2.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent2, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent2, "cache_status_t::found")));

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[]OVcacheDIR[]][Class ovCache] driver log content3 #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent3, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent3, "cache_status_t::found")));

    std::printf("==[]OVcacheDIR[]][Class ovCache] time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
}

//test class3
class CompileAndDriverCachingBypass : public testing::WithParamInterface<CompileAndModelCachingParams>,
                                public OVPluginTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileAndModelCachingParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void SetUp() override {
        std::printf("----setup3-----\n");
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        initStruct = std::make_shared<::intel_npu::ZeroInitStructsHolder>();
        if (!initStruct) {
            GTEST_SKIP() << "ZeroInitStructsHolder init failed, ZeroInitStructsHolder is a nullptr";
        }
        APIBaseTest::SetUp();
    }

    void TearDown() override {
        std::printf("----teardown3----\n");
        if (!m_cachedir.empty()) {
            std::printf("            printf m_cachedir3:#%s# \n", m_cachedir.c_str());
            core->set_property({ov::cache_dir()});
            core.reset();
            ov::test::utils::removeFilesWithExt(m_cachedir, "blob");
            ov::test::utils::removeDir(m_cachedir);
        }
        if(core) {
            std::printf("  core is not empty3\n");
        }
        ov::test::utils::PluginCache::get().reset();
        if(core) {
            std::printf("  core is not empty3\n");
        }

        APIBaseTest::TearDown();
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct;
    std::string m_cachedir;
};



TEST_P(CompileAndDriverCachingBypass, CompilationCacheWithBypassConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();
    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[1]BypassConfig[]][Class BypassConfig] driver log content #%s#\n", driverLogContent.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent, "cache_status_t::found")));

    configuration[ov::intel_npu::bypass_umd_caching.name()] = true;
    ov::CompiledModel execNet;
    function = createModel5();

    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2]BypassConfig[]][Class BypassConfig] driver log content #%s#\n", driverLogContent2.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent2, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent2, "cache_status_t::found")));

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3]BypassConfig[]][Class BypassConfig] driver log content #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent3, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent3, "cache_status_t::found")));

    std::printf("==[4]BypassConfig[]][Class BypassConfig] (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
