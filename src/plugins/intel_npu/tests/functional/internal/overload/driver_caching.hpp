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
    std::printf("------------(-1..2)------------\n");
    auto now = std::chrono::system_clock::now();
    std::printf("------------(0..2)------------\n");
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    std::printf("=========print now timestamp #%ld#\n", timestamp);
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    std::printf("------------(1..2)------------\n");
    param->set_friendly_name("input");
    std::printf("------------(2..2)------------\n");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    std::printf("------------(3..2)------------\n");
    const_value->set_friendly_name("const_val");
    std::printf("------------(4..2)------------\n");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    std::printf("------------(5..2)------------\n");
    add->set_friendly_name("add" + std::to_string(timestamp));
    std::printf("------------(6..2)------------\n");
    return std::make_shared<ov::Model>(ov::OutputVector{add->output(0)}, ov::ParameterVector{param});
}

inline std::shared_ptr<ov::Model> createModel2() {
    std::printf("------------{1}------------\n");
    auto now = std::chrono::system_clock::now();
    std::printf("------------{2}------------\n");
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    std::printf("=========print now timestamp #%ld#\n", timestamp);
    auto data1 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2, 2});
    std::printf("------------{3}------------\n");
    data1->set_friendly_name("input1" + std::to_string(timestamp));
    std::printf("------------{4}------------\n");
    data1->get_output_tensor(0).set_names({"tensor_input1"});
    std::printf("------------{5}------------\n");
    auto op = std::make_shared<op::v0::Relu>(data1);
    std::printf("------------{6}------------\n");
    op->set_friendly_name("Relu");
    std::printf("------------{7}------------\n");
    op->get_output_tensor(0).set_names({"tensor_Relu"});
    std::printf("------------{8}------------\n");
    auto res = std::make_shared<op::v0::Result>(op);
    std::printf("------------{9}------------\n");
    res->set_friendly_name("Result1" + std::to_string(timestamp));
    std::printf("------------{10}------------\n");
    res->get_output_tensor(0).set_names({"tensor_output1"});
    std::printf("------------{11}------------\n");
    return std::make_shared<Model>(ResultVector{res}, ParameterVector{data1});
}

// inline std::shared_ptr<ov::Model> createModel3() {
//     auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
//     param->set_friendly_name("input");
//     auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
//     const_value->set_friendly_name("const_val");
//     auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
//     add->set_friendly_name("add");
//     auto subtract = std::make_shared<ov::op::v1::Subtract>(add, const_value);
//     subtract->set_friendly_name("sub");
//     auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
//     reshape_val->set_friendly_name("reshape_val");
//     auto reshape = std::make_shared<ov::op::v1::Reshape>(subtract, reshape_val, true);
//     reshape->set_friendly_name("reshape");
//     auto result = std::make_shared<ov::op::v0::Result>(reshape);
//     result->set_friendly_name("res");
//     return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
// }

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

    std::printf(">>>>check cache path: #%s#\n", path.c_str());
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            std::printf("  >>>>> cache content: #%s# \n", entry.path().c_str());
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
        std::printf("## Identification code-4linux\n");
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        initStruct = std::make_shared<::intel_npu::ZeroInitStructsHolder>();
        if (!initStruct) {
            GTEST_SKIP() << "ZeroInitStructsHolder init failed, ZeroInitStructsHolder is a nullptr";
        }
        APIBaseTest::SetUp();
    }

    void TearDown() override {
        if (!m_cachedir.empty()) {
            ov::test::utils::removeFilesWithExt(m_cachedir, "blob");
            ov::test::utils::removeDir(m_cachedir);
        }
        ov::test::utils::PluginCache::get().reset();
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
    std::printf("==[1.1][EmptyConfig] driver log content : #%s#\n", driverLogContent.c_str());

    ov::CompiledModel execNet;
    std::printf("------------[1]-------------\n");
    function = createModel1();
    std::printf("-----------[2]--------------\n");

    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[1.2][EmptyConfig] driver log content2 : #%s#\n", driverLogContent2.c_str());
    //To avoid problems with repeatedly calling functiontest
    EXPECT_TRUE(containsCacheStatus(driverLogContent2, "cache_status_t::stored") || containsCacheStatus(driverLogContent2, "cache_status_t::found"));
    checkCacheDirectory();

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[1.3][EmptyConfig] driver log content3 : #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent3, "cache_status_t::stored") || containsCacheStatus(driverLogContent3, "cache_status_t::found"));
    std::printf("==[1.4][EmptyConfig] time: (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());

    
    //final check log and model
    std::printf("----------------------------------------\n");
    std::printf("------------[3]-------------\n");
    checkCacheDirectory();
    std::shared_ptr<ov::Model> function2 = createModel2();
    ov::CompiledModel execNet2;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function2, target_device, configuration));

    checkCacheDirectory();
    std::printf("------------[4]-------------\n");
    std::shared_ptr<ov::Model> function3 = createModel2();
    ov::CompiledModel execNet3;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function3, target_device, configuration));
    checkCacheDirectory();
    std::printf("------------[5]-------------\n");
    std::printf("----------------------------------------\n");
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithOVCacheConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    //Check the initial state if this testp is called separately
    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.1][OVCacheConfig] driver log content : #%s#\n", driverLogContent.c_str());

    configuration[ov::cache_dir.name()] = "./testCacheDir";
    m_cachedir = configuration[ov::cache_dir.name()].as<std::string>();
    ov::CompiledModel execNet;
    function = createModel1();

    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.2][OVCacheConfig] driver log content2 : #%s#\n", driverLogContent2.c_str());
    EXPECT_TRUE(driverLogContent2 == driverLogContent);

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.3][OVCacheConfig] driver log content3 : #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE(driverLogContent3 == driverLogContent);
    std::printf("==[2.4][ovcacheConfig] time: (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithBypassConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();
    //Check the initial state if this testp is called separately
    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3.1][bypassConfig] driver log content1 : #%s#\n", driverLogContent.c_str());

    configuration[ov::intel_npu::bypass_umd_caching.name()] = true;
    ov::CompiledModel execNet;
    function = createModel1();

    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3.2][bypassConfig] driver log content2 : #%s#\n", driverLogContent2.c_str());

    EXPECT_TRUE(driverLogContent2 == driverLogContent);


    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3.3][bypassConfig] driver log content3 : #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE(driverLogContent3 == driverLogContent);

    std::printf("==[3.4][bypassConfig] time: (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
}
//todo: How to check the content of each testp compiled separately for the first time?

}  // namespace behavior
}  // namespace test
}  // namespace ov
