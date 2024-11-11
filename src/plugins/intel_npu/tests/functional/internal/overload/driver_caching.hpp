#include <gtest/gtest.h>

#include <common_test_utils/test_assertions.hpp>
#include <sstream>

// #include "shared_test_classes/base/ov_subgraph.hpp"s
#include "base/ov_behavior_test_utils.hpp"

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset8.hpp"

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

typedef std::tuple<std::shared_ptr<ov::Model>,  // Model
                   std::string,                 // Device name
                   ov::AnyMap                   // Config
                   >
    CompileAndModelCachingParams;

inline std::shared_ptr<ov::Model> getConstantGraph() {
    ResultVector results;
    ParameterVector params;
    auto op = std::make_shared<ov::op::v1::Add>(opset8::Constant::create(ov::element::f32, {1}, {1}),
                                                opset8::Constant::create(ov::element::f32, {1}, {1}));
    op->set_friendly_name("Add");
    auto res = std::make_shared<ov::op::v0::Result>(op);
    res->set_friendly_name("Result");
    res->get_output_tensor(0).set_names({"tensor_output"});
    results.push_back(res);
    return std::make_shared<Model>(results, params);
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


void checkSystemCacheDirectory() {
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

    std::printf(">>>>check cache content1:\n");
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            std::printf("  >>>>>content: #%s# \n", entry.path().c_str());
        }
    }

    std::printf(">>>>remove cache content:\n");
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            std::filesystem::remove_all(entry.path());
            std::printf("  >>>>>remove cache: #%s# \n", entry.path().c_str());
        }
    }

    std::printf(">>>>check cache content2:\n");
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            std::printf("  >>>>> contain : #%s# \n", entry.path().c_str());
        }
    }
}

class CompileAndDriverCaching : public testing::WithParamInterface<CompileAndModelCachingParams>,
                                public OVPluginTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileAndModelCachingParams> obj) {
        std::shared_ptr<ov::Model> model;
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(model, targetDevice, configuration) = obj.param;
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
        std::printf("<====local test name> #%s#\n", result.str().c_str());
        return result.str();
    }

    void SetUp() override {
        std::printf(" will call how much time setup\n");
        std::tie(function, target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        initStruct = std::make_shared<::intel_npu::ZeroInitStructsHolder>();
        if (!initStruct) {
            GTEST_SKIP() << "ZeroInitStructsHolder init failed, ZeroInitStructsHolder is a nullptr";
        }
        APIBaseTest::SetUp();
        
        //remove system cache. contain and remove ? to prevent contine build and make failed.
        checkSystemCacheDirectory();
    }

    void TearDown() override {
        if (!m_cachedir.empty()) {
            std::printf("            printf m_cachedir:#%s# \n", m_cachedir.c_str());
            core->set_property({ov::cache_dir()});
            core.reset();
            ov::test::utils::removeFilesWithExt(m_cachedir, "blob");
            ov::test::utils::removeDir(m_cachedir);
        }
        if(core) {
            std::printf("  core is not empty\n");
        } else {
            std::printf("  core is empty\n");
        }
        ov::test::utils::PluginCache::get().reset();
        if(core) {
            std::printf("  core is not empty\n");
        } else {
            std::printf("  core is empty\n");
        }
        std::printf(">>>how much teardown will be call? TearDown>>>\n");
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        checkSystemCacheDirectory();
        APIBaseTest::TearDown();
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct;
    std::string m_cachedir;
};

// TEST_P(CompileAndDriverCaching, CompilationCacheFlag) {
//     checkCacheDirectory();
//     ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

//     std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
//     std::printf("==[1.1]printf testsuit content1 : #%s#\n", driverLogContent.c_str());//empty, 
//     EXPECT_TRUE(containsCacheStatus(driverLogContent, ""));

//     std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct2 = std::make_shared<::intel_npu::ZeroInitStructsHolder>();
//     ze_graph_dditable_ext_decorator& graph_ddi_table_ext2 = initStruct2->getGraphDdiTable();
//     std::string driverLogContent1_2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
//     std::printf("==[1.1-2]printf testsuit content1_2 : #%s#\n", driverLogContent1_2.c_str()); // need to check

//     ov::CompiledModel execNet;
//     //first run time will long and will generate the model cache.
//     auto startFirst = std::chrono::high_resolution_clock::now(); 
//     OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
//     auto endFirst = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> durationFirst = endFirst - startFirst;

//     std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
//     std::printf("[1.2]printf testsuit content2 : #%s#\n", driverLogContent2.c_str());
// #ifdef WIN32
//     EXPECT_TRUE(containsCacheStatus(driverLogContent2, "cache_status_t::found"));
// #else
//     EXPECT_TRUE(containsCacheStatus(driverLogContent2, "cache_status_t::stored"));
// #endif

//     //second time compilation
//     auto startSecond = std::chrono::high_resolution_clock::now(); 
//     OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
//     auto endSecond = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> durationSecond = endSecond - startSecond;

//     std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
//     std::printf("[1.3]printf testsuit content3 : #%s#\n", driverLogContent3.c_str());
//     if ((configuration.find("CACHE_DIR") != configuration.end()) || configuration.find("NPU_BYPASS_UMD_CACHING") != configuration.end()) {
//         if (configuration.find("CACHE_DIR") != configuration.end()) {
//             m_cachedir = configuration.at(ov::cache_dir.name()).as<std::string>();
//         }
//         EXPECT_TRUE(containsCacheStatus(driverLogContent3, ""));
//     } else {
//         EXPECT_TRUE(containsCacheStatus(driverLogContent3, "cache_status_t::found"));
//     }

//     //With or without enable UMD caching, the compilation time for the second time should be shorter than the first.
//     std::printf("==[1.4]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
//     // EXPECT_GT(durationFirst.count(), durationSecond.count());
// }

TEST_P(CompileAndDriverCaching, CompilationCacheWithEmptyConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2.1]printf testsuit content1 : ##%s##\n", driverLogContent.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent, ""));

    ov::CompiledModel execNet;
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[2.2]first compile testsuit content2 : ##%s##\n", driverLogContent2.c_str());
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
    std::printf("[2.3]second compile testsuit content3 : ##%s##\n", driverLogContent3.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent3, "cache_status_t::found"));

    std::printf("==[2.4]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
    // EXPECT_GT(durationFirst.count(), durationSecond.count());
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithOVCacheConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3.1]printf testsuit content1 : #%s#\n", driverLogContent.c_str());
    // EXPECT_TRUE(containsCacheStatus(driverLogContent, "")); //this part should be no found and stored
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent, "cache_status_t::found")));


    configuration[ov::cache_dir.name()] = "testCacheDir";
    m_cachedir = configuration[ov::cache_dir.name()].as<std::string>();

    ov::CompiledModel execNet;
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[3.2]first compile testsuit content2 : #%s#\n", driverLogContent2.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent2, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent2, "cache_status_t::found")));


    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[3.3]second compile testsuit content3 : #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent3, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent3, "cache_status_t::found")));


    std::printf("==[3.4]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());

    std::shared_ptr<ov::Core> core2 = utils::PluginCache::get().core();
    ov::CompiledModel execNet2;
    auto startThird = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet2 = core2->compile_model(function, target_device, configuration));
    auto endThird = std::chrono::high_resolution_clock::now();

    std::string driverLogContent4 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[3.5]second compile testsuit content4 : ##%s##,   time:%f\n", driverLogContent4.c_str(), (endThird - startThird).count());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent4, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent4, "cache_status_t::found")));
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithBypassConfig) {
    checkCacheDirectory();
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[4.1]printf testsuit content1 : #%s#\n", driverLogContent.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent, "cache_status_t::found")));

    configuration[ov::intel_npu::bypass_umd_caching.name()] = true;
    ov::CompiledModel execNet;
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[4.2]first compile testsuit content2 : #%s#\n", driverLogContent2.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent2, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent2, "cache_status_t::found")));

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[4.3]second compile testsuit content3 : #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent3, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent3, "cache_status_t::found")));

    std::printf("==[4.4] testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());

    std::shared_ptr<ov::Core> core2 = utils::PluginCache::get().core();
    ov::CompiledModel execNet2;
    auto startThird = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet2 = core2->compile_model(function, target_device, configuration));
    auto endThird = std::chrono::high_resolution_clock::now();

    std::string driverLogContent4 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[4.5]second compile testsuit content4 : ##%s##,   time:%f\n", driverLogContent4.c_str(), (endThird - startThird).count());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent4, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent4, "cache_status_t::found")));
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithBypassConfig2NewOVCore) {
    checkCacheDirectory();
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct2 = std::make_shared<::intel_npu::ZeroInitStructsHolder>()
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct2->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[5.1]printf testsuit content1 : #%s#\n", driverLogContent.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent, "cache_status_t::found")));

    configuration[ov::intel_npu::bypass_umd_caching.name()] = true;
    ov::CompiledModel execNet;
    ov::Core core;
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core.compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;


    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct3 = std::make_shared<::intel_npu::ZeroInitStructsHolder>()
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext3 = initStruct3->getGraphDdiTable();
    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext3);
    std::printf("[5.2]first compile testsuit content2 : #%s#\n", driverLogContent2.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent2, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent2, "cache_status_t::found")));

    //second time compilation
    ov::Core core2;
    ov::CompiledModel execNet2;
    auto startSecond = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet2 = core2.compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct4 = std::make_shared<::intel_npu::ZeroInitStructsHolder>()
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext4 = initStruct4->getGraphDdiTable();
    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext4);
    std::printf("[5.3]second compile testsuit content3 : #%s#\n", driverLogContent3.c_str());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent3, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent3, "cache_status_t::found")));

    std::printf("==[5.4] testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());

    std::shared_ptr<ov::Core> core3 = utils::PluginCache::get().core();
    ov::CompiledModel execNet3;
    auto startThird = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet3 = core3->compile_model(function, target_device, configuration));
    auto endThird = std::chrono::high_resolution_clock::now();

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct5 = std::make_shared<::intel_npu::ZeroInitStructsHolder>()
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext5 = initStruct5->getGraphDdiTable();
    std::string driverLogContent4 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext5);
    std::printf("[5.5]second compile testsuit content4 : ##%s##,   time:%f\n", driverLogContent4.c_str(), (endThird - startThird).count());
    EXPECT_TRUE( (!containsCacheStatus(driverLogContent4, "cache_status_t::found")) && (!containsCacheStatus(driverLogContent4, "cache_status_t::found")));
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
