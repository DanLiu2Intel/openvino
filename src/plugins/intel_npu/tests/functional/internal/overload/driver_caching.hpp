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

std::string generateCacheDirName(const std::string& test_name) {
    // Generate unique file names based on test name, thread id and timestamp
    // This allows execution of tests in parallel (stress mode)
    auto hash = std::to_string(std::hash<std::string>()(test_name));
    std::stringstream ss;
    auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
    ss << hash << "_"
        << "_" << ts.count();
    return ss.str();
}

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        // The value of cache_encryption_callbacks cannot be converted to std::string
        if (value.first == ov::cache_encryption_callbacks.name()) {
            continue;
        }
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

bool containsCacheStatus(const std::string& str, const std::string cmpstr) {  
    return str.find(cmpstr) != std::string::npos;  
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

    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            std::filesystem::remove_all(entry);
        }
    }
}

//Does this part is need?
inline std::vector<std::string> listFilesWithExt(const std::string& path) {
    struct dirent* ent;
    DIR* dir = opendir(path.c_str());
    std::vector<std::string> res;
    if (dir != nullptr) {
        while ((ent = readdir(dir)) != NULL) {
            auto file = ov::test::utils::makePath(path, std::string(ent->d_name));
            struct stat stat_path;
            stat(file.c_str(), &stat_path);
            //cache not contian file extension.
            if (!S_ISDIR(stat_path.st_mode) && ov::test::utils::endsWith(file, "")) {
                res.push_back(std::move(file));
            }
        }
        closedir(dir);
    }
    return res;
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
        std::printf("<====local test name> %s\n", result.str().c_str());
        return result.str();
    }

    void SetUp() override {
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
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        APIBaseTest::TearDown();
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> initStruct;
};

TEST_P(CompileAndDriverCaching, CompilationCacheFlag) {
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[1]printf testsuit content1 : %s\n", driverLogContent.c_str());//empty, 
    EXPECT_TRUE(containsCacheStatus(driverLogContent, ""));

    ov::CompiledModel execNet;
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("printf testsuit content2 : %s\n", driverLogContent2.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent2, "cache_status_t::found"));

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("printf testsuit content3 : %s\n", driverLogContent3.c_str());
    if ((configuration.find("CACHE_DIR") != configuration.end()) || configuration.find("NPU_BYPASS_UMD_CACHING") != configuration.end()) {
         EXPECT_TRUE(containsCacheStatus(driverLogContent3, ""));
    } else {
         EXPECT_TRUE(containsCacheStatus(driverLogContent3, "cache_status_t::found"));
    }

    //With or without enable UMD caching, the compilation time for the second time should be shorter than the first.
    std::printf("==[1]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
    EXPECT_GT(durationFirst.count(), durationSecond.count());
}//有一个条件失败，整个test case 就是失败的

TEST_P(CompileAndDriverCaching, CompilationCacheWithEmptyConfig) {
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[2]printf testsuit content1 : %s\n", driverLogContent.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent, ""));
    
    ov::CompiledModel execNet;
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[2]printf testsuit content2 : %s\n", driverLogContent2.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent2, "cache_status_t::found"));

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[2]printf testsuit content3 : %s\n", driverLogContent3.c_str());
    if ((configuration.find("CACHE_DIR") != configuration.end()) || configuration.find("NPU_BYPASS_UMD_CACHING") != configuration.end()) {
        EXPECT_TRUE(containsCacheStatus(driverLogContent3, ""));
    } else {
        EXPECT_TRUE(containsCacheStatus(driverLogContent3, "cache_status_t::found"));
    }

    std::printf("==[2]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
    EXPECT_GT(durationFirst.count(), durationSecond.count());
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithOVCacheConfig) {
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[3]printf testsuit content1 : %s\n", driverLogContent.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent, ""));

    configuration[ov::cache_dir.name()] = "testCacheDir";
    ov::CompiledModel execNet;
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[3]printf testsuit content2 : %s\n", driverLogContent2.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent2, "cache_status_t::found"));

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[3]printf testsuit content3 : %s\n", driverLogContent3.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent3, ""));

    std::printf("==[3]testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
    EXPECT_GT(durationFirst.count(), durationSecond.count());
}

TEST_P(CompileAndDriverCaching, CompilationCacheWithBypassConfig) {
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("==[4]printf testsuit content1 : %s\n", driverLogContent.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent, ""));

    configuration[ov::intel_npu::bypass_umd_caching.name()] = true;
    ov::CompiledModel execNet;
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    std::string driverLogContent2 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[3]printf testsuit content2 : %s\n", driverLogContent2.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent2, ""));

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;

    std::string driverLogContent3 = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("[3]printf testsuit content3 : %s\n", driverLogContent3.c_str());
    EXPECT_TRUE(containsCacheStatus(driverLogContent3, ""));

    std::printf("==[4] testsuit time (1): %f, (2): %f\n", durationFirst.count(), durationSecond.count());
    EXPECT_GT(durationFirst.count(), durationSecond.count());
}


}  // namespace behavior
}  // namespace test
}  // namespace ov
