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

bool containsCacheStatus(const std::string& str) {  
    return str.find("cache_status_t::stored") != std::string::npos;  
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
    }

    void TearDown() override {
        if (!m_cache_dir.empty() && !std::filesystem::exists(m_cache_dir)) {
            std::filesystem::remove_all(m_cache_dir);
            //ov::test::utils::removeDir(m_cache_dir);
        }

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
    std::string m_cache_dir;
};

TEST_P(CompileAndDriverCaching, CompilationCacheFlag) {
    ze_graph_dditable_ext_decorator& graph_ddi_table_ext = initStruct->getGraphDdiTable();
    uint32_t graphDdiExtVersion = graph_ddi_table_ext.version();
    
    // check driver version, if less than 1.5 will not support cache feature.
    if (graphDdiExtVersion < ZE_GRAPH_EXT_VERSION_1_5) {
        GTEST_SKIP() << "Skipping test for Driver version less than 1.5, current driver version: " << graphDdiExtVersion;
    }

    std::string driverLogContent = ::intel_npu::zeroUtils::getLatestBuildError(graph_ddi_table_ext);
    std::printf("printf testsuit content : %s\n", driverLogContent.c_str());
    if ( driverLogContent.find( "::stored" ) != std::string::npos ) {
        std::printf("printf testsuit contain stored");
    }
    
    if ( driverLogContent.find( "::found" ) != std::string::npos ) {
        std::printf("printf testsuit contain found");
    }
    //Note: now this part should be successful on Windows.
    EXPECT_TRUE(containsCacheStatus(driverLogContent));
}

#ifdef WIN32
TEST_P(CompileAndDriverCaching, CompilationTwiceOnWindwos) {
    //windows cache dir located on C:\Users\account\AppData\Local\Intel\NPU
    // attempt to get/create root folder in AppData\Local
    std::filesystem::path path{};
    wchar_t* local = nullptr;
    auto result = SHGetKnownFolderPath( FOLDERID_LocalAppData, 0, NULL, &local );

    if( SUCCEEDED( result ) )
    {
        // prepend to enable long path name support
        path = std::filesystem::path( L"\\\\?\\" + std::wstring( local ) + +L"\\Intel\\NPU" );

        CoTaskMemFree( local );

        if( !std::filesystem::exists(path) )
        {
            std::printf(" create cache folder");
            std::filesystem::create_directories(path);
        } else {
            std::printf(" remove cache folder");
            std::filesystem::remove_all(path);
        }
    }
    size_t blobCountInitial = -1;
    blobCountInitial = listFilesWithExt(path.string()).size();
    size_t blobCountAfterwards = -1;
    std::printf("win-1: blobCountInitial=%zu, blobCountAfterwards=%zu\n", blobCountInitial, blobCountAfterwards);
    ASSERT_GT(blobCountInitial, 0);

    ov::CompiledModel execNet;
    //first run time will long and will generate the model cache.
    auto startFirst = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    blobCountAfterwards = listFilesWithExt(path.string()).size();
    std::printf("win-2: blobCountInitial=%zu, blobCountAfterwards=%zu\n", blobCountInitial, blobCountAfterwards);
    if ((configuration.find("CACHE_DIR") != configuration.end()) || configuration.find("NPU_BYPASS_UMD_CACHING") != configuration.end()) {
        ASSERT_GT(blobCountInitial, blobCountAfterwards);
    } else {
        ASSERT_EQ(blobCountInitial, blobCountAfterwards - 1);
    }

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;
    std::printf("win-3:(time) durationFirst=%f, durationSecond=%f\n", durationFirst.count(), durationSecond.count());

    double epsilon = 20.0;
    if ((configuration.find("CACHE_DIR") != configuration.end()) || configuration.find("NPU_BYPASS_UMD_CACHING") != configuration.end()) {
        EXPECT_NEAR(durationFirst.count(), durationSecond.count(), epsilon);
    } else {
        EXPECT_NEAR(durationFirst.count(), durationSecond.count(), durationFirst.count() / 2.0);
    }

    std::filesystem::remove_all(path);
}

#else

TEST_P(CompileAndDriverCaching, CompilationTwiceOnLinux) {
    //ON linux, cache dir can be set by env variables.
    m_cache_dir = generateCacheDirName(GetTestName());
    auto temp = setenv("ZE_INTEL_NPU_CACHE_DIR", m_cache_dir.c_str(), 1);
    //how to creat folder
    int isCreate = mkdir(m_cache_dir.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    if( !isCreate )
        printf("==>create path:%s\n", m_cache_dir.c_str());
    else
        printf("==>create path failed! error code : %s \n",isCreate, m_cache_dir.c_str());

    size_t blobCountInitial = -1;
    blobCountInitial = listFilesWithExt(m_cache_dir).size();
    size_t blobCountAfterwards = -1;
    ASSERT_GT(blobCountInitial, 0);
    std::printf("win-1: blobCountInitial=%zu, blobCountAfterwards=%zu\n", blobCountInitial, blobCountAfterwards);

    //first run time is longer than second time and will generate the model cache.
    ov::CompiledModel execNet;
    auto startFirst = std::chrono::high_resolution_clock::now();
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endFirst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFirst = endFirst - startFirst;

    blobCountAfterwards = listFilesWithExt(m_cache_dir).size();
    std::printf("win-2: blobCountInitial=%zu, blobCountAfterwards=%zu\n", blobCountInitial, blobCountAfterwards);
    if ((configuration.find("CACHE_DIR") != configuration.end()) || configuration.find("NPU_BYPASS_UMD_CACHING") != configuration.end())  {
        ASSERT_GT(blobCountInitial, blobCountAfterwards);
    } else {
        ASSERT_EQ(blobCountInitial, blobCountAfterwards - 1);
    }

    //second time compilation
    auto startSecond = std::chrono::high_resolution_clock::now(); 
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    auto endSecond = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSecond = endSecond - startSecond;
    std::printf("lin-3:(time) durationFirst=%f, durationSecond=%f\n", durationFirst.count(), durationSecond.count());
    double epsilon = 20.0;
    if ((configuration.find("CACHE_DIR") != configuration.end()) || configuration.find("NPU_BYPASS_UMD_CACHING") != configuration.end()) {
        EXPECT_NEAR(durationFirst.count(), durationSecond.count(), epsilon);
    } else {
        EXPECT_NEAR(durationFirst.count(), durationSecond.count(), durationFirst.count() / 2.0);
    }
}
#endif


}  // namespace behavior
}  // namespace test
}  // namespace ov
