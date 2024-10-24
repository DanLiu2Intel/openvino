#include <gtest/gtest.h>

#include <algorithm>
#include <common_test_utils/test_assertions.hpp>
#include <sstream>

#include "base/ov_behavior_test_utils.hpp"
#include "intel_npu/al/config/common.hpp"
#include "npu_private_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/properties.hpp"
#include "zero_init.hpp"

#include "stdio.h" //

#include<stdlib.h>// env setting

namespace ov {
namespace test {
namespace behavior {

typedef std::tuple<std::shared_ptr<ov::Model>,  // Model xml
                   std::shared_ptr<ov::Model>,  // Model bin
                   std::string,                 // Device name
                   ov::AnyMap                   // Config
                   >
    CompileAndModelCachingParams1;

typedef std::tuple<std::shared_ptr<ov::Model>,  // Model
                   std::string,                 // Device name
                   ov::AnyMap                   // Config
                   >
    CompileAndModelCachingParams2;

//how can i get the model?

// how can i get the generated blob？

inline std::shared_ptr<ov::Model> getConstantGraph(element::Type type) {
    ResultVector results;
    ParameterVector params;
    auto op = std::make_shared<ov::op::v1::Add>(opset8::Constant::create(type, {1}, {1}),
                                                opset8::Constant::create(type, {1}, {1}));
    op->set_friendly_name("Add");
    auto res = std::make_shared<ov::op::v0::Result>(op);
    res->set_friendly_name("Result");
    res->get_output_tensor(0).set_names({"tensor_output"});
    results.push_back(res);
    return std::make_shared<Model>(results, params);
}

inline bool isCommandQueueExtSupported() {
    return std::make_shared<::intel_npu::ZeroInitStructsHolder>()->getCommandQueueDdiTable().version() > 0;
}

std::string generateCacheDirName(const std::string& test_name) {
    using namespace std::chrono;
    // Generate unique file names based on test name, thread id and timestamp
    // This allows execution of tests in parallel (stress mode)
    auto hash = std::to_string(std::hash<std::string>()(test_name));
    std::stringstream ss;
    auto ts = duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch());
    ss << hash << "_"
        << "_" << ts.count();
    return ss.str();
}

class CompileAndModelCaching : public testing::WithParamInterface<CompileAndModelCaching>,
                                 public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileAndModelCachingParams2> obj) {
        std::shared_ptr<ov::Model> model;
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(model, targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        std::printf("<====local test name> %s\n", result.str());
        return result.str();
    }

    void SetUp() override {
        std::tie(function, target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
            m_cache_dir = generateCacheDirName(GetTestName());
#ifdef WIN32
        auto temp = std::setenv("NPU_DYNAMIC_CACHING", m_cache_dir, 1);
        auto temp2 = std::setenv("VPU_DYNAMIC_CACHING", m_cache_dir, 1);
#else
       auto temp = std::setenv("ZE_INTEL_NPU_CACHE_DIR", m_cache_dir, 1);
#endif
        APIBaseTest::SetUp();
    }

    void TearDown() override {
        if (!m_cache_dir.empty()) {
            ov::test::utils::removeFilesWithExt(m_cache_dir, "blob");
            ov::test::utils::removeDir(m_cache_dir);
        }

        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        APIBaseTest::TearDown();
    }

    void run() override {
            //how to check windows and linux model cache dir?  can be manual setting.
    m_cache_dir = generateCacheDirName(GetTestName());
    //linux
        //linux can we write the caching into a file?
        //set a new env to store the model cachine file.
        //and then compare the two file?

        //also can be measured by time. add_abc first 148 and second 52ms
    
    // windows
        //exist location

// #if !defined(_WIN32) && !defined(_WIN64)
//         setenv("OV_GPU_CACHE_MODEL", "", 1);
// #endif

// #ifdef WIN32
//     auto temp = std::setenv("NPU_DYNAMIC_CACHING", m_cache_dir, 1);
//     auto temp2 = std::setenv("VPU_DYNAMIC_CACHING", m_cache_dir, 1);
// #else
//     auto temp = std::setenv("ZE_INTEL_NPU_CACHE_DIR", m_cache_dir, 1);
// #endif
        
    //check the the model cache has been generated successfully.
    }

protected:
    ov::CompiledModel execNet;
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::string targetDevice;
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
    std::string m_cache_dir; //it is need to be distinguished on Windows and Linux?

private:

}

TEST_P(CompileAndModelCaching, CompilationFirst) {
    //check folder is empty?
    size_t blobCountInitial = -1;
    blobCountInitial = ov::test::utils::listFilesWithExt(m_cache_dir, "blob").size();
    size_t blobCountAfterwards = -1;
    ASSERT_GT(blobCountInitial, 0);
    //first run time will long and will generate the model cache.
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    // ##how to check it is successfully?  multi thread will be influenced?
    blobCountAfterwards = ov::test::utils::listFilesWithExt(m_cache_dir, "blob").size();
    ASSERT_EQ(blobCountInitial, blobCountAfterwards - 1);
}

TEST_P(CompileAndModelCaching, CompilationSecond) {
    size_t blobCountInitial = -1;
    size_t blobCountAfterwards = -1;
    for (int i = 0; i < 2; i++) {
        //how to check windows and linux model caching?
        OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

        // ##how to check it is successfully?
        //check the the model cache has been generated successfully.
        if (i == 0) {
            // blob count should be greater than 0 initially
            blobCountInitial = ov::test::utils::listFilesWithExt(m_cache_dir, "blob").size();
            ASSERT_GT(blobCountInitial, 0);
        } else {
            // cache is created and reused. Blob count should be same as it was first time
            blobCountAfterwards = ov::test::utils::listFilesWithExt(m_cache_dir, "blob").size();
            ASSERT_EQ(blobCountInitial, blobCountAfterwards);
        }
    }
    //Need to test loaded cache can inference?
}


}  // namespace behavior
}  // namespace test
}  // namespace ov
