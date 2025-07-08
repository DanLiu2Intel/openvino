// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/opsets/opset6.hpp"
namespace ov {
namespace test {
namespace behavior {

std::shared_ptr<ov::Model> getFunction();

std::shared_ptr<ov::Model> getFunction() {
    const std::vector<size_t> inputShape = {1, 1, 128};
    const ov::element::Type_t inputType = ov::element::Type_t::f32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inputType, ov::Shape(inputShape))};
    params[0]->set_friendly_name("Parameter_1");
    params[0]->get_output_tensor(0).set_names({"Parameter_1"});

    auto relu = std::make_shared<ov::op::v0::Relu>(params[0]);
    relu->set_friendly_name("Relu_2");
    relu->get_output_tensor(0).set_names({"Relu_output"});

    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "Set_value"});

    auto read_value = std::make_shared<ov::op::v6::ReadValue>(relu->output(0), variable);
    read_value->set_friendly_name("ReadValue_3");
    read_value->get_output_tensor(0).set_names({"ReadValue_output"});

    auto assign = std::make_shared<ov::op::v6::Assign>(read_value->output(0), variable);
    assign->set_friendly_name("Assign_4");
    assign->get_output_tensor(0).set_names({"Assign_output"});

    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(assign);
    squeeze->set_friendly_name("Squeeze_5");
    squeeze->get_output_tensor(0).set_names({"Output_5"});

    auto result = std::make_shared<ov::op::v0::Result>(squeeze);
    result->set_friendly_name("Result_6");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, params, "Custom_model");
}

class NPUInferRequestElementTypeTests : public OVInferRequestDynamicTests {
protected:
    // std::string m_dynamic_type_out_xml_path{};
    // std::string m_dynamic_type_out_bin_path{};
    // std::string m_undefined_type_out_xml_path{};
    // std::string m_undefined_type_out_bin_path{};
    // std::string m_filePrefix{};

    // void SetupFileNames() {
    //     m_filePrefix = ov::test::utils::generateTestFilePrefix();
    //     const std::vector<std::pair<std::string*, std::string>> fileInfos = {
    //         {&m_dynamic_type_out_xml_path, "dynamic_type.xml"},
    //         {&m_dynamic_type_out_bin_path, "dynamic_type.bin"},
    //         {&m_undefined_type_out_xml_path, "undefined_type.xml"},
    //         {&m_undefined_type_out_bin_path, "undefined_type.bin"},
    //     };
    //     for (const auto& info : fileInfos) {
    //         *(info.first) = m_filePrefix + info.second;
    //     }
    // }

    // void RemoveFiles() {
    //     std::vector<std::string> files = {m_dynamic_type_out_xml_path,
    //                                       m_dynamic_type_out_bin_path,
    //                                       m_undefined_type_out_xml_path,
    //                                       m_undefined_type_out_bin_path};
    //     for (const auto& file : files) {
    //         std::remove(file.c_str());
    //     }
    // }

    void SetUp() override {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        //SetupFileNames();
        OVInferRequestDynamicTests::SetUp();
    }

    void TearDown() override {
        //RemoveFiles();
        OVInferRequestDynamicTests::TearDown();
    }

    bool files_equal(std::ifstream& f1, std::ifstream& f2) {
        if (!f1.is_open() || !f2.is_open())
            return false;

        return std::equal(std::istreambuf_iterator<char>(f1.rdbuf()),
                          std::istreambuf_iterator<char>(),
                          std::istreambuf_iterator<char>(f2.rdbuf()));
    }

    bool checkTwoTypeOutput(const ov::Tensor& dynamicOutput, const ov::Tensor& undefinedOutput) {
        bool result = true;
        const auto dynamicShape = dynamicOutput.get_shape();
        const auto undefinedShape = undefinedOutput.get_shape();
        if (dynamicShape.size() != undefinedShape.size()) {
            return false;
        }
        if (!std::equal(dynamicShape.cbegin(), dynamicShape.cend(), undefinedShape.cbegin())) {
            return false;
        }
        for (size_t i = 0; i < undefinedOutput.get_size(); i++) {
            if (fabs(dynamicOutput.data<float>()[i] - undefinedOutput.data<float>()[i]) >
                std::numeric_limits<float>::epsilon())
                return false;
        }
        return result;
    }
};

// Test whether the serialization and inference results of the dynamic type model
// and the undefined type model are the same
TEST_P(NPUInferRequestElementTypeTests, CompareDynamicAndUndefinedTypeNetwork) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    // Customize a model with a dynamic type
    std::string dynamicTypeModelXmlString = R"V0G0N(<?xml version="1.0"?>
<net name="custom_model" version="11">
    <layers>
        <layer id="0" name="Parameter_1" type="Parameter" version="opset1">
            <data shape="1,1,128" element_type="f32" />
            <output>
                <port id="0" precision="FP32" names="Parameter_1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Relu_2" type="ReLU" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="ReadValue_3" type="ReadValue" version="opset6">
            <data variable_id="my_var" variable_type="dynamic" variable_shape="..." />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Assign_4" type="Assign" version="opset6">
            <data variable_id="my_var" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="Squeeze_5" type="Squeeze" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32" names="Output_5">
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="Result_6" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>128</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0" />
        <edge from-layer="4" from-port="1" to-layer="5" to-port="0" />
    </edges>
    <rt_info />
</net>
)V0G0N";

    // Customize a model with a undefined type
    std::string undefinedTypeModelXmlString = R"V0G0N(<?xml version="1.0"?>
<net name="custom_model" version="11">
    <layers>
        <layer id="0" name="Parameter_1" type="Parameter" version="opset1">
            <data shape="1,1,128" element_type="f32" />
            <output>
                <port id="0" precision="FP32" names="Parameter_1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Relu_2" type="ReLU" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="ReadValue_3" type="ReadValue" version="opset6">
            <data variable_id="my_var" variable_type="undefined" variable_shape="..." />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Assign_4" type="Assign" version="opset6">
            <data variable_id="my_var" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="Squeeze_5" type="Squeeze" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32" names="Output_5">
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="Result_6" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>128</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0" />
        <edge from-layer="4" from-port="1" to-layer="5" to-port="0" />
    </edges>
    <rt_info />
</net>
)V0G0N";
    std::stringstream dynamicTypeModelXmlStream, undefinedTypeModelXmlStream;
    dynamicTypeModelXmlStream << dynamicTypeModelXmlString;
    undefinedTypeModelXmlStream << undefinedTypeModelXmlString;
    std::stringstream dynamicTypeModelBinStream, undefinedTypeModelBinStream;

    // Test whether the serialization results of the two models are the same
    auto dynamicTypeModel = ie->read_model(dynamicTypeModelXmlString, ov::Tensor());
    auto undefinedTypeModel = ie->read_model(undefinedTypeModelXmlString, ov::Tensor());

    // ov::pass::Serialize(m_dynamic_type_out_xml_path, m_dynamic_type_out_bin_path).run_on_model(dynamicTypeModel);
    // ov::pass::Serialize(m_undefined_type_out_xml_path, m_undefined_type_out_bin_path).run_on_model(undefinedTypeModel);

    // std::ifstream xml_dynamic(m_dynamic_type_out_xml_path, std::ios::in);
    // std::ifstream xml_undefined(m_undefined_type_out_xml_path, std::ios::in);
    // ASSERT_TRUE(files_equal(xml_dynamic, xml_undefined))
    //     << "Serialized XML files are different: " << m_dynamic_type_out_xml_path << " vs "
    //     << m_undefined_type_out_xml_path;

    ov::pass::Serialize(dynamicTypeModelXmlStream, dynamicTypeModelBinStream).run_on_model(dynamicTypeModel);   
    ov::pass::Serialize(undefinedTypeModelXmlStream, undefinedTypeModelBinStream).run_on_model(undefinedTypeModel);

    ASSERT_TRUE(dynamicTypeModelXmlStream.str() == undefinedTypeModelXmlStream.str())
        << "Serialized XML files are different: " << m_dynamic_type_out_xml_path << " vs "
        << m_undefined_type_out_xml_path;

    // Test whether the inference results of the two models are the same
    const std::string inputName = "Parameter_1";
    const std::string outputName = "Output_5";
    ov::Shape shape = inOutShapes[0].first;
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 100, 0);

    auto execNetDynamic = ie->compile_model(dynamicTypeModel, target_device, configuration);
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());

    auto execNetUndefined = ie->compile_model(undefinedTypeModel, target_device, configuration);
    ov::InferRequest reqUndefined;
    OV_ASSERT_NO_THROW(reqUndefined = execNetUndefined.create_infer_request());
    OV_ASSERT_NO_THROW(reqUndefined.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqUndefined.infer());

    ASSERT_TRUE(checkTwoTypeOutput(reqDynamic.get_tensor(outputName), reqUndefined.get_tensor(outputName)))
        << "Inference results are different: " << m_dynamic_type_out_xml_path << " vs "
        << m_undefined_type_out_xml_path;
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
