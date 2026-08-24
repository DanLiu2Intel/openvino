// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"

namespace {

std::shared_ptr<ov::op::v0::Constant> i64_const(const std::vector<int64_t>& vals) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{vals.size()}, vals);
}

std::shared_ptr<ov::Model> build_conv_add_softmax_conv_equivalent_model(size_t sq, size_t sk, size_t d, size_t ev) {
    using namespace ov;

    auto q = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, sq, d});
    q->set_friendly_name("Q");
    auto k = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, sk, d});
    k->set_friendly_name("K");
    auto v = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, sk, ev});
    v->set_friendly_name("V");
    auto m = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, sq, sk});
    m->set_friendly_name("M");

    // Conv1 emulates Q @ K^T for the fixed B=1, H=1 layout.
    auto q_nchw = std::make_shared<op::v1::Transpose>(q, i64_const({0, 3, 1, 2}));
    auto k_2d = std::make_shared<op::v0::Squeeze>(k, i64_const({0, 1}));  // [Sk, D]
    auto k_w = std::make_shared<op::v1::Reshape>(
        k_2d,
        i64_const({static_cast<int64_t>(sk), static_cast<int64_t>(d), 1, 1}),
        false);

    auto qk_nchw = std::make_shared<op::v1::Convolution>(q_nchw,
                                                          k_w,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});    // [1, Sk, 1, Sq]
    auto qk = std::make_shared<op::v1::Transpose>(qk_nchw, i64_const({0, 2, 3, 1}));    // [1, 1, Sq, Sk]

    auto logits = std::make_shared<op::v1::Add>(qk, m);
    auto probs = std::make_shared<op::v8::Softmax>(logits, -1);

    // Conv2 emulates probs @ V.
    auto p_nchw = std::make_shared<op::v1::Transpose>(probs, i64_const({0, 3, 1, 2}));  // [1, Sk, 1, Sq]
    auto v_2d = std::make_shared<op::v0::Squeeze>(v, i64_const({0, 1}));                 // [Sk, Ev]
    auto v_w2d = std::make_shared<op::v1::Transpose>(v_2d, i64_const({1, 0}));            // [Ev, Sk]
    auto v_w = std::make_shared<op::v1::Reshape>(
        v_w2d,
        i64_const({static_cast<int64_t>(ev), static_cast<int64_t>(sk), 1, 1}),
        false);

    auto out_nchw = std::make_shared<op::v1::Convolution>(p_nchw,
                                                           v_w,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});  // [1, Ev, 1, Sq]
    auto out = std::make_shared<op::v1::Transpose>(out_nchw, i64_const({0, 2, 3, 1}));   // [1, 1, Sq, Ev]
    out->set_friendly_name("attn_out_conv_like");

    auto result = std::make_shared<op::v0::Result>(out);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{q, k, v, m}, "conv_add_softmax_conv_like_sdpa");
}

std::shared_ptr<ov::Model> build_sdpa_reference_model(size_t sq, size_t sk, size_t d, size_t ev) {
    using namespace ov;

    auto q = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, sq, d});
    q->set_friendly_name("Q");
    auto k = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, sk, d});
    k->set_friendly_name("K");
    auto v = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, sk, ev});
    v->set_friendly_name("V");
    auto m = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, sq, sk});
    m->set_friendly_name("M");

    auto sdpa = std::make_shared<op::v13::ScaledDotProductAttention>(q, k, v, m, false);
    sdpa->set_friendly_name("attn_out_sdpa");

    auto result = std::make_shared<op::v0::Result>(sdpa);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{q, k, v, m}, "sdpa_reference");
}

void serialize_model(const std::shared_ptr<ov::Model>& model, const std::filesystem::path& xml_path) {
    const auto bin_path = xml_path.parent_path() / (xml_path.stem().string() + ".bin");
    ov::serialize(model, xml_path, bin_path);
    std::cout << "Serialized: " << xml_path << " and " << bin_path << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out_dir = (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::path{"."};
    std::filesystem::create_directories(out_dir);

    // Minimal static shapes for a standalone export demo.
    const size_t sq = 4;
    const size_t sk = 4;
    const size_t d = 8;
    const size_t ev = 8;

    auto conv_like = build_conv_add_softmax_conv_equivalent_model(sq, sk, d, ev);
    auto sdpa_ref = build_sdpa_reference_model(sq, sk, d, ev);

    serialize_model(conv_like, out_dir / "conv_add_softmax_conv_like_sdpa.xml");
    serialize_model(sdpa_ref, out_dir / "sdpa_reference.xml");

    return 0;
}
