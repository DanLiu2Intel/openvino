// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/swiglu.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/dynamic_quantize.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct fully_connected_test_params {
    ov::PartialShape in_shape;
    ov::PartialShape out_shape;
    ov::PartialShape weights_shape;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class FullyConnectedFusingTest : public ::BaseFusingTest<fully_connected_test_params> {
public:

    void execute(fully_connected_test_params& p, bool is_dynamic = false) {
        cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
        cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
        auto input_prim = this->get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, this->cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, this->cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        this->compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(fully_connected_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format,};
    }

    layout get_per_channel_layout(fully_connected_test_params& p) {
        return layout{ ov::PartialShape{1, p.out_shape[1]}, p.default_type, p.default_format };
    }

    size_t get_output_dim_size(fully_connected_test_params& p) {
        return p.out_shape.size();
    }

    layout get_weights_layout(fully_connected_test_params& p) {
        return layout{ p.weights_shape, p.weights_type, p.weights_format };
    }

    size_t get_input_weights_rank(fully_connected_test_params& p) {
        return p.weights_shape.size();
    }

    layout get_bias_layout(fully_connected_test_params& p) {
        auto bias_shape = p.out_shape.size() == 3 ? ov::PartialShape{1, 1, p.out_shape[2]} : ov::PartialShape{1, p.out_shape[1]};
        return layout{ bias_shape, p.default_type, p.default_format };
    }

    layout get_scale_layout(fully_connected_test_params& p, size_t group_size = 1) {
        if (p.weights_type == data_types::u8 || p.weights_type == data_types::i8 || p.weights_type == data_types::u4 || p.weights_type == data_types::i4) {
            auto scale_shape = p.out_shape.size() == 3 ? ov::PartialShape{p.out_shape[2]} : ov::PartialShape{p.out_shape[1]};
            return layout{ scale_shape, p.default_type, p.default_format };
        } else {
            auto groups_num = p.in_shape.size() == 3 ? p.in_shape[2] / group_size : p.in_shape[1] / group_size;
            auto scale_shape = p.out_shape.size() == 3 ? ov::PartialShape{p.out_shape[2], groups_num} : ov::PartialShape{p.out_shape[1], groups_num};
            return layout{ scale_shape, p.default_type, p.default_format };
        }
    }
};


#ifdef ENABLE_ONEDNN_FOR_GPU
class FullyConnectedFusingTestOneDNN : public BaseFusingTest<fully_connected_test_params> {
protected:
    std::unordered_map<std::string, layout> extra_inputs;

public:
    void execute(fully_connected_test_params& p, bool is_caching_test = false, bool is_dynamic = false) {
        // Onednn post operation has issue in a machine that does not support imad.
        if (!engine.get_device_info().supports_immad)
            GTEST_SKIP();

        auto input_prim = p.data_type == data_types::u8 ? get_mem(get_input_layout(p), 0, 10) : get_mem(get_input_layout(p), -1, 1);

        auto impl_forcing = cfg_fused.get_force_implementations();
        auto forcing_format = p.input_format;
        for (auto& forcing : impl_forcing)
            if (forcing.first == "fc_prim")
                forcing_format = forcing.second.output_format;

        ov::intel_gpu::ImplementationDesc fc_impl = { forcing_format, "", impl_types::onednn };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "fc_prim", fc_impl } }));
        cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));

        network::ptr network_not_fused = get_network(this->engine, this->topology_non_fused, cfg_not_fused, get_test_stream_ptr(cfg_not_fused), is_caching_test);
        network::ptr network_fused = get_network(this->engine, this->topology_fused, cfg_fused, get_test_stream_ptr(cfg_fused), is_caching_test);
        network_fused->set_input_data("input", input_prim);
        network_not_fused->set_input_data("input", input_prim);

        for (const auto& [input_id, data_layout] : extra_inputs) {
            auto data_mem = get_mem(data_layout, -1, 1);
            network_fused->set_input_data(input_id, data_mem);
            network_not_fused->set_input_data(input_id, data_mem);
        }

        compare(*network_not_fused, *network_fused, p);
    }

    layout get_input_layout(fully_connected_test_params& p) {
        return layout{ p.in_shape, p.data_type, p.input_format,};
    }

    layout get_per_channel_layout(fully_connected_test_params& p) {
        ov::PartialShape pshape = {1, p.out_shape[1]};
        if (p.out_shape.size() >= 3)
            pshape.push_back(1);
        if (p.out_shape.size() == 4)
            pshape.push_back(1);
        return layout{ pshape, p.default_type, p.default_format };
    }

    size_t get_output_dim_size(fully_connected_test_params& p) {
        return p.out_shape.size();
    }

    layout get_weights_layout(fully_connected_test_params& p) {
        return layout{ p.weights_shape, p.weights_type, p.weights_format };
    }

    size_t get_input_weights_rank(fully_connected_test_params& p) {
        return p.weights_shape.size();
    }

    layout get_bias_layout(fully_connected_test_params& p) {
        auto bias_shape = p.out_shape.size() == 3 ? ov::PartialShape{1, 1, p.out_shape[2]} : ov::PartialShape{1, p.out_shape[1]};
        return layout{ bias_shape, p.default_type, p.default_format };
    }

    layout get_scale_layout(fully_connected_test_params& p, size_t group_size = 1) {
        if (p.weights_type == data_types::u8 || p.weights_type == data_types::i8) {
            auto scale_shape = p.out_shape.size() == 3 ? ov::PartialShape{p.out_shape[2]} : ov::PartialShape{p.out_shape[1]};
            return layout{ scale_shape, p.default_type, p.default_format };
        } else {
            auto groups_num = p.in_shape.size() == 3 ? p.in_shape[2] / group_size : p.in_shape[1] / group_size;
            auto scale_shape = p.out_shape.size() == 3 ? ov::PartialShape{p.out_shape[2], groups_num} : ov::PartialShape{p.out_shape[1], groups_num};
            return layout{ scale_shape, p.default_type, p.default_format };
        }
    }

    layout get_output_layout(fully_connected_test_params& p) {
        return layout{ p.out_shape, p.data_type, p.input_format };
    }
};
#endif  // ENABLE_ONEDNN_FOR_GPU

}  // namespace

// in_shape; out_shape; kernel;  data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_FC_FP32_1 { 1, 3 }, { 1, 4 }, { 4, 3 }, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_2 { 2, 3 }, { 2, 4 }, { 4, 3 }, data_types::f32, format::yxfb, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3 { 2, 32 }, { 2, 16 }, { 16, 32 }, data_types::f32, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_1 { 5, 3, 3 }, { 5, 3, 5 }, { 5, 3 }, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_2 { 2, 1, 1 }, { 2, 1, 32 }, { 32, 1 }, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_3 { 2, 32, 32 }, { 2, 32, 16 }, { 16, 32 }, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx

#define DYN_CASE_FC_FP32_3D_1 { 5, 3, 3 }, { 5, 3, 5 }, { 5, 3 }, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define DYN_CASE_FC_FP32_3D_2 { 2, 1, 1 }, { 2, 1, 32 }, { 32, 1 }, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define DYN_CASE_FC_FP32_3D_3 { 2, 32, 32 }, { 2, 32, 16 }, { 16, 32 }, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx

#define CASE_FC_U8S8_1 { 1, 3 }, { 1, 4 }, { 4, 3 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_2 { 2, 3 }, { 2, 4 }, { 4, 3 }, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3 { 2, 32 }, { 2, 16 }, { 16, 32 }, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_4 { 1, 3 }, { 1, 3 }, { 3, 3 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_1 { 2, 32, 3 }, { 2, 32, 16 }, { 16, 3 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_2 { 1, 1, 3 }, { 1, 1, 32 }, { 32, 3 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_3 { 2, 3, 1 }, { 2, 3, 15 }, { 15, 1 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_4 { 1, 512, 1024 }, { 1, 384, 1024 }, { 1024, 1024 }, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx

#define CASE_FC_FP16_1 { 1, 3 }, { 1, 4 }, { 4, 3 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_2 { 2, 3 }, { 2, 4 }, { 4, 3 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_3 { 2, 32 }, { 2, 16 }, { 16, 32 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_4 { 128, 76 }, { 128, 768 }, { 768, 76 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_5 { 1, 128, 76 }, { 1, 128, 768 }, { 768, 76 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_6 { 2, 1, 76 }, { 2, 1, 768 }, { 768, 76 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_7 { 2, 128, 76 }, { 2, 128, 768 }, { 768, 76 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_3D_1 { 2, 32, 3 }, { 2, 32, 16 }, { 16, 3 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP16_3D_2 { 1, 1, 3 }, { 1, 1, 32 }, { 32, 3 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx

#define DYN_CASE_FC_FP16_5 { 1, 128, 76 }, { 1, 128, 768 }, { 768, 76 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define DYN_CASE_FC_FP16_6 { 2, 1, 76 }, { 2, 1, 768 }, { 768, 76 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define DYN_CASE_FC_FP16_7 { 2, 128, 76 }, { 2, 128, 768 }, { 768, 76 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define DYN_CASE_FC_FP16_3D_1 { 2, 32, 3 }, { 2, 32, 16 }, { 16, 3 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx
#define DYN_CASE_FC_FP16_3D_2 { 1, 1, 3 }, { 1, 1, 32 }, { 32, 3 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f32, format::bfyx

#define DYN_CASE_FC_FP16_4D_1 { 1, 1, 1, 1344 }, { 1, 1, 1, 512 }, { 512, 1344 }, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f16, format::bfyx


#define CASE_FC_FP16_INT4_COMP_1 { 1, 128 }, { 1, 128 }, { 128, 128 }, data_types::f16, format::bfyx, data_types::u4, format::oiyx, data_types::f16, format::bfyx
#define CASE_FC_FP16_INT4_COMP_3D_1 { 1, 1, 128 }, { 1, 1, 128 }, { 128, 128 }, data_types::f16, format::bfyx, data_types::i4, format::oiyx, data_types::f16, format::bfyx
#define CASE_FC_FP16_INT4_COMP_3D_2 { 1, 32, 128 }, { 1, 32, 128 }, { 128, 128 }, data_types::f16, format::bfyx, data_types::i4, format::oiyx, data_types::f16, format::bfyx
#define CASE_FC_FP16_INT4_COMP_3D_3 { 1, 96, 128}, { 1, 96, 128}, { 128, 128 }, data_types::f16, format::bfyx, data_types::i4, format::oiyx, data_types::f16, format::bfyx

#define CASE_FC_FP16_INT8_COMP_1 { 1, 128 }, { 1, 128 }, { 128, 128 }, data_types::f16, format::bfyx, data_types::u8, format::oiyx, data_types::f16, format::bfyx
#define CASE_FC_FP16_3D_INT8_COMP_1 { 2, 32, 4 }, { 2, 32, 16 }, { 16, 4 }, data_types::f16, format::bfyx, data_types::u8, format::oiyx, data_types::f16, format::bfyx

#define CASE_FC_FP16_INT4_SWIGLU_1 { 1, 64 }, { 1, 64 }, { 64, 64 }, data_types::f16, format::bfyx, data_types::u4, format::oiyx, data_types::f16, format::bfyx
#define CASE_FC_FP16_INT4_SWIGLU_2 { 1, 64}, { 1, 128 }, { 128, 64 }, data_types::f16, format::bfyx, data_types::u4, format::oiyx, data_types::f16, format::bfyx
#define CASE_FC_FP16_INT4_SWIGLU_3 { 1, 312 }, { 1, 128 }, { 128, 312 }, data_types::f16, format::bfyx, data_types::u4, format::oiyx, data_types::f16, format::bfyx
#define CASE_FC_FP16_INT4_SWIGLU_4 { 8, 1, 64}, { 8, 1, 128 }, { 128, 64 }, data_types::f16, format::bfyx, data_types::u4, format::oiyx, data_types::f16, format::bfyx

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- FC cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
class fc_fp32_activation : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_activation, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p), get_input_weights_rank(p)),
        activation("activation", input_info("fc_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_activation, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_3, 2, 3 },
}));

class fc_fp32_activation_dynamic : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_activation_dynamic, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().size()), test_input_layout.data_type, test_input_layout.format};
    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p), get_input_weights_rank(p)),
        activation("activation", input_info("fc_prim"), activation_func::abs),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p, true);
}
INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_activation_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_1, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_2, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_3, 2, 3 },
}));

class fc_fp32_bias : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_bias, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "", get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("bias_add", { input_info("fc_prim"), input_info("bias") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("bias_add"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_bias, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3D_3, 2, 3 },
}));

class fc_fp32_bias_dynamic : public FullyConnectedFusingTest {};
TEST_P(fc_fp32_bias_dynamic, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().rank()), test_input_layout.data_type, test_input_layout.format};
    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "", get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("bias_add", { input_info("fc_prim"), input_info("bias") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("bias_add"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_bias_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP32_3, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_1, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_2, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP32_3D_3, 2, 3 },
}));

class fc_compressed_int8_bias_dynamic : public FullyConnectedFusingTest {};
TEST_P(fc_compressed_int8_bias_dynamic, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().rank()), test_input_layout.data_type, test_input_layout.format};
    auto supports_immad = this->engine.get_device_info().supports_immad;

    auto dcomp_zp_mem = engine.allocate_memory({ {1, 1, 1, 1}, data_types::f32, format::bfyx });
    set_values(dcomp_zp_mem, {8.0f});

    auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", dcomp_zp_name, data_types::f16, get_output_dim_size(p), get_input_weights_rank(p));

    fc_prim.decompression_zero_point_scalar = 8.0f;

    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale", get_mem(get_scale_layout(p, 128))),
        data("bias", get_mem(get_bias_layout(p))),
        data("dcomp_zp", dcomp_zp_mem),
        fc_prim,
        eltwise("bias_add", { input_info("fc_prim"), input_info("bias") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("bias_add"), p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_compressed_int8_bias_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_INT4_COMP_1, 2, 3 },
}));

class fc_int8_eltwise_dynamic_residual : public FullyConnectedFusingTest {};
TEST_P(fc_int8_eltwise_dynamic_residual, basic) {
    // The basic purpose of this test is to check crash in fake aligned shape check
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().rank()), test_input_layout.data_type, test_input_layout.format};
    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weight", get_mem(get_weights_layout(p))),
        reorder("reorder", input_info("input"), p.default_format, data_types::i8),
        eltwise("mul", { input_info("reorder"), input_info("input") }, eltwise_mode::div),
        fully_connected("fc", input_info("mul"), "weight", "", data_types::i8, get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("add", { input_info("fc"), input_info("reorder") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("add"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p, true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_eltwise_dynamic_residual, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_4, 3, 4 },
}));


class fc_int8_eltwise : public FullyConnectedFusingTest {};
TEST_P(fc_int8_eltwise, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_eltwise, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 3 },
}));

class fc_int8_quantize_u8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_quantize_u8, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, get_output_dim_size(p), get_input_weights_rank(p)),
        quantize("quantize", input_info("fc_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_quantize_u8, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_3, 2, 3 },
}));

class fc_int8_eltwise_quantize_i8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_eltwise_quantize_i8, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltwise_data", get_mem(get_per_channel_layout(p), 1.0f / get_weights_layout(p).count() / 255)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::prod),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_eltwise_quantize_i8, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_3, 2, 4 },
}));

class fc_int8_eltwise_activation_quantize_i8 : public FullyConnectedFusingTest {};
TEST_P(fc_int8_eltwise_activation_quantize_i8, basic) {
    // TODO: Fix me, refer PR(#15873)
    if (engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltwise_data", get_mem(get_per_channel_layout(p), 1.0f / get_weights_layout(p).count() / 255)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", data_types::f32, get_output_dim_size(p), get_input_weights_rank(p)),
        eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::prod),
        activation("activation_eltwise", input_info("eltwise"), activation_func::exp),
        quantize("quantize", input_info("activation_eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_eltwise_activation_quantize_i8, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 5 },
    fully_connected_test_params{ CASE_FC_U8S8_2, 2, 5 },
    fully_connected_test_params{ CASE_FC_U8S8_3, 2, 5 },

    fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 5 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 5 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_3, 2, 5 },

    fully_connected_test_params{ CASE_FC_FP32_3D_1, 3, 5 },
    fully_connected_test_params{ CASE_FC_FP32_3D_2, 3, 5 },
    fully_connected_test_params{ CASE_FC_FP32_3D_3, 3, 5 },
}));

#ifdef ENABLE_ONEDNN_FOR_GPU

// FC onednn sum case
class fc_int8_inputs_fused_fp32_sum : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        auto shift_layout = layout{ ov::PartialShape{p.weights_shape[0]}, p.default_type, p.default_format };

        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("shift_data", get_mem(shift_layout, 1)),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", cldnn::data_types::f32, get_output_dim_size(p), get_input_weights_rank(p)),
            eltwise("shift", { input_info("fc_prim"), input_info("shift_data") }, eltwise_mode::sum, cldnn::data_types::f32),
            crop("crop", input_info("shift"), get_output_layout(p).get_tensor(), { 0, 0, 0, 0 }),
            reorder("reorder_bfyx", input_info("crop"), p.default_format, data_types::f32)
        );

        tolerance = 1.f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_int8_inputs_fused_fp32_sum, basic) {
    run_test(false);
}

TEST_P(fc_int8_inputs_fused_fp32_sum, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_inputs_fused_fp32_sum, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    // OneDNN has issue with small shapes - ticket 7064
    // fully_connected_test_params{ CASE_FC_U8S8_3D_1, 2, 4 },
    // fully_connected_test_params{ CASE_FC_U8S8_3D_2, 2, 4 },
    fully_connected_test_params{ CASE_FC_U8S8_3D_4, 2, 4 },
}));


class fc_fp16_eltwise_add : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("weights", get_mem(get_weights_layout(p), -1, 1)),
            data("bias", get_mem(get_bias_layout(p), -2, 2)),
            data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p)),
            eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sum),
            reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
        );

        tolerance = 1e-2f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_fp16_eltwise_add, basic) {
    run_test(false);
}

TEST_P(fc_fp16_eltwise_add, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_add, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_4, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_5, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_6, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_7, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_2, 2, 3 },
}));

class fc_fp16_eltwise_add_full_tensor : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        create_topologies(
            input_layout("input", get_input_layout(p)),
            input_layout("input2", get_output_layout(p)),
            data("weights", get_mem(get_weights_layout(p), -1, 1)),
            data("bias", get_mem(get_bias_layout(p), -2, 2)),
            activation("activation", input_info("input2"), activation_func::relu),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p)),
            eltwise("eltwise", { input_info("fc_prim"), input_info("activation") }, eltwise_mode::sum),
            reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f16)
        );

        extra_inputs["input2"] = get_output_layout(p);

        tolerance = 1e-2f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_fp16_eltwise_add_full_tensor, basic) {
    run_test(false);
}

TEST_P(fc_fp16_eltwise_add_full_tensor, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_add_full_tensor, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_2, 4, 5 },
    fully_connected_test_params{ CASE_FC_FP16_3D_1, 4, 5 },
}));

class fc_fp16_eltwise_add_dynamic : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_fp16_eltwise_add_dynamic, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().size()), test_input_layout.data_type, test_input_layout.format};
    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p)),
        eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    bool is_dynamic = true;
    cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
    tolerance = 1e-2f;
    execute(p, false, is_dynamic);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_add_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_4, 2, 3 }
}));

class fc_fp16_eltwise_prod_unfused_dynamic : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_fp16_eltwise_prod_unfused_dynamic, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().size()), test_input_layout.data_type, test_input_layout.format};
    auto data_layout = layout{ ov::PartialShape{p.out_shape[0], 1}, p.default_type, p.default_format };
    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weights", get_mem(get_weights_layout(p), -1, 1)),
        data("bias", get_mem(get_bias_layout(p), -2, 2)),
        data("eltwise_data", get_mem(data_layout, -1, 1)),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p)),
        eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    bool is_dynamic = true;
    cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
    tolerance = 0.5f;
    execute(p, false, is_dynamic);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_prod_unfused_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_4, 2, 3 }
}));

class fc_compressed_int8_bias_eltwise_quantize_u8_onednn : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_compressed_int8_bias_eltwise_quantize_u8_onednn, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);

    auto supports_immad = engine.get_device_info().supports_immad;
    auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", dcomp_zp_name, data_types::f16, get_output_dim_size(p), get_input_weights_rank(p));
    fc_prim.decompression_zero_point_scalar = 8.0f;

    // onednn FC supports scalar ZP for int4 compressed weight.
    auto dcomp_zp_layout = layout{ {1, 1, 1, 1}, data_types::u8, format::bfyx };

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale", get_mem(get_scale_layout(p, 128))),
        data("bias", get_mem(get_bias_layout(p))),
        data("dcomp_zp", get_mem(dcomp_zp_layout, 8.0f)),
        data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
        data("in_lo", get_mem(get_per_channel_layout(p), -2, -2)),
        data("in_hi", get_mem(get_per_channel_layout(p), 2, 2)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        fc_prim,
        eltwise("bias_add", { input_info("fc_prim"), input_info("bias") }, eltwise_mode::sum),
        eltwise("eltwise", { input_info("bias_add"), input_info("eltwise_data") }, eltwise_mode::sum),
        quantize("quantize", input_info("eltwise"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    bool is_dynamic = false;
    cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
    cfg_not_fused.set_property(ov::hint::dynamic_quantization_group_size(0));
    tolerance = 1.0f;
    execute(p, false, is_dynamic);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_compressed_int8_bias_eltwise_quantize_u8_onednn, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_INT8_COMP_1, 2, 5 },
    fully_connected_test_params{ CASE_FC_FP16_3D_INT8_COMP_1, 2, 5 },
}));

// Check whether dyn_quan_fc can create quantized output. Currently, OneDNN cannot.
class fc_compressed_dyn_quan_and_quantized : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_compressed_dyn_quan_and_quantized, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);

    if (!engine.get_device_info().supports_immad)
        return;

    auto fc_prim_fused = fully_connected("fc_prim", input_info("dyn_quan", 0), "weights", "", "scale", "", input_info("dyn_quan", 1), input_info("", 0), data_types::f16, get_output_dim_size(p), get_input_weights_rank(p));
    auto fc_prim_unfused = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, get_output_dim_size(p), get_input_weights_rank(p));
    auto weights = data("weights", get_mem(get_weights_layout(p)));
    auto scale = data("scale", get_mem(get_scale_layout(p, 128), 0.05f));
    auto in_lo = data("in_lo", get_mem(get_per_channel_layout(p), -200.f));
    auto in_hi = data("in_hi", get_mem(get_per_channel_layout(p), 200.f));
    auto out_lo = data("out_lo", get_mem(get_single_element_layout(p), 0));
    auto out_hi = data("out_hi", get_mem(get_single_element_layout(p), 255));

    fc_prim_fused.decompression_zero_point_scalar = 8.0f;
    fc_prim_unfused.decompression_zero_point_scalar = 8.0f;

    ov::op::internal::DynamicQuantize::Attributes dyn_quan_attr;
    dyn_quan_attr.group_sizes = std::vector<uint64_t>(get_output_dim_size(p) - 1, 1);
    dyn_quan_attr.group_sizes.emplace_back(get_input_layout(p).feature()); // per-token quantization
    dyn_quan_attr.scale_dt = ov::element::f16;
    dyn_quan_attr.quantization_dt = ov::element::i8;

    // OneDNN does not support quantized output of dyn_quan_fc
    topology_fused.add(
        input_layout("input", get_input_layout(p)),
        weights, scale, in_lo, in_hi, out_lo, out_hi,
        dynamic_quantize("dyn_quan", input_info("input"), dyn_quan_attr, 2),
        fc_prim_fused,
        quantize("quantize", input_info("fc_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    // Non-fused does not have dyn_quan because it will use cldnn FC
    topology_non_fused.add(
        input_layout("input", get_input_layout(p)),
        weights, scale, in_lo, in_hi, out_lo, out_hi,
        fc_prim_unfused,
        quantize("quantize", input_info("fc_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 256, data_types::u8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );


    cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    tolerance = 10.0f;  // tolerance is OK to be high because it is supposed to have high error due to dyn_quan
    execute(p, false, true);
}

#define CASE_FC_FP16_INT8_COMP_DYN_QUAN { 64, 128 }, { 64, 128 }, { 128, 128 }, data_types::f16, format::bfyx, data_types::u8, format::oiyx, data_types::f16, format::bfyx
INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_compressed_dyn_quan_and_quantized, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_INT8_COMP_DYN_QUAN, 4, 3 },
}));

class fc_compressed_int8_bias_dynamic_onednn : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_compressed_int8_bias_dynamic_onednn, basic) {
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().rank()), test_input_layout.data_type, test_input_layout.format};

    auto supports_immad = engine.get_device_info().supports_immad;
    auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", dcomp_zp_name, data_types::f16, get_output_dim_size(p), get_input_weights_rank(p));
    fc_prim.decompression_zero_point_scalar = 8.0f;

    // onednn FC supports scalar ZP for int4 compressed weight.
    auto dcomp_zp_layout = layout{ {1, 1, 1, 1}, data_types::u8, format::bfyx };

    create_topologies(
        input_layout("input", dynamic_input_layout),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale", get_mem(get_scale_layout(p, 128))),
        data("bias", get_mem(get_bias_layout(p))),
        data("dcomp_zp", get_mem(dcomp_zp_layout, 8.0f)),
        fc_prim,
        eltwise("bias_add", { input_info("fc_prim"), input_info("bias") }, eltwise_mode::sum),
        reorder("reorder_bfyx", input_info("bias_add"), p.default_format, data_types::f32)
    );

    bool is_dynamic = true;
    cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
    cfg_not_fused.set_property(ov::hint::dynamic_quantization_group_size(0));
    tolerance = 1.0f;
    execute(p, false, is_dynamic);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_compressed_int8_bias_dynamic_onednn, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_INT4_COMP_1, 2, 3 },
}));

class fc_compressed_int8_bias_prod_unfused_dynamic_onednn : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_compressed_int8_bias_prod_unfused_dynamic_onednn, basic) {
    // Unfusion will happen because of this mul_data_shape
    auto p = GetParam();
    auto test_input_layout = get_input_layout(p);
    auto feature_len = test_input_layout.get_partial_shape()[-1].get_length();   // This is per-token quantization. feature_len == group_size
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().rank()), test_input_layout.data_type, test_input_layout.format};

    ov::PartialShape mul_data_partial_shape;
    mul_data_partial_shape.emplace_back(2);
    for (size_t i = 0; i < p.in_shape.size() - 1; i++)
        mul_data_partial_shape.emplace_back(1);

    auto mul_data_shape = layout{ mul_data_partial_shape, p.default_type, p.default_format };

    auto supports_immad = engine.get_device_info().supports_immad;
    auto dcomp_zp_name = supports_immad ? "" : "";

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", dcomp_zp_name, data_types::f16, get_output_dim_size(p), get_input_weights_rank(p));
    auto fc_prim_dyn_quan = fully_connected("fc_prim", input_info("dyn_quan", 0), "weights", "", "scale", dcomp_zp_name, input_info("dyn_quan", 1), input_info(""), data_types::f16, get_output_dim_size(p), get_input_weights_rank(p));

    auto dcomp_zp_layout = layout{ {1, 1}, data_types::u8, format::bfyx };

    auto weights = data("weights", get_mem(get_weights_layout(p)));
    auto scale = data("scale", get_mem(get_scale_layout(p, feature_len)));
    auto dcomp_zp = data("dcomp_zp", get_mem(dcomp_zp_layout, 8.0f));
    auto mul_data = data("mul_data", get_mem(mul_data_shape, -2, 2));

    dynamic_quantize::Attributes dq_config;
    dq_config.quantization_dt = data_types::i8;
    dq_config.scale_dt = data_types::f16;
    dq_config.group_sizes = std::vector<uint64_t>(p.in_shape.size() - 1, 1);
    dq_config.group_sizes.push_back(feature_len);

    topology_non_fused.add(
        input_layout("input", dynamic_input_layout),
        weights,
        scale,
        dcomp_zp,
        mul_data,
        fc_prim,
        eltwise("mul", { input_info("fc_prim"), input_info("mul_data") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("mul"), p.default_format, data_types::f32)
    );

    topology_fused.add(
        input_layout("input", dynamic_input_layout),
        weights,
        scale,
        dcomp_zp,
        mul_data,
        dynamic_quantize("dyn_quan", input_info("input"), dq_config, 3),
        fc_prim_dyn_quan,
        eltwise("mul", { input_info("fc_prim"), input_info("mul_data") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("mul"), p.default_format, data_types::f32)
    );

    bool is_dynamic = true;
    cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
    cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(is_dynamic));
    tolerance = 1.0f;
    execute(p, false, is_dynamic);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_compressed_int8_bias_prod_unfused_dynamic_onednn, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_INT4_COMP_3D_1, 2, 3 },   // dyn_quan is skipeed at runtime
    fully_connected_test_params{ CASE_FC_FP16_INT4_COMP_3D_2, 2, 3 },   // dyn_quan is skipped at runtime
    fully_connected_test_params{ CASE_FC_FP16_INT4_COMP_3D_3, 3, 3 },   // dyn_quan is not skipped
}));

class fc_fp16_eltwise_sub : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p)),
            eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sub),
            reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
        );

        tolerance = 1e-1f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_fp16_eltwise_sub, basic) {
    run_test(false);
}

TEST_P(fc_fp16_eltwise_sub, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_sub, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_2, 2, 3 },
}));

class fc_fp16_eltwise_prod : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("eltwise_data", get_mem(get_per_channel_layout(p), 1, 9)),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p)),
            eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::prod),
            reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
        );

        tolerance = 1e-1f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_fp16_eltwise_prod, basic) {
    run_test(false);
}

TEST_P(fc_fp16_eltwise_prod, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_prod, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_2, 2, 3 },
}));

class fc_fp16_eltwise_sum : public FullyConnectedFusingTestOneDNN {
public:
    void run_test(bool is_caching_test = false) {
        auto p = GetParam();
        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("eltwise_data", get_mem(get_output_layout(p))),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p)),
            eltwise("sum", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sum),
            reorder("reorder_bfyx", input_info("sum"), p.default_format, data_types::f32)
        );

        tolerance = 1e-1f;
        execute(p, is_caching_test);
    }
};

TEST_P(fc_fp16_eltwise_sum, basic) {
    run_test(false);
}

TEST_P(fc_fp16_eltwise_sum, basic_cached) {
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_sum, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3D_2, 2, 3 },
}));

class fc_fp32_activation_prelu : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_fp32_activation_prelu, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("data", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p), get_input_weights_rank(p)),
        activation("activation", input_info("fc_prim"), "data", activation_func::relu_negative_slope),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_activation_prelu, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 3, 3 }
}));

class fc_fp32_activation_relu : public FullyConnectedFusingTestOneDNN {};
TEST_P(fc_fp32_activation_relu, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p), get_input_weights_rank(p)),
        activation("activation", input_info("fc_prim"), activation_func::relu_negative_slope),
        reorder("reorder_bfyx", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_activation_relu, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP32_1, 2, 3 }
}));
#endif

class fc_fp16_eltwise_add_ocl_dynamic : public FullyConnectedFusingTest {
public:
    void run_test() {
        auto p = GetParam();
        auto test_input_layout = get_input_layout(p);
        auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().size()), test_input_layout.data_type, test_input_layout.format};
        auto eltwise_data_shape = p.out_shape.size() == 3 ? ov::PartialShape{1, 1, p.out_shape[2]} : ov::PartialShape{1, p.out_shape[1]};
        auto eltwise_data_layout = layout{eltwise_data_shape, p.default_type, p.default_format};
        create_topologies(
            input_layout("input", dynamic_input_layout),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("eltwise_data", get_mem(eltwise_data_layout, 1, 9)),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p)),
            eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sum),
            reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
        );

        tolerance = 1e-2f;
        execute(p, true);
    }
};

TEST_P(fc_fp16_eltwise_add_ocl_dynamic, basic) {
    if (engine.get_device_info().supports_immad)
        return;
    run_test();
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_eltwise_add_ocl_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_4, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP16_5, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP16_6, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP16_7, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP16_3D_1, 2, 3 },
    fully_connected_test_params{ DYN_CASE_FC_FP16_3D_2, 2, 3 },
}));

class fc_fp16_swiglu_ocl_dynamic : public FullyConnectedFusingTest {
public:
    void run_test(bool is_per_channel_quan) {
        auto p = GetParam();
        auto test_input_layout = get_input_layout(p);
        auto dynamic_input_layout = layout{ov::PartialShape::dynamic(test_input_layout.get_partial_shape().size()),
                                           test_input_layout.data_type,
                                           test_input_layout.format};
        int64_t swiglu_length = p.weights_shape[0].get_length()/2;
        auto fc_prim = fully_connected("fc_prim",
                                       input_info("input"),
                                       "weights",
                                       "",
                                       "scale",
                                       "",
                                       data_types::f16,
                                       get_output_dim_size(p),
                                       get_input_weights_rank(p));
        fc_prim.decompression_zero_point_scalar = 8.0f;
        auto group_size = is_per_channel_quan ? (p.in_shape.size() == 3 ? p.in_shape[2].get_length() : p.in_shape[1].get_length()) : 64;
        auto groups_num = p.in_shape.size() == 3 ? p.in_shape[2] / group_size : p.in_shape[1] / group_size;
        auto scale_shape = p.out_shape.size() == 3 ? ov::PartialShape{p.out_shape[2], groups_num} : ov::PartialShape{p.out_shape[1], groups_num};

        create_topologies(input_layout("input", dynamic_input_layout),
                          data("weights", get_mem(get_weights_layout(p))),
                          data("scale", get_mem(layout{scale_shape, p.default_type, p.default_format}, 0.1)),
                          fc_prim,
                          swiglu("swiglu",
                                 input_info("fc_prim"),
                                 -1,
                                 swiglu_length,
                                 ov::op::internal::GLU::GluType::Swish,
                                 0,
                                 tensor()),
                          reorder("reorder_bfyx", input_info("swiglu"), p.default_format, data_types::f32));

        tolerance = 1.0f;
        execute(p, true);
    }
};

TEST_P(fc_fp16_swiglu_ocl_dynamic, basic) {
    if (engine.get_device_info().supports_immad)
        return;

    if (engine.get_device_info().execution_units_count < 128)
        return;
    run_test(false);
}

TEST_P(fc_fp16_swiglu_ocl_dynamic, per_channel_quan) {
    if (engine.get_device_info().supports_immad)
        return;

    if (engine.get_device_info().execution_units_count < 128)
        return;
    run_test(true);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_swiglu_ocl_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_FP16_INT4_SWIGLU_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_INT4_SWIGLU_2, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_INT4_SWIGLU_3, 2, 3 },
    fully_connected_test_params{ CASE_FC_FP16_INT4_SWIGLU_4, 2, 3 },
}));

class fc_imad_int8_eltwise_add_ocl_dynamic : public FullyConnectedFusingTest {
public:
    void run_test() {
        auto p = GetParam();
        auto test_input_layout = get_input_layout(p);

        auto dyn_input_pshape = ov::PartialShape::dynamic(test_input_layout.get_partial_shape().size());
        dyn_input_pshape[p.in_shape.size() - 1] = p.in_shape[p.in_shape.size() - 1];
        auto dynamic_input_layout = layout{dyn_input_pshape, test_input_layout.data_type, test_input_layout.format};

        auto eltwise_data_shape = p.out_shape.size() == 3 ? ov::PartialShape{1, 1, p.out_shape[2]} : ov::PartialShape{1, p.out_shape[1]};
        auto eltwise_data_layout = layout{eltwise_data_shape, p.default_type, p.default_format};

        create_topologies(
            input_layout("input", dynamic_input_layout),
            data("weights", get_mem(get_weights_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("eltwise_data", get_mem(eltwise_data_layout, 1, 9)),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", get_output_dim_size(p)),
            eltwise("eltwise", { input_info("fc_prim"), input_info("eltwise_data") }, eltwise_mode::sum),
            reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
        );

        ov::intel_gpu::ImplementationDesc fc_impl = { p.input_format, "fully_connected_gpu_imad", impl_types::ocl };
        cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "fc_prim", fc_impl } }));

        tolerance = default_tolerance(p.data_type);
        execute(p, true);
    }
};

TEST_P(fc_imad_int8_eltwise_add_ocl_dynamic, basic) {
    if (engine.get_device_info().supports_immad)
        return;
    run_test();
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_imad_int8_eltwise_add_ocl_dynamic, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ CASE_FC_U8S8_1, 2, 3 },
    fully_connected_test_params{ CASE_FC_U8S8_4, 2, 3 },
}));

class fc_fp16_fuse_bias_and_find_eltwise_4d : public FullyConnectedFusingTest {
public:
    void run_test() {
        auto p = GetParam();
        auto test_input_layout = get_input_layout(p);
        auto in_layout = layout{ ov::PartialShape::dynamic(test_input_layout.get_partial_shape().size()),
                                 test_input_layout.data_type,
                                 test_input_layout.format };
        auto data_layout = layout{ p.out_shape, p.default_type, p.default_format };
        auto weight = layout{ { 29, 512 }, data_types::f16, format::bfyx };
        auto bias = layout{ { 1, 1, 1, 29 }, data_types::f16, format::bfyx };

        create_topologies(
            input_layout("input", in_layout),
            data("weights", get_mem(get_weights_layout(p))),
            data("data", get_mem(data_layout, 1, 9)),
            fully_connected("fc_prim", input_info("input"), "weights"),
            eltwise("add", { input_info("fc_prim"), input_info("data") }, eltwise_mode::sum),
            activation("relu", input_info("add"), activation_func::relu),
            reorder("reorder", input_info("relu"), p.default_format, data_types::f16)
        );

        tolerance = 1e-2f;
        execute(p, true);
    }
};

TEST_P(fc_fp16_fuse_bias_and_find_eltwise_4d, basic) {
    if (engine.get_device_info().supports_immad)
        return;
    run_test();
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp16_fuse_bias_and_find_eltwise_4d, ::testing::ValuesIn(std::vector<fully_connected_test_params>{
    fully_connected_test_params{ DYN_CASE_FC_FP16_4D_1, 2, 4 },
}));
