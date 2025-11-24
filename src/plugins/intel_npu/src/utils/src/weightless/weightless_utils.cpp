// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/weightless/weightless_utils.hpp"

namespace intel_npu {

// why not put this function in  plugin_compiler_adapter.cpp, like the src/plugins/intel_npu/src/compiler_adapter/src/driver_compiler_adapter.cpp
// https://github.com/openvinotoolkit/openvino/blame/master/src/plugins/intel_npu/src/compiler_adapter/src/driver_compiler_adapter.cpp#L42
/// XXX driver cadapter 和plugin adapter 都能用这个函数
void storeWeightlessCacheAttribute(const std::shared_ptr<ov::Model>& model) const {
    size_t constantId = 0;
    for (auto&& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            ov::RTMap& runtimeInfoMap = node->get_rt_info();
            const auto& weightlessCacheAttrIt =
                runtimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());

            const std::string constantIdString = std::to_string(constantId++);
            if (weightlessCacheAttrIt != runtimeInfoMap.end()) {
                auto& weightlessCacheAttr = weightlessCacheAttrIt->second.as<ov::WeightlessCacheAttribute>();
                model->set_rt_info(weightlessCacheAttr.bin_offset, "ws_bin_offset_" + constantIdString);
                model->set_rt_info(weightlessCacheAttr.original_size, "ws_original_size_" + constantIdString);
                model->set_rt_info(weightlessCacheAttr.original_dtype, "ws_original_dtype_" + constantIdString);
            }
        }
    }
}


}  // namespace intel_npu