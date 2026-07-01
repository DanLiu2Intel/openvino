// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "intel_npu/common/igraph.hpp"

namespace intel_npu {

template <typename HandleT>
HandleT get_graph_handle_or_throw(const IGraph& graph) {
    static_assert(std::is_same_v<HandleT, ze_graph_handle_t> ||
                      std::is_same_v<HandleT, npu_vm_runtime_handle_t>,
                  "Unsupported graph handle type");

    const auto native_handle = graph.get_handle();
    const auto* handle = std::get_if<HandleT>(&native_handle);
    OPENVINO_ASSERT(handle != nullptr, "Graph handle type mismatch");
    return *handle;
}

}  // namespace intel_npu
