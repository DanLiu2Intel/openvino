// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"

struct _npu_vm_runtime_execution_context_handle_t;

namespace intel_npu {

class IDynamicGraph : public IGraph {
public:
    IDynamicGraph() = default;
    ~IDynamicGraph() override = default;

    virtual _npu_vm_runtime_handle_t* get_vm_runtime_handle() const = 0;

    virtual uint64_t get_num_subgraphs() const = 0;

    /// Lazily create (on first call) and return the VM execution context owned by the graph.
    /// The same context is shared across all infer requests of this graph; callers must
    /// serialize @c npuVMRuntimeExecute on it.
    virtual _npu_vm_runtime_execution_context_handle_t* ensure_execution_context() = 0;
};

}  // namespace intel_npu