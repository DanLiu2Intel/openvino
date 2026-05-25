// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {

namespace {
constexpr std::string_view g_defaultLibName{"npu_mlir_runtime"};

std::string g_selectedLibName{g_defaultLibName};
std::mutex g_runtimeApiMutex;
std::unordered_map<std::string, std::shared_ptr<NPUVMRuntimeApi>> g_runtimeApis;

std::string resolveLibName(std::string_view libName) {
    return libName.empty() ? std::string(g_defaultLibName) : std::string(libName);
}

std::string resolveLibNameFromBlob(const void* data, size_t size) {
    const size_t headerSize = std::min(size, size_t{20});
    const std::string_view header(static_cast<const char*>(data), headerSize);
    return (header.find("NPUByte\x00") != std::string_view::npos) ? "npu_interpreter_runtime"
                                                                    : std::string(g_defaultLibName);
}

std::shared_ptr<NPUVMRuntimeApi> getOrCreateRuntimeApi(const std::string& libName) {
    const auto found = g_runtimeApis.find(libName);
    if (found != g_runtimeApis.end()) {
        return found->second;
    }

    auto instance = std::make_shared<NPUVMRuntimeApi>(libName);
    const auto [inserted, _] = g_runtimeApis.emplace(libName, std::move(instance));
    return inserted->second;
}
}  // namespace

NPUVMRuntimeApi::NPUVMRuntimeApi(std::string_view libName) {
    const std::string baseName = libName.empty() ? "npu_mlir_runtime" : std::string(libName);
    try {
        auto libPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
        this->lib = ov::util::load_shared_object(libPath);
    } catch (const std::runtime_error& error) {
        OPENVINO_THROW(error.what());
    }

    try {
#define nmr_symbol_statement(symbol) \
    this->symbol = reinterpret_cast<decltype(&::symbol)>(ov::util::get_symbol(lib, #symbol));
        nmr_symbols_list();
#undef nmr_symbol_statement
    } catch (const std::runtime_error& error) {
        OPENVINO_THROW(error.what());
    }

#define nmr_symbol_statement(symbol)                                                              \
    try {                                                                                         \
        this->symbol = reinterpret_cast<decltype(&::symbol)>(ov::util::get_symbol(lib, #symbol)); \
    } catch (const std::runtime_error&) {                                                         \
        this->symbol = nullptr;                                                                   \
    }
#undef nmr_symbol_statement

#define nmr_symbol_statement(symbol) symbol = this->symbol;
    nmr_symbols_list();
#undef nmr_symbol_statement
}

void NPUVMRuntimeApi::initializeFromBlob(const void* data, size_t size) {
    initialize(resolveLibNameFromBlob(data, size));
}

void NPUVMRuntimeApi::initialize(std::string_view libName) {
    const std::string resolvedName = resolveLibName(libName);
    std::lock_guard<std::mutex> lock(g_runtimeApiMutex);
    g_selectedLibName = resolvedName;
    (void)getOrCreateRuntimeApi(resolvedName);
}

std::shared_ptr<NPUVMRuntimeApi> NPUVMRuntimeApi::getInstance() {
    std::lock_guard<std::mutex> lock(g_runtimeApiMutex);
    return getOrCreateRuntimeApi(g_selectedLibName);
}

std::shared_ptr<NPUVMRuntimeApi> NPUVMRuntimeApi::getInstance(std::string_view libName) {
    const std::string resolvedName = resolveLibName(libName);
    std::lock_guard<std::mutex> lock(g_runtimeApiMutex);
    return getOrCreateRuntimeApi(resolvedName);
}

std::shared_ptr<NPUVMRuntimeApi> NPUVMRuntimeApi::getInstanceFromBlob(const void* data, size_t size) {
    return getInstance(resolveLibNameFromBlob(data, size));
}

}  // namespace intel_npu
