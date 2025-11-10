// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "driver_compiler_adapter.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "plugin_compiler_adapter.hpp"

namespace intel_npu {

// class CompilerAdapterFactory final {
// public:
//     std::unique_ptr<ICompilerAdapter> getCompiler(const ov::SoPtr<IEngineBackend>& engineBackend,
//                                                   const ov::intel_npu::CompilerType type) const {
//         switch (type) {
//         case ov::intel_npu::CompilerType::MLIR: {
//             if (engineBackend == nullptr || engineBackend->getName() != "LEVEL0") {
//                 return std::make_unique<PluginCompilerAdapter>(nullptr);
//             }
//             return std::make_unique<PluginCompilerAdapter>(engineBackend->getInitStructs());
//         }
//         case ov::intel_npu::CompilerType::DRIVER: {
//             if (engineBackend == nullptr || engineBackend->getName() != "LEVEL0") {
//                 OPENVINO_THROW("NPU Compiler Adapter must be used with LEVEL0 backend");
//             }
//             return std::make_unique<DriverCompilerAdapter>(engineBackend->getInitStructs());
//         }
//         default:
//             OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
//         }
//     }
// };

class CompilerAdapterFactory final {
public:
    std::unique_ptr<ICompilerAdapter> getCompiler(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                  const ov::intel_npu::CompilerType type) const {
        // AUTO mode: try in order - Driver -> MLIR Plugin -> VCL
        if (type == ov::intel_npu::CompilerType::AUTO) {
            return getCompilerAuto(engineBackend);
        }
        
        switch (type) {
        case ov::intel_npu::CompilerType::MLIR:
            return createPluginCompiler(engineBackend);
        case ov::intel_npu::CompilerType::DRIVER:
            return createDriverCompiler(engineBackend);
        default:
            OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
        }
    }

private:
    std::unique_ptr<ICompilerAdapter> getCompilerAuto(const ov::SoPtr<IEngineBackend>& engineBackend) const {
        Logger logger("CompilerAdapterFactory", Logger::global().level()); // 可能不会有log输出
        
        // 1. Try DriverCompilerAdapter first
        logger.info("Attempting to create DriverCompilerAdapter...");
        try {
            auto driverAdapter = createDriverCompiler(engineBackend);
            if (driverAdapter && validateCompiler(driverAdapter.get(), "DriverCompilerAdapter")) {
                logger.info("DriverCompilerAdapter created successfully");
                return driverAdapter;
            }
        } catch (const std::exception& e) {
            logger.warning("DriverCompilerAdapter creation failed: %s", e.what());
        }
        
        // 2. Try PluginCompilerAdapter (MLIR)
        logger.info("Attempting to create PluginCompilerAdapter with MLIR...");
        try {
            auto pluginAdapter = createPluginCompiler(engineBackend, false);  // false = MLIR
            if (pluginAdapter && validateCompiler(pluginAdapter.get(), "PluginCompilerAdapter")) {
                logger.info("PluginCompilerAdapter (MLIR) created successfully");
                return pluginAdapter;
            }
        } catch (const std::exception& e) {
            logger.warning("PluginCompilerAdapter (MLIR) creation failed: %s", e.what());
        }
        
        // 3. Try PluginCompilerAdapter (VCL)
        logger.info("Attempting to create PluginCompilerAdapter with VCL...");
        try {
            auto vclAdapter = createPluginCompiler(engineBackend, true);  // true = VCL
            if (vclAdapter && validateCompiler(vclAdapter.get()), "PluginCompilerAdapter") {
                logger.info("PluginCompilerAdapter (VCL) created successfully");
                return vclAdapter;
            }
        } catch (const std::exception& e) {
            logger.warning("PluginCompilerAdapter (VCL) creation failed: %s", e.what());
        }
        
        OPENVINO_THROW("Failed to create any compiler adapter");
    }
    
    std::unique_ptr<ICompilerAdapter> createDriverCompiler(const ov::SoPtr<IEngineBackend>& engineBackend) const {
        if (engineBackend == nullptr || engineBackend->getName() != "LEVEL0") {
            throw std::runtime_error("DriverCompilerAdapter requires LEVEL0 backend");
        }
        return std::make_unique<DriverCompilerAdapter>(engineBackend->getInitStructs());
    }
    
    std::unique_ptr<ICompilerAdapter> createPluginCompiler(const ov::SoPtr<IEngineBackend>& engineBackend, 
                                                           bool forceVCL = false) const {
        std::shared_ptr<ZeroInitStructsHolder> zeroInitStruct = nullptr;
        if (engineBackend != nullptr && engineBackend->getName() == "LEVEL0") {
            zeroInitStruct = engineBackend->getInitStructs();
        }
        return std::make_unique<PluginCompilerAdapter>(zeroInitStruct, forceVCL);
    }
    
     /**
     * @brief compiler validation function
     * @param adapter compiler adapter pointer
     * @param adapterName compiler name for log
     * @return check result
     */
    bool validateCompiler(ICompilerAdapter* adapter, const std::string& adapterName = "") const {
        if (adapter == nullptr) {
            return false;
        }
        
        try {
            // checker compiler version ///对于vcl中一定会有吗？
            uint32_t version = adapter->get_version();
            if (version == 0) {
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            if (!adapterName.empty()) {
                Logger logger("CompilerAdapterFactory", Logger::global().level());
                logger.debug("Validation failed for %s: %s", adapterName.c_str(), e.what());
            }
            return false;
        } catch (...) {
            return false;
        }
    }
};

}  // namespace intel_npu
