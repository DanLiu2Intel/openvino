# OpenVINO™ AUTO 插件详解

## 概述

AUTO 插件是 OpenVINO™ 中的一个特殊插件，它与其他硬件插件（如 CPU、GPU、NPU）的关系和工作方式可能会让初学者感到困惑。本文档旨在回答关于 AUTO 插件的常见问题。

## 常见问题解答（FAQ）

### 1. AUTO 插件和其他插件有什么关系？

**答案：** AUTO 插件是一个"元插件"（meta plugin）或"虚拟插件"（virtual plugin），它本身**不直接执行**任何硬件操作。相反，它充当一个智能调度器和管理器，负责：

- **自动选择**最适合的硬件设备（CPU、GPU、NPU 等）
- **委托**实际的推理工作给选定的硬件插件
- **协调**多个设备之间的工作负载

```
应用程序
    ↓
AUTO 插件（决策层）
    ↓
    ├─→ CPU 插件（实际执行推理）
    ├─→ GPU 插件（实际执行推理）
    └─→ NPU 插件（实际执行推理）
```

### 2. 为什么 AUTO 插件很多函数都没有实现？

**答案：** 这是一个常见的误解。当你在源代码中看到类似 `OPENVINO_NOT_IMPLEMENTED` 的函数时，这并不意味着 AUTO 插件"不完整"。原因如下：

#### 不需要实现的函数

某些函数对于 AUTO 插件来说**不需要实现**，因为：

1. **create_context()** / **get_default_context()**: 
   - 这些函数用于创建设备特定的上下文（如 GPU 上下文）
   - AUTO 不直接操作硬件，所以不需要创建自己的上下文
   - 当需要上下文时，AUTO 会将请求转发给实际的硬件插件

2. **import_model()**: 
   - 模型导入是设备特定的操作
   - AUTO 会委托给目标设备的插件来处理

#### 已实现的关键函数

AUTO 插件**确实实现**了以下核心功能：

- **compile_model()**: 将模型编译到选定的设备
- **query_model()**: 查询哪些操作可以在哪些设备上运行
- **set_property()** / **get_property()**: 配置和查询插件属性
- **select_device()**: 智能选择最佳设备
- **get_valid_device()**: 获取可用的设备列表

### 3. AUTO 插件是怎么使用的？

**答案：** AUTO 插件的使用非常简单，它会自动处理大部分复杂的决策。

#### 基本使用方式

```cpp
// C++ 示例
#include <openvino/openvino.hpp>

ov::Core core;
auto model = core.read_model("model.xml");

// 方法 1: 使用 AUTO，让它自动选择最佳设备
auto compiled_model = core.compile_model(model, "AUTO");

// 方法 2: 指定候选设备列表（优先级从左到右）
auto compiled_model = core.compile_model(model, "AUTO:GPU,CPU");
```

```python
# Python 示例
import openvino as ov

core = ov.Core()
model = core.read_model("model.xml")

# 方法 1: 使用 AUTO
compiled_model = core.compile_model(model, "AUTO")

# 方法 2: 指定候选设备
compiled_model = core.compile_model(model, "AUTO:GPU,CPU")
```

#### AUTO 插件的工作流程

1. **设备发现**: AUTO 插件检测系统中所有可用的硬件设备
2. **设备选择**: 根据以下因素选择最佳设备：
   - 性能提示（LATENCY、THROUGHPUT、CUMULATIVE_THROUGHPUT）
   - 设备优先级列表
   - 模型精度（FP32、FP16、INT8）
   - 设备能力
3. **模型编译**: 将模型编译到选定的设备
4. **CPU 加速启动**: 
   - 如果目标设备编译较慢（如 GPU），AUTO 会先在 CPU 上启动推理
   - 同时在后台编译 GPU 版本
   - 编译完成后自动切换到 GPU
5. **运行时故障转移**: 如果某个设备失败，AUTO 可以自动切换到备用设备

## AUTO 插件的架构

### 核心组件

```
Plugin (plugin.cpp)
    ├─ 设备管理
    │   ├─ parse_meta_devices()    # 解析设备列表
    │   ├─ get_valid_device()       # 获取可用设备
    │   └─ select_device()          # 选择最佳设备
    │
    ├─ 模型编译
    │   ├─ compile_model()          # 编译模型
    │   └─ compile_model_impl()     # 实现细节
    │
    └─ 编译模型类型
        ├─ AutoCompiledModel        # LATENCY/THROUGHPUT 模式
        └─ CumulativeCompiledModel  # CUMULATIVE_THROUGHPUT 模式

Schedule (schedule.cpp)
    ├─ AutoSchedule                 # 单设备调度
    └─ CumulativeSchedule           # 多设备调度

CompiledModel
    ├─ create_infer_request()       # 创建推理请求
    └─ get_property()               # 查询编译模型属性
```

### 设计模式

AUTO 插件使用了以下设计模式：

1. **代理模式（Proxy Pattern）**: AUTO 作为代理，将请求转发给实际的硬件插件
2. **策略模式（Strategy Pattern）**: 根据性能提示选择不同的调度策略
3. **工厂模式（Factory Pattern）**: 根据配置创建不同类型的编译模型

## 高级特性

### 1. 加速首次推理延迟（FIL）

当 AUTO 选择 GPU 等编译较慢的设备时：
- CPU 立即开始推理（低延迟）
- GPU 在后台并行编译
- 编译完成后切换到 GPU

可通过 `ov::intel_auto::enable_startup_fallback` 控制此功能。

### 2. 多设备推理

使用 `CUMULATIVE_THROUGHPUT` 性能提示：
- AUTO 将模型加载到多个设备
- 推理请求分配到多个设备以提高吞吐量

### 3. 运行时故障转移

如果某个设备推理失败：
- AUTO 自动选择备用设备
- 重新发送推理请求
- 对应用程序透明

可通过 `ov::intel_auto::enable_runtime_fallback` 控制此功能。

### 4. 性能提示

```cpp
// 延迟优化（选择最快的单个设备）
compiled_model = core.compile_model(model, "AUTO", 
    ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

// 吞吐量优化（优化单设备吞吐量）
compiled_model = core.compile_model(model, "AUTO",
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));

// 累积吞吐量（使用多个设备）
compiled_model = core.compile_model(model, "AUTO",
    ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
```

## 设备选择逻辑

AUTO 插件按以下优先级选择设备：

| 优先级 | 设备类型 | 支持的精度 |
|-------|---------|-----------|
| 1 | 独立 GPU (dGPU) | FP32, FP16, INT8, BIN |
| 2 | 集成 GPU (iGPU) | FP32, FP16, BIN |
| 3 | CPU | FP32, FP16, INT8, BIN |
| 4 | NPU | (需要显式指定) |

**注意**: NPU 当前默认不在优先级列表中，需要显式指定。

## 调试和日志

启用详细日志：

```cpp
// C++
core.set_property("AUTO", ov::log::level(ov::log::Level::DEBUG));

// 或使用环境变量
// Linux: export OPENVINO_LOG_LEVEL=4
// Windows: set OPENVINO_LOG_LEVEL=4
```

日志级别：
- 0: NO
- 1: ERR
- 2: WARNING
- 3: INFO
- 4: DEBUG
- 5: TRACE

## 最佳实践

1. **默认使用 AUTO**: 对于大多数应用，直接使用 "AUTO" 即可
2. **指定设备列表**: 如果需要排除某些设备，使用 "AUTO:GPU,CPU" 格式
3. **选择性能提示**: 根据应用需求设置 LATENCY、THROUGHPUT 或 CUMULATIVE_THROUGHPUT
4. **测试精度时**: 禁用 CPU 启动加速（`enable_startup_fallback=false`），以避免 CPU 和 GPU 之间的精度差异
5. **生产环境**: 如果已知最佳设备，考虑直接指定设备以减少开销

## 与 MULTI 插件的关系

OpenVINO 还有一个 MULTI 插件，它与 AUTO 类似但有区别：

- **AUTO**: 自动选择**最佳单个设备**（默认），或使用 CUMULATIVE_THROUGHPUT 时使用多设备
- **MULTI**: 总是在**指定的多个设备**上并行运行

```cpp
// AUTO: 自动选择一个最佳设备
compiled_model = core.compile_model(model, "AUTO");

// MULTI: 同时在 GPU 和 CPU 上运行
compiled_model = core.compile_model(model, "MULTI:GPU,CPU");
```

## 总结

AUTO 插件的核心价值在于：

1. **简化开发**: 一次编码，到处部署
2. **智能选择**: 自动选择最佳硬件
3. **透明优化**: 自动应用各种优化策略
4. **灵活配置**: 支持多种使用场景

AUTO 插件不是一个"不完整"的插件，而是一个精心设计的智能调度层，它通过委托给实际的硬件插件来完成推理工作。

## 参考资料

- [AUTO Plugin README (English)](../README.md)
- [AUTO Plugin Architecture (English)](./architecture.md)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [Debugging Auto-Device Plugin](../../../../docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection/debugging-auto-device.rst)
