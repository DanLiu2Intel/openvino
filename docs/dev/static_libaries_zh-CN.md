# 构建 OpenVINO 静态库

## 目录

- [简介](#简介)
- [OPENVINO_STATIC_LIBRARY 宏](#openvino_static_library-宏)
- [系统要求](#系统要求)
- [在 CMake 阶段配置 OpenVINO 运行时](#在-cmake-阶段配置-openvino-运行时)
- [构建静态 OpenVINO 库](#构建静态-openvino-库)
- [链接静态 OpenVINO 运行时](#链接静态-openvino-运行时)
- [静态 OpenVINO 库 + 针对特定模型的条件编译](#静态-openvino-库--针对特定模型的条件编译)
- [使用静态 MSVC 运行时构建](#使用静态-msvc-运行时构建)
- [限制](#限制)
- [另见](#另见)

## 简介

构建静态 OpenVINO 运行时库可以在与条件编译结合使用时进一步减小二进制文件的大小。这是可能的，因为在静态构建期间，并非所有 OpenVINO 运行时库的接口符号都会导出给最终用户，链接器可以删除这些符号。请参阅[静态 OpenVINO 库 + 针对特定模型的条件编译](#静态-openvino-库--针对特定模型的条件编译)

## OPENVINO_STATIC_LIBRARY 宏

### 概述

`OPENVINO_STATIC_LIBRARY` 是一个预处理器宏，在整个 OpenVINO 项目中起着关键作用。它控制着符号可见性、API 导出/导入以及在以静态库形式构建或使用 OpenVINO 时的插件/前端加载机制的行为。

### 何时定义？

该宏由 CMake 构建系统在以静态库模式构建 OpenVINO 时自动定义：

```cmake
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()
```

这发生在项目中的多个 CMakeLists.txt 文件中，包括：
- `src/cmake/openvino.cmake` - 主 OpenVINO 库
- `src/inference/CMakeLists.txt` - 运行时组件
- `src/core/CMakeLists.txt` - 核心库
- `src/frontends/common/CMakeLists.txt` - 前端基础设施
- 各种插件的 CMakeLists.txt 文件

当用户使用 CMake 选项 `-DBUILD_SHARED_LIBS=OFF` 构建 OpenVINO 时，所有这些目标都会自动获得 `OPENVINO_STATIC_LIBRARY` 定义作为 PUBLIC 编译定义，这意味着它会传播到任何链接这些库的代码。

### 它影响什么？

#### 1. 符号可见性和 API 导出/导入

在共享库构建中，OpenVINO 使用平台特定的机制来控制导出哪些符号：
- **Windows**: 构建时使用 `__declspec(dllexport)`，使用时使用 `__declspec(dllimport)`
- **Linux/GCC**: `__attribute__((visibility("default")))`

当定义了 `OPENVINO_STATIC_LIBRARY` 时，这些导出/导入声明会被移除，因为静态库不需要符号可见性控制。这在各种可见性头文件中实现：

**核心 API** (`src/core/include/openvino/core/core_visibility.hpp`):
```cpp
#ifdef OPENVINO_STATIC_LIBRARY
#    define OPENVINO_API                // 空 - 不需要导出/导入
#    define OPENVINO_API_C(...) __VA_ARGS__
#else
#    ifdef IMPLEMENT_OPENVINO_API
#        define OPENVINO_API        OPENVINO_CORE_EXPORTS  // __declspec(dllexport) 或 visibility("default")
#        define OPENVINO_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS __VA_ARGS__ OPENVINO_CDECL
#    else
#        define OPENVINO_API        OPENVINO_CORE_IMPORTS  // __declspec(dllimport)
#        define OPENVINO_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS __VA_ARGS__ OPENVINO_CDECL
#    endif
#endif
```

**运行时 API** (`src/inference/include/openvino/runtime/common.hpp`):
```cpp
#if defined(OPENVINO_STATIC_LIBRARY) || defined(USE_STATIC_IE)
#    define OPENVINO_RUNTIME_API_C(...) OPENVINO_EXTERN_C __VA_ARGS__
#    define OPENVINO_RUNTIME_API
#else
    // DLL 导出/导入声明
#endif
```

**转换 API** (`src/common/transformations/include/transformations_visibility.hpp`):
```cpp
#ifdef OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#else
    // DLL 导出/导入声明
#endif
```

**前端 API** (`src/frontends/common/include/openvino/frontend/visibility.hpp`):
```cpp
#ifdef OPENVINO_STATIC_LIBRARY
#    define FRONTEND_API
#    define FRONTEND_C_API
#else
    // DLL 导出/导入声明
#endif
```

**C API** (`src/bindings/c/include/openvino/c/ov_common.h`):
```cpp
#if defined(OPENVINO_STATIC_LIBRARY) || defined(__GNUC__) && (__GNUC__ < 4)
#    define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __VA_ARGS__
#else
    // 平台特定的 DLL 导出/导入
#endif
```

#### 2. 插件和前端加载机制

该宏从根本上改变了插件和前端的加载方式：

**动态库模式**（默认）：
- 插件和前端在运行时从单独的共享库文件（.dll/.so）加载
- 系统在文件系统中搜索插件文件
- 插件路径存储在注册表中

**静态库模式**（定义了 `OPENVINO_STATIC_LIBRARY`）：
- 所有插件和前端直接编译到应用程序中
- 不发生动态加载
- 通过函数指针直接调用插件创建函数
- 注册在程序启动时通过静态注册表进行

这在 `src/inference/src/dev/core_impl.cpp` 中很明显：
```cpp
#ifdef OPENVINO_STATIC_LIBRARY
    // 在静态构建中：使用函数指针创建插件
    PluginDescriptor desc{value.m_create_plugin_func, config, value.m_create_extensions_func};
    register_plugin_in_registry_unsafe(device_name, desc);
#else
    // 在动态构建中：使用插件文件路径
    const auto& plugin_path = ov::util::get_compiled_plugin_path(...);
    PluginDescriptor desc{plugin_path, config};
    register_plugin_in_registry_unsafe(device_name, desc);
#endif
```

在 `src/frontends/common/src/plugin_loader.cpp` 中：
```cpp
#ifdef OPENVINO_STATIC_LIBRARY
    // 从静态注册表加载前端
    for (const auto& frontend : getStaticFrontendsRegistry()) {
        // 直接函数调用创建前端
    }
#else
    // 从共享库文件加载前端
#endif
```

#### 3. 插件注册表结构

插件注册表结构本身会根据此宏而改变（`cmake/developer_package/plugins/plugins.hpp.in`）：

**静态模式**：
```cpp
struct Value {
    CreatePluginEngineFunc * m_create_plugin_func;      // 函数指针
    CreateExtensionFunc * m_create_extensions_func;     // 函数指针
    std::map<std::string, std::string> m_default_config;
};
```

**动态模式**：
```cpp
struct Value {
    std::string m_plugin_path;                          // 插件文件路径
    std::map<std::string, std::string> m_default_config;
};
```

### 在用户代码中的使用

当链接静态 OpenVINO 库时，用户的代码会自动接收 `OPENVINO_STATIC_LIBRARY` 定义，因为它被定义为 `PUBLIC` 编译定义。这确保了：

1. 头文件正确定义 API 宏而不带导出/导入属性
2. 应用程序代码与库的期望相匹配
3. 库和应用程序之间不会出现符号可见性不匹配

用户通常不需要手动定义此宏；当使用静态库时，CMake 在使用 `find_package(OpenVINO)` 时会自动处理。

### 总结

总之，`OPENVINO_STATIC_LIBRARY` 作为中央控制点：
- **消除** DLL 导出/导入声明（静态库不需要）
- **切换** 插件/前端加载从基于动态文件的加载到基于静态函数指针的注册
- **更改** 插件注册表的内部结构
- **确保** 在静态构建时整个代码库的编译一致性

此宏在使用 `-DBUILD_SHARED_LIBS=OFF` 时自动定义，并在整个构建系统和链接 OpenVINO 静态库的用户应用程序中传播。

## 系统要求

* 必须使用 CMake 3.18 或更高版本来构建静态 OpenVINO 库。
* 支持的操作系统：
    * Windows x64
    * Linux x64
    * 所有其他操作系统可能可以工作，但尚未明确测试

## 在 CMake 阶段配置 OpenVINO 运行时

OpenVINO 运行时的默认架构假设以下组件在执行期间会动态加载：
* （设备）推理后端（CPU、GPU、NPU、MULTI、HETERO 等）
* （模型）前端（IR、ONNX、PDPD、TF、JAX 等）

使用静态 OpenVINO 运行时时，所有这些模块都应链接到最终用户应用程序中，并且**模块/配置列表必须在 CMake 配置阶段已知**。要最小化总二进制文件大小，您可以明确关闭不必要的组件。使用 [[CMake Options for Custom Compilation|CMakeOptionsForCustomCompilation]] 作为 OpenVINO CMake 配置的参考。

例如，要仅启用 IR v11 读取和 CPU 推理功能，使用：
```sh
cmake -DENABLE_INTEL_GPU=OFF \
      -DENABLE_INTEL_NPU=OFF \
      -DENABLE_TEMPLATE=OFF \
      -DENABLE_HETERO=OFF \
      -DENABLE_MULTI=OFF \
      -DENABLE_AUTO=OFF \
      -DENABLE_AUTO_BATCH=OFF \
      -DENABLE_OV_ONNX_FRONTEND=OFF \
      -DENABLE_OV_PADDLE_FRONTEND=OFF \
      -DENABLE_OV_TF_FRONTEND=OFF \
      -DENABLE_OV_TF_LITE_FRONTEND=OFF \
      -DENABLE_OV_JAX_FRONTEND=OFF \
      -DENABLE_OV_PYTORCH_FRONTEND=OFF \
      -DENABLE_OV_JAX_FRONTEND=OFF \
      -DENABLE_INTEL_CPU=ON \
      -DENABLE_OV_IR_FRONTEND=ON
```

> **注意**：位于外部存储库中的推理后端也可以在静态构建中使用。使用 `-DOPENVINO_EXTRA_MODULES=<外部插件根路径>` 来启用它们。不能使用 `OpenVINODeveloperPackage.cmake` 来构建外部插件，只有 `OPENVINO_EXTRA_MODULES` 是有效的解决方案。

> **注意**：还可以传递 `ENABLE_LTO` CMake 选项以启用链接时优化来减少二进制文件大小。但是，这样的属性也应该通过 `set_target_properties(<target_name> PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)` 在链接静态 OpenVINO 库的目标上启用。

## 构建静态 OpenVINO 库

要以静态模式构建 OpenVINO 运行时，您需要指定额外的 CMake 选项：

```sh
cmake -DBUILD_SHARED_LIBS=OFF <所有其他 CMake 选项> <openvino 源代码根目录>
```

然后，使用常规的 CMake 'build' 命令：

```sh
cmake --build . --target openvino --config Release -j12
```

然后，安装步骤：

```sh
cmake -DCMAKE_INSTALL_PREFIX=<安装根目录> -P cmake_install.cmake
```

OpenVINO 运行时位于 `<安装根目录>/runtime/lib`

## 链接静态 OpenVINO 运行时

一旦构建并安装了静态 OpenVINO 运行时库，您可以使用以下两种方法之一将它们添加到您的项目中：

### CMake 接口

像往常一样使用 CMake 的 `find_package` 并链接 `openvino::runtime`：

```cmake
find_package(OpenVINO REQUIRED)
target_link_libraries(<application> PRIVATE openvino::runtime)
```

`openvino::runtime` 传递性地将所有其他静态 OpenVINO 库添加到链接器命令。

### 直接将库传递给链接器

如果您想直接配置项目，您需要将 `<安装根目录>/runtime/lib` 中的所有库传递给链接器命令。

> **注意**：由于必须使用正确的静态库顺序（依赖库应该在链接器命令中出现在依赖项**之前**），请考虑使用以下特定于编译器的标志来链接静态 OpenVINO 库：

Microsoft Visual Studio 编译器：
```sh
/WHOLEARCHIVE:<ov_library 0> /WHOLEARCHIVE:<ov_library 1> ...
```

GCC 类编译器：
```sh
gcc main.cpp -Wl,--whole-archive <所有来自 <root>/runtime/lib 的库> -Wl,--no-whole-archive -o a.out
```

## 静态 OpenVINO 库 + 针对特定模型的条件编译

OpenVINO 运行时可以针对特定模型进行编译，如 [[Conditional compilation for particular models|ConditionalCompilation]] 指南所示。
条件编译功能可以与静态 OpenVINO 库配对使用，以构建在二进制文件大小方面更小的最终用户应用程序。可以使用以下过程（基于详细的 [[Conditional compilation for particular models|ConditionalCompilation]] 指南）：

* 使用 CMake 选项 `-DSELECTIVE_BUILD=COLLECT` 像往常一样构建 OpenVINO 运行时。
* 在目标模型和目标平台上运行目标应用程序以收集跟踪。
* 使用 `-DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv -DBUILD_SHARED_LIBS=OFF` 构建最终的 OpenVINO 静态运行时

## 使用静态 MSVC 运行时构建

为了使用静态 MSVC 运行时构建，使用特殊的 [OpenVINO 工具链](https://github.com/openvinotoolkit/openvino/blob/master/cmake/toolchains/mt.runtime.win32.toolchain.cmake) 文件：

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<openvino 源代码目录>/cmake/toolchains/mt.runtime.win32.toolchain.cmake <其他选项>
```

> **注意**：所有其他依赖应用程序和库都必须使用相同的 `mt.runtime.win32.toolchain.cmake` 工具链构建，以具有 `MSVC_RUNTIME_LIBRARY` 目标属性的一致值。

## 限制

* 在静态构建中启用和测试的 OpenVINO 运行时功能：
    * OpenVINO 通用运行时 - 使用 `ov::Model`，在特定设备上执行模型加载
    * MULTI、HETERO、AUTO 和 BATCH 推理模式
    * IR、ONNX、PDPD、TF 和 TF Lite 前端读取 `ov::Model`
* 静态构建支持仅为 OpenVINO 运行时库构建静态库。所有其他第三方预构建依赖项保持相同格式：
    * `TBB` 是共享库；要从 [[oneTBB 源代码|https://github.com/oneapi-src/oneTBB]] 提供自己的 TBB 构建，在运行 OpenVINO CMake 脚本之前使用 `export TBBROOT=<tbb_root>`。

    > **注意**：TBB 团队不建议将 oneTBB 用作静态库，请参阅 [[Why onetbb does not like a static library?|https://github.com/oneapi-src/oneTBB/issues/646]]

* `TBBBind_2_5` 在静态 OpenVINO 构建期间在 Windows x64 上不可用（请参阅 `ENABLE_TBBBIND_2_5` CMake 选项的说明 [[here|CMakeOptionsForCustomCompilation]] 以了解此库的职责）。因此，`TBBBind_2_5` 启用的功能不可用。要启用它们，从 [[oneTBB 源代码|https://github.com/oneapi-src/oneTBB]] 构建并在运行 OpenVINO CMake 脚本之前通过 `TBBROOT` 环境变量提供构建的 oneTBB 工件的路径。

## 另见

 * [OpenVINO README](../../README.md)
 * [OpenVINO 开发者文档](index.md)
 * [如何构建 OpenVINO](build.md)
