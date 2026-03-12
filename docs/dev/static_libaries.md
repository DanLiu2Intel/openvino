# Building OpenVINO static libraries

## Contents

- [Introduction](#introduction)
- [The OPENVINO_STATIC_LIBRARY macro](#the-openvino_static_library-macro)
- [System requirements](#system-requirements)
- [Configure OpenVINO runtime in CMake stage](#configure-openvino-runtime-in-cmake-stage)
- [Build static OpenVINO libraries](#build-static-openvino-libraries)
- [Link static OpenVINO runtime](#link-static-openvino-runtime)
- [Static OpenVINO libraries + Conditional compilation for particular models](#static-openvino-libraries--conditional-compilation-for-particular-models)
- [Building with static MSVC Runtime](#building-with-static-msvc-runtime)
- [Limitations](#limitations)
- [See also](#see-also)

## Introduction

Building static OpenVINO Runtime libraries allows to additionally reduce the size of a binary when it is used together with conditional compilation.
It is possible because not all interface symbols of OpenVINO Runtime libraries are exported to end users during a static build and can be removed by linker. See [Static OpenVINO libraries + Conditional compilation for particular models](#static-openvino-libraries--conditional-compilation-for-particular-models)

## The OPENVINO_STATIC_LIBRARY macro

### Overview

`OPENVINO_STATIC_LIBRARY` is a preprocessor macro that plays a critical role throughout the OpenVINO project. It controls the behavior of symbol visibility, API exports/imports, and plugin/frontend loading mechanisms when building or using OpenVINO as a static library.

### When is it defined?

The macro is automatically defined by the CMake build system when building OpenVINO in static library mode:

```cmake
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()
```

This occurs in multiple CMakeLists.txt files across the project, including:
- `src/cmake/openvino.cmake` - Main OpenVINO library
- `src/inference/CMakeLists.txt` - Runtime components
- `src/core/CMakeLists.txt` - Core library
- `src/frontends/common/CMakeLists.txt` - Frontend infrastructure
- Various plugin CMakeLists.txt files

When users build OpenVINO with the CMake option `-DBUILD_SHARED_LIBS=OFF`, all these targets automatically get the `OPENVINO_STATIC_LIBRARY` definition as a PUBLIC compile definition, which means it propagates to any code that links against these libraries.

### What does it affect?

#### 1. Symbol Visibility and API Exports/Imports

In shared library builds, OpenVINO uses platform-specific mechanisms to control which symbols are exported:
- **Windows**: `__declspec(dllexport)` when building, `__declspec(dllimport)` when using
- **Linux/GCC**: `__attribute__((visibility("default")))`

When `OPENVINO_STATIC_LIBRARY` is defined, these export/import declarations are removed, as static libraries don't require symbol visibility control. This is implemented in various visibility headers:

**Core API** (`src/core/include/openvino/core/core_visibility.hpp`):
```cpp
#ifdef OPENVINO_STATIC_LIBRARY
#    define OPENVINO_API                // Empty - no export/import needed
#    define OPENVINO_API_C(...) __VA_ARGS__
#else
#    ifdef IMPLEMENT_OPENVINO_API
#        define OPENVINO_API        OPENVINO_CORE_EXPORTS  // __declspec(dllexport) or visibility("default")
#        define OPENVINO_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS __VA_ARGS__ OPENVINO_CDECL
#    else
#        define OPENVINO_API        OPENVINO_CORE_IMPORTS  // __declspec(dllimport)
#        define OPENVINO_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS __VA_ARGS__ OPENVINO_CDECL
#    endif
#endif
```

**Runtime API** (`src/inference/include/openvino/runtime/common.hpp`):
```cpp
#if defined(OPENVINO_STATIC_LIBRARY) || defined(USE_STATIC_IE)
#    define OPENVINO_RUNTIME_API_C(...) OPENVINO_EXTERN_C __VA_ARGS__
#    define OPENVINO_RUNTIME_API
#else
    // DLL export/import declarations
#endif
```

**Transformations API** (`src/common/transformations/include/transformations_visibility.hpp`):
```cpp
#ifdef OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#else
    // DLL export/import declarations
#endif
```

**Frontend API** (`src/frontends/common/include/openvino/frontend/visibility.hpp`):
```cpp
#ifdef OPENVINO_STATIC_LIBRARY
#    define FRONTEND_API
#    define FRONTEND_C_API
#else
    // DLL export/import declarations
#endif
```

**C API** (`src/bindings/c/include/openvino/c/ov_common.h`):
```cpp
#if defined(OPENVINO_STATIC_LIBRARY) || defined(__GNUC__) && (__GNUC__ < 4)
#    define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __VA_ARGS__
#else
    // Platform-specific DLL export/import
#endif
```

#### 2. Plugin and Frontend Loading Mechanisms

The macro fundamentally changes how plugins and frontends are loaded:

**Dynamic Library Mode** (default):
- Plugins and frontends are loaded at runtime from separate shared library files (.dll/.so)
- The system searches for plugin files in the filesystem
- Plugin paths are stored in the registry

**Static Library Mode** (`OPENVINO_STATIC_LIBRARY` defined):
- All plugins and frontends are compiled directly into the application
- No dynamic loading occurs
- Plugin creation functions are called directly through function pointers
- Registration happens via static registry at program startup

This is evident in `src/inference/src/dev/core_impl.cpp`:
```cpp
#ifdef OPENVINO_STATIC_LIBRARY
    // In static build: use function pointers to create plugins
    PluginDescriptor desc{value.m_create_plugin_func, config, value.m_create_extensions_func};
    register_plugin_in_registry_unsafe(device_name, desc);
#else
    // In dynamic build: use plugin file paths
    const auto& plugin_path = ov::util::get_compiled_plugin_path(...);
    PluginDescriptor desc{plugin_path, config};
    register_plugin_in_registry_unsafe(device_name, desc);
#endif
```

And in `src/frontends/common/src/plugin_loader.cpp`:
```cpp
#ifdef OPENVINO_STATIC_LIBRARY
    // Load frontends from static registry
    for (const auto& frontend : getStaticFrontendsRegistry()) {
        // Direct function calls to create frontends
    }
#else
    // Load frontends from shared library files
#endif
```

#### 3. Plugin Registry Structure

The plugin registry structure itself changes based on this macro (`cmake/developer_package/plugins/plugins.hpp.in`):

**Static Mode**:
```cpp
struct Value {
    CreatePluginEngineFunc * m_create_plugin_func;      // Function pointer
    CreateExtensionFunc * m_create_extensions_func;     // Function pointer
    std::map<std::string, std::string> m_default_config;
};
```

**Dynamic Mode**:
```cpp
struct Value {
    std::string m_plugin_path;                          // File path to plugin
    std::map<std::string, std::string> m_default_config;
};
```

### Usage in User Code

When linking against static OpenVINO libraries, users' code automatically receives the `OPENVINO_STATIC_LIBRARY` definition because it's defined as a `PUBLIC` compile definition. This ensures that:

1. Header files correctly define API macros without export/import attributes
2. The application code matches the library's expectations
3. No symbol visibility mismatches occur between library and application

Users typically don't need to manually define this macro; it's handled automatically by CMake when using `find_package(OpenVINO)` with static libraries.

### Summary

In summary, `OPENVINO_STATIC_LIBRARY` serves as the central control point that:
- **Eliminates** DLL export/import declarations (not needed for static libraries)
- **Switches** plugin/frontend loading from dynamic file-based loading to static function pointer-based registration
- **Changes** the internal structure of plugin registries
- **Ensures** consistent compilation across the entire codebase when building statically

This macro is defined automatically when using `-DBUILD_SHARED_LIBS=OFF` and propagates throughout the build system and to user applications linking against OpenVINO static libraries.

## System requirements

* CMake version 3.18 or higher must be used to build static OpenVINO libraries.
* Supported OSes:
    * Windows x64
    * Linux x64
    * All other OSes may work, but have not been explicitly tested

## Configure OpenVINO Runtime in the CMake stage

The default architecture of OpenVINO Runtime assumes that the following components are subject to dynamic loading during execution:
* (Device) Inference backends (CPU, GPU, NPU, MULTI, HETERO, etc.)
* (Model) Frontends (IR, ONNX, PDPD, TF, JAX, etc.)

With the static OpenVINO Runtime, all these modules should be linked into a final user application and **the list of modules/configuration must be known for the CMake configuration stage**. To minimize the total binary size, you can explicitly turn `OFF` unnecessary components. Use [[CMake Options for Custom Compilation|CMakeOptionsForCustomCompilation ]] as a reference for OpenVINO CMake configuration.

For example, to enable only IR v11 reading and CPU inference capabilities, use:
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
      -DENABLE_INTEL_CPU=ON \
      -DENABLE_OV_IR_FRONTEND=ON
```

> **NOTE**: Inference backends located in external repositories can also be used in a static build. Use `-DOPENVINO_EXTRA_MODULES=<path to external plugin root>` to enable them. `OpenVINODeveloperPackage.cmake` must not be used to build external plugins, only `OPENVINO_EXTRA_MODULES` is a working solution.

> **NOTE**: The `ENABLE_LTO` CMake option can also be passed to enable link time optimizations to reduce the binary size. But such property should also be enabled on the target which links with static OpenVINO libraries via `set_target_properties(<target_name> PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)`
-
## Build static OpenVINO libraries

To build OpenVINO Runtime in a static mode, you need to specify the additional CMake option:

```sh
cmake -DBUILD_SHARED_LIBS=OFF <all other CMake options> <openvino_sources root>
```

Then, use the usual CMake 'build' command:

```sh
cmake --build . --target openvino --config Release -j12
```

Then, the installation step:

```sh
cmake -DCMAKE_INSTALL_PREFIX=<install_root> -P cmake_install.cmake
```

The OpenVINO runtime is located in `<install_root>/runtime/lib`

## Link static OpenVINO Runtime

Once you build static OpenVINO Runtime libraries and install them, you can use one of the two ways to add them to your project:

### CMake interface

Just use CMake's `find_package` as usual and link `openvino::runtime`:

```cmake
find_package(OpenVINO REQUIRED)
target_link_libraries(<application> PRIVATE openvino::runtime)
```

`openvino::runtime` transitively adds all other static OpenVINO libraries to a linker command. 

### Pass libraries to linker directly

If you want to configure your project directly, you need to pass all libraries from `<install_root>/runtime/lib` to linker command.

> **NOTE**: Since the proper order of static libraries must be used (dependent library should come **before** dependency in a linker command), consider using the following compiler specific flags to link static OpenVINO libraries:

Microsoft Visual Studio compiler:
```sh
/WHOLEARCHIVE:<ov_library 0> /WHOLEARCHIVE:<ov_library 1> ...
```

GCC like compiler:
```sh
gcc main.cpp -Wl,--whole-archive <all libraries from <root>/runtime/lib> > -Wl,--no-whole-archive -o a.out
```

## Static OpenVINO libraries + Conditional compilation for particular models

OpenVINO Runtime can be compiled for particular models, as shown in the [[Conditional compilation for particular models|ConditionalCompilation]] guide.
The conditional compilation feature can be paired with static OpenVINO libraries to build even smaller end-user applications in terms of binary size. The following procedure can be used, (based on the detailed [[Conditional compilation for particular models|ConditionalCompilation]] guide):

* Build OpenVINO Runtime as usual with the CMake option of `-DSELECTIVE_BUILD=COLLECT`.
* Run target applications on target models and target platforms to collect traces.
* Build the final OpenVINO static Runtime with `-DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv -DBUILD_SHARED_LIBS=OFF`

## Building with static MSVC Runtime

In order to build with static MSVC runtime, use the special [OpenVINO toolchain](https://github.com/openvinotoolkit/openvino/blob/master/cmake/toolchains/mt.runtime.win32.toolchain.cmake) file:

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<openvino source dir>/cmake/toolchains/mt.runtime.win32.toolchain.cmake <other options>
```

> **NOTE**: all other dependent application and libraries must be built with the same `mt.runtime.win32.toolchain.cmake ` toolchain to have conformed values of the `MSVC_RUNTIME_LIBRARY` target property.

## Limitations

* The enabled and tested capabilities of OpenVINO Runtime in a static build:
    * OpenVINO common runtime - work with `ov::Model`, perform model loading on particular device
    * MULTI, HETERO, AUTO, and BATCH inference modes
    * IR, ONNX, PDPD, TF and TF Lite frontends to read `ov::Model`
* Static build support for building static libraries only for OpenVINO Runtime libraries. All other third-party prebuilt dependencies remain in the same format:
    * `TBB` is a shared library; to provide your own TBB build from [[oneTBB source code|https://github.com/oneapi-src/oneTBB]] use `export TBBROOT=<tbb_root>` before OpenVINO CMake scripts are run.

    > **NOTE**: The TBB team does not recommend using oneTBB as a static library, see [[Why onetbb does not like a static library?|https://github.com/oneapi-src/oneTBB/issues/646]]

* `TBBBind_2_5` is not available on Windows x64 during a static OpenVINO build (see description for `ENABLE_TBBBIND_2_5` CMake option [[here|CMakeOptionsForCustomCompilation]] to understand what this library is responsible for). So, capabilities enabled by `TBBBind_2_5` are not available. To enable them, build [[oneTBB from source code|https://github.com/oneapi-src/oneTBB]] and provide the path to built oneTBB artifacts via `TBBROOT` environment variable before OpenVINO CMake scripts are run.

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [How to Build OpenVINO](build.md)
