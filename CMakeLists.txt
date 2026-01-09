# cmake
cmake_minimum_required(VERSION 3.22)
project(HandTrackingService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Explicitly set CUDA compiler path for Jetson
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
set(ENV{CUDACXX} "${CMAKE_CUDA_COMPILER}")

enable_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
endif()

find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    set(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv4")
    find_package(OpenCV REQUIRED)
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBLO REQUIRED liblo)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBLO REQUIRED liblo)

# Network: Always use Tailscale IP 100.101.16.21 for the Jetson.

# CUDA Support
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found.")
else()
    message(WARNING "CUDA Toolkit not found. Zero-copy optimizations will be disabled.")
endif()

# TensorRT Support (for V3 inference on Jetson)
find_library(NVINFER_LIB nvinfer PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib)
find_library(NVONNXPARSER_LIB nvonnxparser PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include/aarch64-linux-gnu /usr/local/include)

if(NVINFER_LIB AND NVONNXPARSER_LIB AND TENSORRT_INCLUDE_DIR)
    message(STATUS "TensorRT found: ${NVINFER_LIB}")
    set(TENSORRT_FOUND TRUE)
else()
    message(WARNING "TensorRT not found. V3 inference will be disabled.")
    set(TENSORRT_FOUND FALSE)
endif()

# Shared Libraries erzwingen, bevor Pakete geladen werden
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
set(DEPTHAI_BUILD_SHARED_LIBS ON CACHE BOOL "Build depthai shared libraries" FORCE)

# 1. Versuch: DepthAI via Config-Mode finden (bevorzugt, da es Abhängigkeiten kennt)
find_package(depthai CONFIG QUIET)

if(depthai_FOUND)
    message(STATUS "DepthAI via Config gefunden: ${depthai_VERSION}")
    set(DEPTHAI_TARGET depthai::core)
else()
    message(STATUS "DepthAI Config nicht gefunden, falle zurück auf manuelle Suche...")

    # Cache bereinigen, um sicherzustellen, dass wir nicht an einer alten .a hängen bleiben
    unset(DEPTHAI_LIBRARY CACHE)
    unset(DEPTHAI_INCLUDE_DIR CACHE)

    # Erzwinge Suche nach .so (Shared Object)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")

    find_library(DEPTHAI_LIBRARY
        NAMES depthai-core libdepthai-core
        PATHS
            /usr/local/lib
            /usr/lib
            /home/nvidia/depthai-core/build/vcpkg_installed/arm64-linux/lib
        NO_DEFAULT_PATH
    )

    # Sicherheitsnetz: Abbruch wenn statische Lib gefunden wurde
    if(DEPTHAI_LIBRARY MATCHES "\\.a$")
        message(FATAL_ERROR "CRITICAL: CMake hat statische Lib gefunden: ${DEPTHAI_LIBRARY}. Wir brauchen .so! Bitte .a auf dem Jetson löschen: sudo rm /usr/local/lib/libdepthai-core.a")
    endif()

    # Fallback Suche (aber immer noch strikt nach .so durch CMAKE_FIND_LIBRARY_SUFFIXES)
    if(NOT DEPTHAI_LIBRARY)
        find_library(DEPTHAI_LIBRARY NAMES depthai-core libdepthai-core)
    endif()

    if(DEPTHAI_LIBRARY)
        message(STATUS "DepthAI Shared Library gefunden: ${DEPTHAI_LIBRARY}")

        # Header separat suchen, falls nicht durch Config gefunden
        find_path(DEPTHAI_INCLUDE_DIR
            NAMES depthai/depthai.hpp
            PATHS
                /usr/local/include
                /usr/include
                /home/nvidia/depthai-core/include
        )

        if (DEPTHAI_INCLUDE_DIR)
             message(STATUS "DepthAI Header gefunden: ${DEPTHAI_INCLUDE_DIR}")
             add_library(depthai::core SHARED IMPORTED
                     include/core/ProcessingLoop.hpp)
             set_target_properties(depthai::core PROPERTIES
                IMPORTED_LOCATION "${DEPTHAI_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${DEPTHAI_INCLUDE_DIR}"
            )
            set(DEPTHAI_TARGET depthai::core)
        else()
             message(FATAL_ERROR "DepthAI Library gefunden, aber Header fehlen.")
        endif()
    else()
        message(FATAL_ERROR "Konnte 'depthai-core' (shared library .so) nicht finden. Bitte sicherstellen, dass sie mit BUILD_SHARED_LIBS=ON gebaut wurde.")
    endif()
endif()

# System-Threads und dl portabel
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(HandTrackingService src/main.cpp
        src/core/StereoKernel.cu
        src/core/Logger.cpp
        src/core/PipelineManager.cpp
        src/core/InputLoop.cpp
        src/core/ProcessingLoop.cpp
        src/core/SystemMonitor.cpp
        src/core/HandTracker.cpp
        src/core/GestureFSM.cpp
        src/core/StereoDepth.cpp
        src/inference/TensorRTEngine.cpp
        src/inference/PalmDetector.cpp
        src/inference/HandLandmark.cpp
        src/math/Filters.cpp
        src/net/OscSender.cpp
        src/net/MjpegServer.cpp
        include/core/Logger.hpp
        include/core/StereoKernel.hpp
        include/core/SpscQueue.hpp
        include/core/PipelineManager.hpp
        include/core/MemoryUtils.hpp
        include/core/Types.hpp
        include/core/Frame.hpp
        include/core/FramePool.hpp
        include/core/InputLoop.hpp
        include/core/ProcessingLoop.hpp
        include/core/SystemMonitor.hpp
        include/core/HandTracker.hpp
        include/core/GestureFSM.hpp
        include/core/StereoDepth.hpp
        include/inference/TensorRTEngine.hpp
        include/inference/PalmDetector.hpp
        include/inference/HandLandmark.hpp
        include/math/Filters.hpp
        include/net/OscSender.hpp
        include/net/MjpegServer.hpp)

target_include_directories(HandTrackingService PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${LIBLO_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
)

# Linken
target_link_libraries(HandTrackingService PRIVATE
        ${DEPTHAI_TARGET}
        ${OpenCV_LIBS}
        ${LIBLO_LIBRARIES}
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

if(CUDAToolkit_FOUND)
    target_compile_definitions(HandTrackingService PRIVATE ENABLE_CUDA)
    target_link_libraries(HandTrackingService PRIVATE CUDA::cudart CUDA::nppicc)
endif()

if(TENSORRT_FOUND)
    target_compile_definitions(HandTrackingService PRIVATE ENABLE_TENSORRT)
    target_link_libraries(HandTrackingService PRIVATE ${NVINFER_LIB} ${NVONNXPARSER_LIB})
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_compile_options(HandTrackingService PRIVATE -O3 -march=native -mtune=native -Wno-psabi)
else()
    target_compile_options(HandTrackingService PRIVATE -O3)
endif()

# Copy models directory to the build directory so the executable can find them
add_custom_command(TARGET HandTrackingService POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    $<TARGET_FILE_DIR:HandTrackingService>/models
    COMMENT "Copying models directory to build output..."
)

# Pre-build: Kill old service to free OAK-D connection (only on Jetson)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_custom_command(TARGET HandTrackingService PRE_BUILD
        COMMAND pkill -SIGTERM HandTrackingService 2>/dev/null || true
        COMMAND sleep 1
        COMMAND pkill -9 HandTrackingService 2>/dev/null || true
        COMMENT "Stopping old HandTrackingService to free OAK-D..."
    )
endif()

