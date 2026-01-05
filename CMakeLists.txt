# cmake
cmake_minimum_required(VERSION 3.22)
project(HandTrackingService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    set(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv4")
    find_package(OpenCV REQUIRED)
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBLO REQUIRED liblo)

find_path(DEPTHAI_INCLUDE_DIR NAMES depthai/depthai.hpp PATHS /usr/local/include)
find_library(DEPTHAI_LIBRARY NAMES depthai-core PATHS /usr/local/lib /usr/lib)

# System-Threads und dl portabel
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(HandTrackingService src/main.cpp)

if(DEPTHAI_INCLUDE_DIR AND DEPTHAI_LIBRARY)
    target_include_directories(HandTrackingService PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            ${DEPTHAI_INCLUDE_DIR}
            ${LIBLO_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(HandTrackingService PRIVATE
            ${DEPTHAI_LIBRARY}
            ${OpenCV_LIBS}
            ${LIBLO_LIBRARIES}
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        target_compile_options(HandTrackingService PRIVATE -O3 -march=native -mtune=native)
    else()
        target_compile_options(HandTrackingService PRIVATE -O3)
    endif()

    message(STATUS "DepthAI Header: ${DEPTHAI_INCLUDE_DIR}")
    message(STATUS "DepthAI Lib: ${DEPTHAI_LIBRARY}")
else()
    message(FATAL_ERROR "Konnte 'depthai-core' nicht finden.")
endif()
