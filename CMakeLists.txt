cmake_minimum_required(VERSION 3.22)
project(HandTrackingService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# OpenCV Suche
find_package(OpenCV REQUIRED)

find_path(DEPTHAI_INCLUDE_DIR NAMES depthai/depthai.hpp PATHS /usr/local/include)
find_library(DEPTHAI_LIBRARY NAMES libdepthai-core.a PATHS /usr/local/lib)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBLO REQUIRED liblo)

add_executable(HandTrackingService main.cpp)

if(DEPTHAI_INCLUDE_DIR AND DEPTHAI_LIBRARY)
    target_include_directories(HandTrackingService PRIVATE ${DEPTHAI_INCLUDE_DIR})

    target_link_libraries(HandTrackingService PRIVATE
            ${DEPTHAI_LIBRARY}
            ${OpenCV_LIBS}
            ${LIBLO_LIBRARIES}
            pthread
            dl
    )
    message(STATUS "DepthAI Header: ${DEPTHAI_INCLUDE_DIR}")
    message(STATUS "DepthAI Static Lib: ${DEPTHAI_LIBRARY}")
else()
    message(FATAL_ERROR "Konnte libdepthai-core.a nicht finden.")
endif()