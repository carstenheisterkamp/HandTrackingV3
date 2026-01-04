cmake_minimum_required(VERSION 3.22)
project(HandTrackingService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# DepthAI v3 via FetchContent
include(FetchContent)
FetchContent_Declare(
        depthai
        GIT_REPOSITORY https://github.com/luxonis/depthai-core.git
        GIT_TAG v3.2.1
)
set(DEPTHAI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(depthai)

find_package(OpenCV REQUIRED)

add_executable(HandTrackingService main.cpp)
target_link_libraries(HandTrackingService PRIVATE depthai::core ${OpenCV_LIBS})