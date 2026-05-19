cmake_minimum_required(VERSION 3.8)
project(tensorrt_cpp_api)

# Speed up rebuilds with ccache when available
include(cmake/ccache.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations")
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -DNDEBUG")
endif()

# Allow FindTensorRT.cmake to be located
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TensorRT install location. Can be overridden by:
#   - setting TensorRT_DIR on the command line: -DTensorRT_DIR=/path/to/TensorRT
#   - exporting the TENSORRT_DIR environment variable
# DEB packages install to /usr/src/tensorrt (default). TAR packages live wherever they were unpacked.
if(NOT DEFINED TensorRT_DIR)
    if(DEFINED ENV{TENSORRT_DIR})
        set(TensorRT_DIR "$ENV{TENSORRT_DIR}")
    else()
        set(TensorRT_DIR "/usr/src/tensorrt")
    endif()
endif()

if(NOT EXISTS "${TensorRT_DIR}")
    message(FATAL_ERROR "TensorRT_DIR='${TensorRT_DIR}' does not exist. Install TensorRT or set "
        "TensorRT_DIR / TENSORRT_DIR to its root directory.")
endif()

# CUDA toolkit. Override with -DCUDA_VERSION=<x.y> or CUDA_TOOLKIT_ROOT_DIR if needed.
if(NOT DEFINED CUDA_VERSION)
    set(CUDA_VERSION "11.8")
endif()
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${CUDA_VERSION}")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-${CUDA_VERSION}/bin/nvcc")

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV 4.8.0 REQUIRED)

message(STATUS "TensorRT_DIR: ${TensorRT_DIR}")
message(STATUS "TensorRT include: ${TensorRT_INCLUDE_DIRS}")
message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_library(tensorrt_cpp_api SHARED src/engine.cpp)
target_include_directories(tensorrt_cpp_api PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_api PUBLIC
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${TensorRT_LIBRARIES})

# Standalone inference benchmark (not used by the ROS node)
add_executable(run_inference_benchmark src/main.cpp)
target_link_libraries(run_inference_benchmark tensorrt_cpp_api)
