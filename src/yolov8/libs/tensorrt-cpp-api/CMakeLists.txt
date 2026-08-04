cmake_minimum_required(VERSION 3.8)
project(tensorrt_cpp_api)

# Vendored from https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP (MIT, see ./LICENSE).
# Kept close to upstream to make future syncs tractable; structural refactors are out of scope
# for this project. The wrapper in ../../src/yolov8.{h,cpp} drives this library from the ROS node.

# Speed up rebuilds with ccache when available
include(cmake/ccache.cmake)

set(CMAKE_CXX_STANDARD 17)
# -Wno-deprecated-declarations silences benign deprecations from the TensorRT/OpenCV headers.
# -Ofast: deliberate for inference workloads; see the comment in ../../CMakeLists.txt.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations")
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -DNDEBUG")
endif()

# Allow FindTensorRT.cmake to be located. Use CURRENT_SOURCE_DIR so it resolves whether this
# CMakeLists is built standalone or as a subdirectory of the yolov8 package (colcon), where
# CMAKE_SOURCE_DIR points at the package root, not this directory.
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TensorRT install location. Can be overridden by:
#   - setting TensorRT_ROOT on the command line: -DTensorRT_ROOT=/path/to/TensorRT
#   - exporting the TENSORRT_DIR environment variable
# DEB packages install to /usr/src/tensorrt (default). TAR packages live wherever they were unpacked.
# NB: the hint is TensorRT_ROOT, not TensorRT_DIR — find_package(TensorRT) reserves TensorRT_DIR for
# its config-file search and pins it to NOTFOUND, which would clobber the hint on every re-configure.
if(NOT DEFINED TensorRT_ROOT)
    if(DEFINED ENV{TENSORRT_DIR})
        set(TensorRT_ROOT "$ENV{TENSORRT_DIR}")
    else()
        set(TensorRT_ROOT "/usr/src/tensorrt")
    endif()
endif()

if(NOT EXISTS "${TensorRT_ROOT}")
    message(FATAL_ERROR "TensorRT_ROOT='${TensorRT_ROOT}' does not exist. Install TensorRT or set "
        "TensorRT_ROOT / TENSORRT_DIR to its root directory.")
endif()

# CUDA toolkit. Override with -DCUDA_VERSION=<x.y> or CUDA_TOOLKIT_ROOT_DIR if needed.
# Pin with -DCUDA_VERSION=<x.y>; otherwise follow the /usr/local/cuda symlink, which tracks
# whatever toolkit is installed (13.x on the AV24 and the dev laptops).
if(DEFINED CUDA_VERSION)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${CUDA_VERSION}")
else()
    # Derive from nvcc on PATH, not the /usr/local/cuda symlink: OpenCV-CUDA pins the exact
    # toolkit it was built against, and a machine can have several installed.
    find_program(_nvcc nvcc)
    if(_nvcc)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${_nvcc}" DIRECTORY)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
    else()
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    endif()
endif()
if(NOT EXISTS "${CUDA_TOOLKIT_ROOT_DIR}")
    message(FATAL_ERROR "CUDA toolkit not found at ${CUDA_TOOLKIT_ROOT_DIR}. "
        "Install it or pass -DCUDA_VERSION=<x.y>.")
endif()
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV 4.8.0 REQUIRED)

message(STATUS "TensorRT_DIR: ${TensorRT_DIR}")
message(STATUS "TensorRT include: ${TensorRT_INCLUDE_DIRS}")
message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
message(STATUS "CUDA toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
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
