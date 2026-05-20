cmake_minimum_required(VERSION 3.8)
project(yolov8)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CUDA toolkit (override with -DCUDA_VERSION=<x.y>)
if(NOT DEFINED CUDA_VERSION)
    set(CUDA_VERSION "11.8")
endif()
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${CUDA_VERSION}")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-${CUDA_VERSION}/bin/nvcc")

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(yolov8_interfaces REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # We skip:
  #   - copyright: this project ships under a single MIT LICENSE and does not enforce per-file headers.
  #   - cpplint:   the bundled style differs significantly from Google C++ style and reformatting
  #                vendored upstream code would defeat the purpose of vendoring.
  # No unit tests are shipped — the hot path is GPU/TensorRT/OpenCV-with-CUDA coupled and
  # meaningful tests need the target hardware. See README "Design notes".
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Use ccache to speed up rebuilds when available
include(cmake/ccache.cmake)

set(CMAKE_CXX_STANDARD 17)
# -Wno-deprecated-declarations silences benign deprecations from the TensorRT/OpenCV headers we
# do not control. Project code itself stays free of deprecated APIs.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
# -Ofast is deliberate: this is a real-time inference workload, the fast-math relaxations are
# acceptable (we do not rely on strict IEEE NaN/inf semantics on the hot path), and the
# measured speedup over -O3 is non-trivial. -Ofast is skipped under Debug so `make build-debug`
# keeps assertions enabled and produces useful stack traces.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -DNDEBUG")
endif()

# TensorRT inference engine (finds TensorRT/CUDA/OpenCV itself)
add_subdirectory(libs/tensorrt-cpp-api)

# YoloV8 library wraps the engine with detection/segmentation post-processing.
# tensorrt_cpp_api already links OpenCV/CUDA/TensorRT PUBLIC, so we inherit them transitively.
add_library(YoloV8_TRT SHARED src/yolov8.cpp)
target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api)
target_include_directories(YoloV8_TRT PUBLIC libs/tensorrt-cpp-api/src)

# ROS2 node
add_executable(ros_segmentation src/ros_segmentation.cpp)
ament_target_dependencies(ros_segmentation rclcpp sensor_msgs cv_bridge yolov8_interfaces)
target_link_libraries(ros_segmentation YoloV8_TRT)

install(TARGETS ros_segmentation
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS YoloV8_TRT tensorrt_cpp_api
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

# Launch, config, and models directories go into share/yolov8 so the launch files can locate them.
install(DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME}/
  PATTERN ".gitignore" EXCLUDE)

# yolov8.env lives at the workspace root next to this package's directory.
install(FILES ../../yolov8.env
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

ament_export_include_directories(include)
ament_export_libraries(YoloV8_TRT tensorrt_cpp_api)

ament_package()
