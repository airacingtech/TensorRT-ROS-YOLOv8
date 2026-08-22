cmake_minimum_required(VERSION 3.8)
project(yolov8)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# OpenCV-CUDA lives at a non-default prefix; prefer it when present so clean
# builds don't silently fall back to the CUDA-less system OpenCV.
if(NOT DEFINED OpenCV_DIR)
    foreach(_cv_prefix /opt/opencv-cuda $ENV{HOME}/ART/opencv-cuda)
        if(EXISTS "${_cv_prefix}/lib/cmake/opencv4/OpenCVConfig.cmake")
            set(OpenCV_DIR "${_cv_prefix}/lib/cmake/opencv4" CACHE PATH "OpenCV-CUDA build")
            break()
        endif()
    endforeach()
endif()

# CUDA toolkit selection. Pin with -DCUDA_VERSION=<x.y>; otherwise match the
# exact version the OpenCV-CUDA build was compiled with -- its config fatals on
# any other, and a machine can carry several toolkits plus a newer nvcc on PATH
# (e.g. 13.0 on the AV24 while OpenCV/TensorRT are the 12.6 family).
if(NOT DEFINED CUDA_VERSION AND DEFINED OpenCV_DIR AND EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    file(STRINGS "${OpenCV_DIR}/OpenCVConfig.cmake" _opencv_cuda_line
         REGEX "set\\(OpenCV_CUDA_VERSION \"[0-9.]+\"\\)")
    if(_opencv_cuda_line MATCHES "\"([0-9]+\\.[0-9]+)\"")
        set(CUDA_VERSION "${CMAKE_MATCH_1}")
        message(STATUS "CUDA toolkit pinned to ${CUDA_VERSION} by ${OpenCV_DIR}/OpenCVConfig.cmake")
    endif()
endif()
if(DEFINED CUDA_VERSION)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${CUDA_VERSION}")
else()
    # No OpenCV-CUDA build found: derive from nvcc on PATH, else the /usr/local/cuda symlink.
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

# OpenCV-CUDA lives at a non-standard prefix (see AV24_DEPENDENCIES.md), so record the
# resolved library directories in the RPATH rather than relying on LD_LIBRARY_PATH at runtime.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# ros_segmentation installs to lib/yolov8/ but its sibling libraries land in lib/.
set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(yolov8_interfaces REQUIRED)
find_package(rclcpp_components REQUIRED)

# NITROS ingest is optional: when Isaac ROS is in the workspace the node can take camera frames
# as GXF device buffers and skip cv_bridge and the host->device upload entirely. Absent it, the
# node builds exactly as before against sensor_msgs.
find_package(isaac_ros_managed_nitros QUIET)
find_package(isaac_ros_nitros_image_type QUIET)
if(isaac_ros_managed_nitros_FOUND AND isaac_ros_nitros_image_type_FOUND)
    set(YOLOV8_WITH_NITROS TRUE)
    message(STATUS "yolov8: NITROS found, building the device-buffer ingest path")
else()
    set(YOLOV8_WITH_NITROS FALSE)
    message(STATUS "yolov8: no NITROS, building the sensor_msgs-only ingest path")
endif()

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
ament_target_dependencies(ros_segmentation
    rclcpp rclcpp_components sensor_msgs cv_bridge yolov8_interfaces)
target_link_libraries(ros_segmentation YoloV8_TRT)

if(YOLOV8_WITH_NITROS)
    target_compile_definitions(ros_segmentation PRIVATE YOLOV8_WITH_NITROS)
    ament_target_dependencies(ros_segmentation isaac_ros_managed_nitros isaac_ros_nitros_image_type)
endif()

# Same node as a loadable component. Composing it into the camera driver's container is the
# only way a NITROS device buffer stays resolvable -- across processes NITROS falls back to
# sensor_msgs and the frame is copied off the GPU.
add_library(yolov8_component SHARED src/ros_segmentation.cpp)
ament_target_dependencies(yolov8_component
    rclcpp rclcpp_components sensor_msgs cv_bridge yolov8_interfaces)
target_link_libraries(yolov8_component YoloV8_TRT)
if(YOLOV8_WITH_NITROS)
    target_compile_definitions(yolov8_component PRIVATE YOLOV8_WITH_NITROS)
    ament_target_dependencies(yolov8_component isaac_ros_managed_nitros isaac_ros_nitros_image_type)
endif()
rclcpp_components_register_nodes(yolov8_component "YoloV8Node")

install(TARGETS yolov8_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

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
