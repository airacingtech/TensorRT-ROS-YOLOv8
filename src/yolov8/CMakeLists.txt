cmake_minimum_required(VERSION 3.8)
project(yolov8)

# Set compiler warnings if using GCC or Clang
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find CUDA
find_package(CUDA REQUIRED)
# # Set CUDA paths
# link_directories(${CUDA_LIBRARY_DIRS})
message(STATUS "Found CUDA: ${CUDA_VERSION} at ${CUDA_TOOLKIT_ROOT_DIR}")
# Add OpenCV to LD_LIBRARY_PATH
# SET(ENV{LD_LIBRARY_PATH} "/home/chris/testing/race_common/src/perception/TensorRT-ROS-YOLOv8/src/yolov8/scripts/opencv_build_files/opencv-4.8.0/build/lib:$ENV{LD_LIBRARY_PATH}")

# Find ROS2 dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(yolov8_interfaces REQUIRED)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

include_directories(
  ${CUDA_INCLUDE_DIRS}
)

# # Use ccache to speed up rebuilds
# include(cmake/ccache.cmake)

# Build the TensorRT inference engine library
add_subdirectory(libs/tensorrt-cpp-api)

# Build the YoloV8 library
add_library(YoloV8_TRT SHARED src/yolov8.cpp src/utils.cpp)
target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(YoloV8_TRT PUBLIC libs/tensorrt-cpp-api/src)

# Build the DPT Library for depth estimation
cuda_add_library(cuda_postprocessing SHARED src/cuda_kernels/dpt_postprocess.cu)
add_library(DPT_TRT SHARED src/dpt.cpp src/utils.cpp)
target_include_directories(DPT_TRT PUBLIC 
  ${OpenCV_INCLUDE_DIRS} 
  ${CUDA_INCLUDE_DIRS} 
  ${TensorRT_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_kernels
)
target_link_libraries(DPT_TRT PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} cuda_postprocessing)

# # Add utility library for command line arguments
# add_library(CmdLineUtil SHARED src/cmd_line_util.h)
# target_include_directories(CmdLineUtil PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build ROS2 node
add_executable(ros_segmentation src/ros_segmentation.cpp)
ament_target_dependencies(ros_segmentation rclcpp sensor_msgs cv_bridge yolov8_interfaces)
# target_link_libraries(ros_segmentation YoloV8_TRT)
target_link_libraries(ros_segmentation YoloV8_TRT DPT_TRT)

# Install targets
install(TARGETS
  ros_segmentation
  DESTINATION lib/${PROJECT_NAME}
)

# Moves YoloV8_TRT library to the install folder
install(TARGETS
  YoloV8_TRT
  tensorrt_cpp_api
  DPT_TRT
  cuda_postprocessing
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  DESTINATION lib/${PROJECT_NAME}
)

# Places the launch, config, and models folders in the install/yolov8/share/yolov8 directory so they can be found by the launch file
install(DIRECTORY
  launch
  config
  models
  DESTINATION share/${PROJECT_NAME}/
  PATTERN ".gitignore" EXCLUDE # Exclude .gitignore files from being installed
)

# Install environment file
install(FILES
  ../../yolov8.env
  DESTINATION share/${PROJECT_NAME}
)

# Install Python node
install(
  PROGRAMS
    src/ros_collect_with_radar.py
  DESTINATION lib/${PROJECT_NAME}
)

# Allows other packages to find and link to the YoloV8_TRT and tensorrt_cpp_api libraries
ament_export_include_directories(include)
ament_export_libraries(YOLOV8_TRT tensorrt_cpp_api yolov8_interfaces)

# Generate package information
ament_package()