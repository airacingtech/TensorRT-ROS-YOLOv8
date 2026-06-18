# Convenience targets for the TensorRT-ROS-YOLOv8 workspace.
# Usage: make <target>

.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  deps                   Import source-only deps (cv_bridge) into src/"
	@echo "  build                  Build with colcon (release)"
	@echo "  build-debug            Build with colcon (debug symbols)"
	@echo "  clean                  Remove build/, install/, and log/"
	@echo "  install-opencv-cuda    Build & install OpenCV with CUDA system-wide"
	@echo "                         (OPENCV_VERSION=<x.y.z> CUDA_BIN_ARCH=<x.y>)"
	@echo "  uninstall-opencv-cuda  Uninstall the OpenCV build produced above"
	@echo "                         (OPENCV_VERSION=<x.y.z>)"
	@echo "  copy-engine            Copy built engines out of install/ to ./"
	@echo "  return-engine          Restore an engine into install/ (ENGINE=<file>)"

.PHONY: clean
clean:
	rm -rf build/ install/ log/

# Import source-only dependencies (cv_bridge / vision_opencv) into src/.
# Needed on a ROS 2 source build (e.g. Jazzy on Ubuntu 22.04) where cv_bridge has
# no apt package. On a binary ROS install, `rosdep install` covers it instead.
.PHONY: deps
deps:
	mkdir -p src
	vcs import src < dependencies.repos
	# opencv_tests isn't needed to build/run yolov8 and pulls extra test deps.
	if [ -d src/vision_opencv/opencv_tests ]; then touch src/vision_opencv/opencv_tests/COLCON_IGNORE; fi
	@echo "Source deps imported into src/. Now run 'make build'."

.PHONY: build
build:
	colcon build

.PHONY: build-debug
build-debug:
	colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug

# Build and install OpenCV with CUDA support system-wide.
# Inputs: OPENCV_VERSION (e.g. 4.8.0), CUDA_BIN_ARCH (your GPU's compute capability, e.g. 8.9)
.PHONY: install-opencv-cuda
install-opencv-cuda:
	@if [ -z "$(OPENCV_VERSION)" ] || [ -z "$(CUDA_BIN_ARCH)" ]; then \
		echo "OPENCV_VERSION and CUDA_BIN_ARCH must be set."; exit 1; \
	fi
	source ./src/yolov8/scripts/install_opencv.sh $(OPENCV_VERSION) $(CUDA_BIN_ARCH)

.PHONY: uninstall-opencv-cuda
uninstall-opencv-cuda:
	@if [ -z "$(OPENCV_VERSION)" ]; then \
		echo "OPENCV_VERSION must be set (the version originally installed)."; exit 1; \
	fi
	source ./src/yolov8/scripts/uninstall_opencv.sh $(OPENCV_VERSION)

# Copy generated engine files out of install/ so they survive a `make clean`.
# Fails (intentionally, via cp's exit code) if there are no engines to copy.
.PHONY: copy-engine
copy-engine:
	cp install/yolov8/share/yolov8/models/engines/* .

# Restore a previously saved engine into install/. Inputs: ENGINE=<path>
.PHONY: return-engine
return-engine:
	@if [ -z "$(ENGINE)" ]; then echo "ENGINE parameter not set."; exit 1; fi
	mkdir -p install/yolov8/share/yolov8/models/engines/
	cp $(ENGINE) install/yolov8/share/yolov8/models/engines/
