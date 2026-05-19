#!/usr/bin/env bash
# Downloads, builds, and installs OpenCV (with the contrib modules) with CUDA support, system-wide.
# Refreshes the ldconfig cache afterwards.
# Usage: ./install_opencv.sh <OPENCV_VERSION> <CUDA_ARCH_BIN>
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <OPENCV_VERSION> <CUDA_ARCH_BIN>"
    echo "  OPENCV_VERSION  e.g. 4.8.0"
    echo "  CUDA_ARCH_BIN   Compute capability for your GPU (https://developer.nvidia.com/cuda-gpus), e.g. 8.9"
    exit 1
fi

OPENCV_VERSION="$1"
CUDA_ARCH_BIN="$2"
INSTALL_LOCATION="/usr/local"

CURR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${CURR_SCRIPT_DIR}"

mkdir -p opencv_build_files
cd opencv_build_files

echo "================================================================================"
echo "WARNING: Building OpenCV ${OPENCV_VERSION} with CUDA will take a long time and"
echo "         install OpenCV system-wide. It may conflict with existing OpenCV installs."
echo "================================================================================"

# Fetch the main OpenCV repo
if [ ! -e "${OPENCV_VERSION}.zip" ]; then
    wget "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip"
fi
if [ ! -d "opencv-${OPENCV_VERSION}" ]; then
    unzip "${OPENCV_VERSION}.zip"
fi

# Fetch the contrib repo
if [ ! -e "opencv_extra_${OPENCV_VERSION}.zip" ]; then
    wget -O "opencv_extra_${OPENCV_VERSION}.zip" "https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.zip"
fi
if [ ! -d "opencv_contrib-${OPENCV_VERSION}" ]; then
    unzip "opencv_extra_${OPENCV_VERSION}.zip"
fi

cd "opencv-${OPENCV_VERSION}"
if [ -d "build" ]; then
    echo "Removing previous build directory..."
    sudo rm -rf build
fi
mkdir build
cd build

echo "Configuring OpenCV ${OPENCV_VERSION} (CUDA arch ${CUDA_ARCH_BIN})..."
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX="${INSTALL_LOCATION}" \
      -D WITH_TBB=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_CUDA=ON \
      -D BUILD_opencv_cudacodec=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D BUILD_opencv_apps=OFF \
      -D BUILD_opencv_python2=OFF \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_PC_FILE_NAME=opencv.pc \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-${OPENCV_VERSION}/modules" \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
      -D WITH_FFMPEG=ON \
      -D CUDNN_INCLUDE_DIR=/usr/include/ \
      -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so \
      -D WITH_GTK=ON \
      ..

num_cores=$(( $(nproc) / 2 ))
if [ "${num_cores}" -lt 1 ]; then
    num_cores=1
fi
echo "Compiling with ${num_cores} of $(nproc) cores..."
make -j "${num_cores}"

echo "Installing OpenCV ${OPENCV_VERSION} to ${INSTALL_LOCATION}..."
sudo make -j "${num_cores}" install

echo "Refreshing ldconfig cache..."
sudo ldconfig -v

echo "Done. To undo, run uninstall_opencv.sh ${OPENCV_VERSION}."
