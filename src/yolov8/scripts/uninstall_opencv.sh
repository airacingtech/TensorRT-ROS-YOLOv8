#!/usr/bin/env bash
# Uninstalls the OpenCV build produced by install_opencv.sh.
# Usage: ./uninstall_opencv.sh <OPENCV_VERSION>
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <OPENCV_VERSION>"
    echo "Example: $0 4.8.0"
    exit 1
fi

OPENCV_VERSION="$1"

CURR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${CURR_SCRIPT_DIR}"

OPENCV_BUILD_DIR="opencv_build_files/opencv-${OPENCV_VERSION}/build"
if [ ! -d "${OPENCV_BUILD_DIR}" ]; then
    echo "Build directory not found: ${OPENCV_BUILD_DIR}"
    echo "Cannot uninstall without the original build tree."
    exit 1
fi

cd "${OPENCV_BUILD_DIR}"
echo "Uninstalling OpenCV ${OPENCV_VERSION}..."
sudo make uninstall

echo "Refreshing ldconfig cache..."
sudo ldconfig -v
