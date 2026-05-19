# TensorRT YOLOv8 ROS Instance Segmentation

Real-time multi-camera instance segmentation for AI Racing Tech, implemented as a ROS 2 C++ node that runs a fine-tuned YOLOv8 segmentation model through TensorRT. Designed for ROS 2 Iron on Ubuntu 22.04 with NVIDIA GPUs.

The node consumes one image topic per camera, batches a fixed-size group of frames per inference tick (matching the model's batch size), and publishes detections, optional overlay images, and an optional one-channel mask image suitable for LiDAR projection.

This package assumes you have already fine-tuned a YOLOv8 segmentation model and exported it to ONNX. The companion training repos are `YOLOv8-Fine-Tune` and `SAM2_YOLOv8_autolabeler_finetune`.

## Published Topics

For each camera topic `/<camera>` in `CAMERA_TOPICS`, the node publishes:

### `/yolov8/<camera>/detections` — `yolov8_interfaces/Yolov8Detections`

Per-frame detection bundle. All arrays are parallel — the i-th entry of `labels`, `probabilities`, `class_names`, and `bounding_boxes` describe the same detection.

- `header` — copied from the source camera image (timestamp + frame_id).
- `indexes` — 1-based instance ids. 0 is reserved for background in `seg_mask_one_channel`.
- `labels` — model class ids.
- `probabilities` — confidence in `[0, 1]`.
- `class_names` — human-readable names, indexed by `CLASS_NAMES`.
- `seg_mask_one_channel` — `sensor_msgs/Image` (mono8). All instance masks combined into a single channel, with pixel value = instance index. Same `header`/`height`/`width`/`step` as the source camera image.
- `bounding_boxes` — `yolov8_interfaces/Yolov8BBox[]`, each containing top-left `Point2D`, `rect_width`, `rect_height`.

### `/yolov8/<camera>/image` — `sensor_msgs/Image` (bgr8)

Published when `visualize_masks=true`. The source camera image with bounding boxes, labels, and translucent segmentation masks rendered on top — useful for RViz2 / Foxglove.

### `/yolov8/<camera>/seg_mask_one_channel` — `sensor_msgs/Image` (rgb8)

Published when both `enable_one_channel_mask=true` and `visualize_one_channel_mask=true`. The same data as `detections.seg_mask_one_channel`, but normalized to `[0, 255]` and converted to RGB so it can be displayed. The normalization changes pixel colors when the number of detections changes — this image is only for visualization, never for downstream consumers.

## Installation

Tested on Ubuntu 22.04 with ROS 2 Iron. Other distributions are unsupported.

### 1. ROS 2 Iron

Install from the Debian packages: https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debians.html

Do not source a conda environment when building or running this package — ROS 2 does not officially support conda and it tends to break `rclpy`.

### 2. CUDA 11.8

Install via the runfile (not the deb): https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local

Confirm `nvidia-smi` runs cleanly afterwards. If it errors out, the deb installer likely replaced your low-level NVIDIA drivers — reinstall the appropriate proprietary driver from `Software & Updates → Additional Drivers` before rebooting.

### 3. cuDNN 8.2.4

Download from https://developer.nvidia.com/cudnn and install per https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html.

### 4. TensorRT

Install either the DEB package or the TAR archive — not both. By default this package looks for TensorRT at `/usr/src/tensorrt` (the DEB layout). To use a TAR install, set `TensorRT_DIR` or the `TENSORRT_DIR` environment variable to the TAR's root directory:

```bash
export TENSORRT_DIR="$HOME/libs/TensorRT-10.x.x.x/"
```

- **DEB (recommended)**: TensorRT 10 GA from https://developer.nvidia.com/tensorrt/download/10x (install full C++ and Python runtimes). Headers live in `/usr/src/tensorrt`, shared libs in `/usr/lib/x86_64-linux-gnu/`.
- **TAR**: TensorRT 8.6 or 10 GA. Unpack into `~/libs/` and follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar. Remember to add the TensorRT `lib/` to `LD_LIBRARY_PATH` in your shell rc.

### 5. OpenCV 4.8.0 with CUDA

We need OpenCV built with CUDA support. **This will overwrite any existing system OpenCV.** From the workspace root:

```bash
make install-opencv-cuda OPENCV_VERSION=4.8.0 CUDA_BIN_ARCH=<your-compute-capability>
```

Find your GPU's compute capability at https://developer.nvidia.com/cuda-gpus. The build takes a while.

To revert:

```bash
make uninstall-opencv-cuda
```

## Building

Create a ROS 2 workspace and copy (or symlink) the `yolov8` and `yolov8_interfaces` packages into its `src/` directory. Then from the workspace root:

```bash
cp example.env yolov8.env
# Edit yolov8.env for your model path, camera topics, etc.

make build
source install/setup.bash
```

`make build-debug` builds with debug symbols for use with GDB / the VS Code ROS extension.

## Running

```bash
ros2 launch yolov8 yolov8.launch.py
```

The launch file reads parameters from `install/yolov8/share/yolov8/yolov8.env`, which is installed from the workspace-root `yolov8.env`. **You must rebuild after editing `yolov8.env`** so the share copy is refreshed.

The first run on a given GPU will build the TensorRT engine and cache it in `install/yolov8/share/yolov8/models/engines/`. Subsequent runs load the cached engine and start in seconds. Use `make copy-engine` to preserve the engine across `make clean`.

You can also run the node directly without the launch file:

```bash
ros2 run yolov8 ros_segmentation --model <path/to/model.onnx>
```

## Tmuxp

Two tmuxp sessions are provided in `tmuxp_configs/`:

- `rviz_yolov8.yaml` — node + `rosbag play` + RViz2 + topic echo.
- `foxglove_yolov8.yaml` — node + `rosbag play` + Foxglove Bridge + topic echo.

Edit the `rosbag play` path before running.

Install tmuxp with `sudo apt install tmuxp`.

## Debugging

Use `debug_yolov8.launch.py` with the [ROS 2 VSCode extension](https://docs.ros.org/en/rolling/How-To-Guides/ROS-2-IDEs.html). It is identical to `yolov8.launch.py` except the `nice -n` prefix is omitted (GDB cannot attach to a `nice`'d process).

## Troubleshooting

Check the most recent ROS log first:

```bash
cat ~/.ros/log/latest.log
```

Common errors:

- **Exit code `-9` during engine build/load** — the process was OOM-killed. Close other apps; monitor with `htop` + `nvidia-smi`.
- **`CUDA initialization failure with error: 46`** — another process is holding the GPU. Sometimes a reboot is the only fix.
- **`No module named rclpy`** — you are in a conda environment. Deactivate it.
- **`libnvinfer.so.*: cannot open shared object file`** — TAR install of TensorRT not on `LD_LIBRARY_PATH`. Re-source after exporting it.
- **`cuda_runtime_api.h could not determine number of CUDA-capable devices`** — usually appears after long uptime; reboot.

## Sources

This project incorporates code from [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP) (MIT).
