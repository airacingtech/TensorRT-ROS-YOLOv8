# TensorRT YOLOv8 ROS Instance Segmentation

Real-time multi-camera instance segmentation for AI Racing Tech, implemented as a ROS 2 C++ node that runs a fine-tuned YOLOv8 segmentation model through TensorRT. Designed for ROS 2 Jazzy with NVIDIA GPUs.

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

Targets ROS 2 Jazzy. On Ubuntu 24.04 use the Jazzy debs; on Ubuntu 22.04 there are no Jazzy debs, so use a Jazzy source build (this is how the ART stack runs Jazzy on 22.04).

### 1. ROS 2 Jazzy

- Ubuntu 24.04 (debs): https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
- Ubuntu 22.04 (source build): https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Development-Setup.html

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

To revert (pass the version that was originally installed):

```bash
make uninstall-opencv-cuda OPENCV_VERSION=4.8.0
```

### 6. ROS dependencies (cv_bridge)

This package depends on `cv_bridge`. On a **binary** ROS 2 install,
`rosdep install --from-paths src --ignore-src -r -y` pulls it in. On a ROS 2
**source build** (e.g. Jazzy on Ubuntu 22.04) `cv_bridge` has no apt package, so
build it from source — `make deps` imports `vision_opencv` (pinned in
`dependencies.repos`) into `src/`, and the subsequent `make build` builds it
alongside `yolov8`. It picks up the same `/usr/local` CUDA OpenCV from step 5.

### 7. Model weights (ONNX)

The node loads a YOLOv8 **segmentation** model in ONNX form. Weights are **not**
committed to this repo (`models/` is git-ignored) — get the team's published
ONNX or export one from the training repo (`YOLOv8-Fine-Tune`), and place it in
`src/yolov8/models/`:

```bash
cp <your-model>.onnx src/yolov8/models/
```

**The batch size must match the camera count.** The ONNX batch dimension,
`BATCH_SIZE`, and the number of entries in `CAMERA_TOPICS` (all in `yolov8.env`)
must be **equal** — TensorRT specializes the engine to that batch at build time.
Typical setups run **4 or 6 cameras**, so use a **batch-4 or batch-6** ONNX
(export at the right batch with `export_onnx.py --batch <N>` from the training
repo). On first launch the ONNX is compiled into a cached TensorRT engine (see
[Running](#running)).

## Building

Build from this repository's root (it is the colcon workspace — `yolov8` and
`yolov8_interfaces` live in `src/`). Then:

```bash
source /opt/ros/jazzy/setup.bash    # or your Jazzy source build's install/setup.bash
cp example.env yolov8.env
# In yolov8.env set: ONNX_MODEL (file in models/), CAMERA_TOPICS, CAMERA_TOPIC_SUFFIX,
# and BATCH_SIZE == ONNX batch == number of CAMERA_TOPICS. See "Model weights" (step 7).

make deps      # first time only: import source-only deps (cv_bridge) — see step 6
make build     # also builds cv_bridge; installs yolov8.env + models/ into install/
source install/setup.bash
```

`make build-debug` builds with debug symbols for use with GDB / the VS Code ROS extension.

## Running

```bash
ros2 launch yolov8 yolov8.launch.py
```

The launch file reads parameters from `install/yolov8/share/yolov8/yolov8.env`, which is installed from the workspace-root `yolov8.env`. **You must rebuild after editing `yolov8.env`** so the share copy is refreshed.

The first run on a given GPU will build the TensorRT engine and cache it in `install/yolov8/share/yolov8/models/engines/`. Subsequent runs load the cached engine and start in seconds. Use `make copy-engine` to preserve the engine across `make clean`.

## Tmuxp

Two tmuxp sessions are provided in `tmuxp_configs/`:

- `rviz_yolov8.yaml` — node + `rosbag play` + RViz2 + topic echo.
- `foxglove_yolov8.yaml` — node + `rosbag play` + Foxglove Bridge + topic echo.

Edit the `rosbag play` path before running.

Install tmuxp with `sudo apt install tmuxp`.

## Debugging

Use `debug_yolov8.launch.py` with the [ROS 2 VSCode extension](https://docs.ros.org/en/rolling/How-To-Guides/ROS-2-IDEs.html). It is a thin wrapper around `yolov8.launch.py` that passes `debug:=true`, which drops the `nice -n` prefix (GDB cannot attach across `nice`). Equivalent to running `ros2 launch yolov8 yolov8.launch.py debug:=true` directly.

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

## Design notes

A few deliberate choices that may look odd at first glance:

- **`src/yolov8/libs/tensorrt-cpp-api/` is vendored MIT code** from [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP). It is kept close to upstream so future syncs are tractable; we only apply bug fixes and small cleanups, not structural refactors. It logs to `std::cout`/`std::cerr` because it is designed to be usable outside ROS.
- **`-Ofast` (not `-O3`)** in the CMakeLists — this is a real-time inference workload, the fast-math relaxations are acceptable, and the measured speedup over `-O3` is non-trivial. Debug builds drop the flag.
- **`BATCH_SIZE` is fixed at engine-build time** and must equal `len(CAMERA_TOPICS)`. The TensorRT engine is specialized to exactly that batch size for performance; varying it would require an engine rebuild.
- **Camera frame buffering**: the node accumulates the next frame from each camera and flushes either when all cameras have published (most of the time) or when `CAMERA_BUFFER_HZ` elapses (graceful degradation when a camera stalls). Cameras that did not publish in a given window contribute a zero placeholder image to the batch but do not produce detections.
- **No unit tests are shipped.** The hot path is GPU-coupled (TensorRT + CUDA + OpenCV-with-CUDA), and meaningful tests require the target hardware; we lean on integration testing via rosbags instead.

## Sources

This project incorporates code from [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP) (MIT).
