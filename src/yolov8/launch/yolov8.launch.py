"""Launch the yolov8 segmentation node with parameters read from yolov8.env."""
import os

from ament_index_python.packages import get_package_share_directory
from environs import Env
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share_dir = get_package_share_directory('yolov8')
    models_dir = os.path.join(package_share_dir, 'models')
    env_file_path = os.path.join(package_share_dir, 'yolov8.env')

    if not os.path.exists(env_file_path):
        raise FileNotFoundError(
            f'Expected yolov8.env at {env_file_path}. Copy example.env to yolov8.env at '
            'the workspace root and rebuild the yolov8 package so it is installed in the '
            "package's share directory."
        )

    env = Env()
    env.read_env(env_file_path)

    model_path = os.path.join(models_dir, env('ONNX_MODEL'))
    camera_topics = [t.strip() for t in env.str('CAMERA_TOPICS').split(',') if t.strip()]
    camera_topic_suffix = env('CAMERA_TOPIC_SUFFIX')
    camera_buffer_hz = env.float('CAMERA_BUFFER_HZ')
    visualize_masks = env.bool('VISUALIZE_MASKS')
    enable_one_channel_mask = env.bool('ENABLE_ONE_CHANNEL_MASK')
    visualize_one_channel_mask = env.bool('VISUALIZE_ONE_CHANNEL_MASK')
    nice_level = env.int('NICE_LEVEL')
    precision = env('PRECISION')
    calibration_data_directory = env('CALIBRATION_DATA_DIRECTORY')
    probability_threshold = env.float('PROBABILITY_THRESHOLD')
    nms_threshold = env.float('NMS_THRESHOLD')
    top_k = env.int('TOP_K')
    seg_channels = env.int('SEG_CHANNELS')
    seg_h = env.int('SEG_H')
    seg_w = env.int('SEG_W')
    segmentation_threshold = env.float('SEGMENTATION_THRESHOLD')

    cli_args = [
        '--model', model_path,
        '--precision', precision,
        '--prob-threshold', str(probability_threshold),
        '--nms-threshold', str(nms_threshold),
        '--top-k', str(top_k),
        '--seg-channels', str(seg_channels),
        '--seg-h', str(seg_h),
        '--seg-w', str(seg_w),
        '--seg-threshold', str(segmentation_threshold),
    ]
    if calibration_data_directory:
        cli_args.extend(['--calibration-data', calibration_data_directory])

    return LaunchDescription([
        Node(
            package='yolov8',
            executable='ros_segmentation',
            parameters=[{
                'camera_topics': camera_topics,
                'camera_topic_suffix': camera_topic_suffix,
                'camera_buffer_hz': camera_buffer_hz,
                'visualize_masks': visualize_masks,
                'enable_one_channel_mask': enable_one_channel_mask,
                'visualize_one_channel_mask': visualize_one_channel_mask,
            }],
            arguments=cli_args,
            prefix=[f'nice -n {nice_level}'],
        )
    ])
