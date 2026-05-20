"""Launch the yolov8 segmentation node with parameters read from yolov8.env.

Pass `debug:=true` to skip the `nice -n` prefix so a debugger can attach;
that is what debug_yolov8.launch.py does.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_env_file(path):
    """Parse a simple KEY=value env file. Strips matching surrounding quotes and ignores comments."""
    values = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            values[key.strip()] = value
    return values


def _as_bool(value):
    if value.lower() not in ('true', 'false'):
        raise ValueError(f'Expected true/false, got {value!r}')
    return value.lower() == 'true'


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

    env = _load_env_file(env_file_path)

    model_path = os.path.join(models_dir, env['ONNX_MODEL'])
    camera_topics = [t.strip() for t in env['CAMERA_TOPICS'].split(',') if t.strip()]
    camera_topic_suffix = env['CAMERA_TOPIC_SUFFIX']
    camera_buffer_hz = float(env['CAMERA_BUFFER_HZ'])
    visualize_masks = _as_bool(env['VISUALIZE_MASKS'])
    enable_one_channel_mask = _as_bool(env['ENABLE_ONE_CHANNEL_MASK'])
    visualize_one_channel_mask = _as_bool(env['VISUALIZE_ONE_CHANNEL_MASK'])
    nice_level = int(env['NICE_LEVEL'])
    precision = env['PRECISION']
    calibration_data_directory = env.get('CALIBRATION_DATA_DIRECTORY', '')
    probability_threshold = float(env['PROBABILITY_THRESHOLD'])
    nms_threshold = float(env['NMS_THRESHOLD'])
    top_k = int(env['TOP_K'])
    seg_channels = int(env['SEG_CHANNELS'])
    seg_h = int(env['SEG_H'])
    seg_w = int(env['SEG_W'])
    segmentation_threshold = float(env['SEGMENTATION_THRESHOLD'])
    batch_size = int(env['BATCH_SIZE'])
    class_names = [c.strip() for c in env['CLASS_NAMES'].split(',') if c.strip()]

    if len(camera_topics) != batch_size:
        raise ValueError(
            f'BATCH_SIZE ({batch_size}) must equal the number of CAMERA_TOPICS '
            f'({len(camera_topics)}). Update yolov8.env so they match.'
        )

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
        '--batch-size', str(batch_size),
    ]
    if calibration_data_directory:
        cli_args.extend(['--calibration-data', calibration_data_directory])
    # Keep --class-names last so its variadic argument list isn't split by another flag.
    cli_args.extend(['--class-names', *class_names])

    node_params = [{
        'camera_topics': camera_topics,
        'camera_topic_suffix': camera_topic_suffix,
        'camera_buffer_hz': camera_buffer_hz,
        'visualize_masks': visualize_masks,
        'enable_one_channel_mask': enable_one_channel_mask,
        'visualize_one_channel_mask': visualize_one_channel_mask,
    }]

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='If true, omit the `nice -n` prefix so GDB / the ROS 2 VS Code extension can attach.',
    )
    debug = LaunchConfiguration('debug')

    # Two near-identical Nodes selected by `debug`: GDB cannot attach across a `nice` exec, so the
    # debug path must drop the prefix. Using IfCondition/UnlessCondition keeps the rest of the node
    # configuration in a single place.
    return LaunchDescription([
        debug_arg,
        Node(
            package='yolov8',
            executable='ros_segmentation',
            parameters=node_params,
            arguments=cli_args,
            prefix=[f'nice -n {nice_level}'],
            condition=UnlessCondition(debug),
        ),
        Node(
            package='yolov8',
            executable='ros_segmentation',
            parameters=node_params,
            arguments=cli_args,
            condition=IfCondition(debug),
        ),
    ])
