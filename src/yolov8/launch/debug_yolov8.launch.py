"""Debug variant: includes yolov8.launch.py with `debug:=true` (drops the `nice -n` prefix
so GDB / the ROS 2 VS Code extension can attach to the node).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    yolov8_launch = os.path.join(
        get_package_share_directory('yolov8'), 'launch', 'yolov8.launch.py')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(yolov8_launch),
            launch_arguments={'debug': 'true'}.items(),
        ),
    ])
