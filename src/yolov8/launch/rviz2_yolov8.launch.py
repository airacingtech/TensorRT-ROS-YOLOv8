"""Launch RViz2 preloaded with the yolov8 configuration."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rviz2_config_path = os.path.join(
        get_package_share_directory('yolov8'),
        'config',
        'yolov8.rviz',
    )

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz2_config_path],
            output='screen',
        )
    ])
