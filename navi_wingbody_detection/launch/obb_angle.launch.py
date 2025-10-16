from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="navi_wingbody_detection",
            executable="obb_angle_error_node.py",
            name="obb_angle_error_node",
            output="screen",
            parameters=[{'use_sim_time': True}]
        )
    ])
