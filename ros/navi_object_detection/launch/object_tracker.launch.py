from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='patchworkpp',
            executable='obb_sort_tracker_node',
            name='obb_sort_tracker_node',
            output='screen',
            parameters=[{
                'input_markers_topic': '/obb_markers',
                'output_tracked_topic': '/tracked_obb',
                'iou_threshold': 0.25,
                'max_age': 3,
                'min_hits': 2,
                'marker_lifetime': 0.2,
            }]
        )
    ])
