from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='navi_object_detection',   # 패키지명에 맞게
            executable='obb_extractor_node',
            name='obb_extractor_node',
            output='screen',
            parameters=[{
                'input_topic':  '/patchworkpp/nonground',
                'marker_topic': '/obb_markers',
                'marker_topic_2': '/obb_markers_2',
                'cluster_topic': '/cluster/points',
                'cluster_tolerance': 0.7,
                'min_cluster_size': 40,
                'max_cluster_size': 20000,
                'marker_lifetime': 0.0,
                'min_box_xy': 0.10
            }]
        )
    ])
