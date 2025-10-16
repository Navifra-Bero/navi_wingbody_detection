from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='navi_wingbody_detection',
            executable='object_tracker_node',
            name='object_tracker',
            output='screen',
            parameters=[{
                'input_topic': '/obb_markers',
                'output_topic': '/tracked_objects',
                'max_age': 5,        # 5 프레임 동안 보이지 않으면 추적 중단
                'min_hits': 2,       # 2 프레임 이상 연속으로 보여야 추적 시작
                'association_threshold': 2.5 # 2.5m 이내에 있어야 같은 객체로 판단
            }]
        )
    ])