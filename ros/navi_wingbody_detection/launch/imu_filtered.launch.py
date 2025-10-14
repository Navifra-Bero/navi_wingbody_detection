import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # 1. Madgwick 필터 노드 설정 (기존과 동일)
    # raw 데이터를 받아서 /imu/data_filtered 토픽으로 발행
    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'world_frame': 'enu',
            'publish_tf': False,
            'use_sim_time': True,
        }],
        remappings=[
            ('/imu/data_raw', '/camera/gyro_accel/sample'),
            ('/imu/data', '/imu/data_filtered') # 필터링된 결과 (고속)
        ]
    )

    # 2. Topic_tools throttle 노드 추가
    # /imu/data_filtered 토픽을 받아서 속도를 조절한 뒤
    # /imu/data_filtered_throttled 토픽으로 발행
    imu_throttle_node = Node(
        package='topic_tools',
        executable='throttle',
        name='imu_throttle',
        arguments=[
            'messages',                         # 모드: messages
            '/imu/data_filtered',            # 입력 토픽 (위 필터 노드의 출력)
            '30.0',                          # 원하는 출력 속도 (Hz)
            '/imu/data_filtered_throttled'   # 최종 출력 토픽 (저속)
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )


    # 3. 위에서 정의한 두 개의 노드를 함께 실행
    return LaunchDescription([
        imu_filter_node,
        imu_throttle_node
    ])