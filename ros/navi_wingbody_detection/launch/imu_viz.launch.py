from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # imu_filter_node = Node(
    #     package='imu_filter_madgwick',
    #     executable='imu_filter_madgwick_node',
    #     name='imu_filter',
    #     output='screen',
    #     parameters=[{
    #         'use_mag': False,
    #         'world_frame': 'enu',
    #         'publish_tf': False,
    #         'use_sim_time': True,
    #     }],
    #     remappings=[
    #         ('/imu/data_raw', '/camera/gyro_accel/sample'),
    #         ('/imu/data', '/imu/data_filtered')
    #     ]
    # )

    imu_orient_node = Node(
        package='navi_wingbody_detection',
        executable='imu_viz_node.py',
        name='imu_viz_node',
        output='screen',
        parameters=[{
            'topic': '/imu/data',
            'fixed_frame': 'vanjee_lidar'
        }]
    )

    return LaunchDescription([
        # imu_filter_node,
        imu_orient_node
    ])
