from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='navi_wingbody_detection',
            executable='object_detector.py',   # 설치/심볼릭 설정에 맞추세요
            name='obb_detector_py',
            output='screen',
            parameters=[{
                'input_topic':  '/patchworkpp/nonground',
                'marker_topic':   '/obb/markers_area',
                'marker_topic_2': '/obb/markers_closs',
                'marker_topic_3': '/obb/markers_inlier',
                'marker_topic_4': '/obb/markers_var',
                'poses_topic':   '/obb_poses',
                'poses_topic_2': '/obb_poses_2',
                'poses_topic_3': '/obb_poses_3',
                'poses_topic_4': '/obb_poses_4',

                'cluster_tolerance': 0.1,
                'min_cluster_size': 200,
                'max_cluster_size': 20000,
                'marker_lifetime': 0.8,
                'min_box_xy': 0.3,
                'voxel_leaf': 0.1,

                'lshape.dtheta_deg': 1.0,
                'lshape.inlier_threshold': 0.15,
                'lshape.min_dist_nearest': 0.01,
            }]
        )
    ])
