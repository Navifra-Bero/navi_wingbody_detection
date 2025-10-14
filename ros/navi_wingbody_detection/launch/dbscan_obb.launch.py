from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="navi_object_detection",
            executable="dbscan_obb.py",
            name="dbscan_obb_node",
            output="screen",
            parameters=[{
                # Input/Output topics
                "input_topic": "/patchworkpp/nonground",
                "marker_topic": "/dbscan/obb_markers",

                # DBSCAN parameters
                "eps": 0.5,
                "min_samples": 50,
                "max_cluster": 2000,

                # Downsampling
                "voxel": 0.0,

                # OBB robust 설정
                "percentile": True,
                "min_box_xy": 0.10,
                "min_points_after_voxel": 10,

                # Smoothing
                "smooth.alpha_pos": 0.3,
                "smooth.alpha_size": 0.4,
                "smooth.alpha_yaw": 0.3,
                "smooth.match_radius": 1.5,

                # Marker lifetime
                "marker_lifetime": 0.2,
            }]
        )
    ])
