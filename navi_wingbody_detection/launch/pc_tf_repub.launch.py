from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    params = {
        "input_topic":  "/camera/depth/points",
        "output_topic": "/camera/depth/tf_ch_points",
        "output_frame": "vanjee_lidar",

        "depth_to_color.rotation": [
            0.9944840669631958,  0.00024602707708254457, -0.002201989758759737,
           -0.000013758952263742685, 0.9944864511489868,  0.10486499965190887,
            0.0022156487684696913,  -0.10486471652984619, 0.994484007358551
        ],
        "depth_to_color.translation": [
            -0.03218218994140625, -0.0005416243076324463, 0.0025637595653533935
        ],

        "lidar_to_camera.row_major": [
            0.03028327, -0.99951961, -0.00659278, -0.03088140,
            0.00134207,  0.00663646, -0.99997708,  0.11743733,
            0.99954046,  0.03027373,  0.00154240, -0.08846909,
            0.0,         0.0,         0.0,         1.0
        ],
    }

    return LaunchDescription([
        Node(
            package='navi_wingbody_detection',
            executable='pc_tf_repub_node',
            name='pc_tf_repub',
            output='screen',
            parameters=[params],
        ),
    ])
