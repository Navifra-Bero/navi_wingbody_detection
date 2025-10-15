from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='navi_wingbody_detection',
            executable='obb_extractor_node',
            name='obb_extractor_node',
            output='screen',

            parameters=[{
                'input_topic':        '/patchworkpp/nonground',
                'marker_topic':       '/obb_markers_area',
                'marker_topic_2':     '/obb_markers_near',
                'marker_topic_3':     '/obb_markers_inlier',
                'marker_topic_4':     '/obb_markers_var',
                'cluster_topic':      '/cluster/points',

                'cluster_tolerance':  0.7,
                'min_cluster_size':   300,
                'max_cluster_size':   20000,
                'min_box_xy':         0.3,
                'marker_lifetime':    0.8,

                'voxel_leaf':         0.07,

                'viz_reliable':       True, 
                'viz_depth':          2,  

                'cluster_reliable':   False, 
                'cluster_depth':      1,
                'lshape.dtheta_deg': 1.0,           # 각도 탐색 간격 (도)
                'lshape.inlier_threshold': 0.15,    # Inlier 기준 임계값 (미터)
                'lshape.min_dist_nearest': 0.01,
                'long_axis_hys': 0.05
            }]
        )
    ])
