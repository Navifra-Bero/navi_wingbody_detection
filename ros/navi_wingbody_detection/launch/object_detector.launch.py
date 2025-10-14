from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='navi_wingbody_detection',
            executable='obb_extractor_node',
            name='obb_extractor_node',
            output='screen',
            # 필요하면 로깅 레벨 조정:
            # arguments=['--ros-args', '--log-level', 'obb_extractor_node:=info'],
            parameters=[{
                # 입력/출력 토픽
                'input_topic':        '/patchworkpp/nonground',
                'marker_topic':       '/obb_markers',
                'marker_topic_2':     '/obb_markers_2',
                'cluster_topic':      '/cluster/points',

                # 클러스터링/피팅 관련
                'cluster_tolerance':  0.7,
                'min_cluster_size':   500,
                'max_cluster_size':   20000,
                'min_box_xy':         0.3,
                'marker_lifetime':    0.8,

                # 다운샘플(프레임 변동성 안정화)
                'voxel_leaf':         0.07,   # 0.07~0.10 권장 (0.0이면 비활성)

                # 시각화 퍼블리셔 QoS (깊이 작게, 신뢰성 높게)
                'viz_reliable':       True,   # RViz 깜빡임 완화
                'viz_depth':          2,      # KeepLast(2)

                # 클러스터 포인트클라우드 퍼블리셔 QoS
                'cluster_reliable':   False,  # 디버그용이면 BestEffort 권장
                'cluster_depth':      1
            }]
        )
    ])
