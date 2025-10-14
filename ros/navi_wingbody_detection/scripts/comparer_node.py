#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

def to_sec(t: TimeMsg) -> float:
    return float(t.sec) + float(t.nanosec) * 1e-9

class OrientationComparer(Node):
    def __init__(self):
        super().__init__('orientation_comparer')

        # topics
        imu_topic   = '/imu/data_filtered'
        obb_topic_1 = '/obb_markers'
        obb_topic_2 = '/obb_markers_2'

        # QoS: RViz/마커 퍼블리셔가 Reliable이라면 여기서도 Reliable로 맞춤
        qos_viz = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # IMU는 센서 QoS 유사: 최신만 받게 하려면 depth=10 정도면 충분
        qos_imu = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50
        )

        # 개별 구독(수동 동기화용)
        self.sub_imu  = self.create_subscription(Imu,          imu_topic,   self.on_imu,  qos_imu)
        self.sub_ob1  = self.create_subscription(MarkerArray,  obb_topic_1, self.on_obb1, qos_viz)
        self.sub_ob2  = self.create_subscription(MarkerArray,  obb_topic_2, self.on_obb2, qos_viz)

        # 버퍼(최신 1개만 유지)
        self.last_imu  = None          # (stamp_sec, imu_msg)
        self.last_ob1  = None          # (stamp_sec, MarkerArray)
        self.last_ob2  = None          # (stamp_sec, MarkerArray)

        # 파라미터: 동기화 허용 오차(초)
        self.slop = self.declare_parameter('sync_slop', 0.5).get_parameter_value().double_value

        # 수집/결과
        self.timestamps = []
        self.start_time = None
        self.yaw_diffs_1 = []
        self.yaw_diffs_2 = []

        self.get_logger().info('Orientation Comparer (manual sync) started.')

    # -------- subscribers --------
    def on_imu(self, msg: Imu):
        t = to_sec(msg.header.stamp)
        self.last_imu = (t, msg)
        # self.get_logger().info(f'[DEBUG] IMU {t:.4f}')
        self.try_sync()

    def on_obb1(self, arr: MarkerArray):
        if not arr.markers:
            return
        t = to_sec(arr.markers[0].header.stamp)  # 첫 마커의 stamp 사용
        self.last_ob1 = (t, arr)
        # self.get_logger().info(f'[DEBUG] OBB1 {t:.4f}')
        self.try_sync()

    def on_obb2(self, arr: MarkerArray):
        if not arr.markers:
            return
        t = to_sec(arr.markers[0].header.stamp)
        self.last_ob2 = (t, arr)
        # self.get_logger().info(f'[DEBUG] OBB2 {t:.4f}')
        self.try_sync()

    # -------- manual sync --------
    def try_sync(self):
        if self.last_imu is None or self.last_ob1 is None or self.last_ob2 is None:
            return
        t_imu, imu_msg = self.last_imu
        t_ob1, obb1 = self.last_ob1
        t_ob2, obb2 = self.last_ob2

        # 세 타임스탬프가 모두 slop 이내인지 확인
        if abs(t_imu - t_ob1) <= self.slop and abs(t_imu - t_ob2) <= self.slop:
            # 동기화 성공
            self.process(imu_msg, obb1, obb2)

            # 재중복 호출 방지용: 사용한 obb들을 소거(혹은 최신만 유지)
            # IMU는 고주파수라 남겨두고, OBB는 소비 처리
            self.last_ob1 = None
            self.last_ob2 = None

    # -------- processing --------
    def process(self, imu_msg: Imu, obb1_arr: MarkerArray, obb2_arr: MarkerArray):
        # 비어있지 않음을 재확인
        if not obb1_arr.markers or not obb2_arr.markers:
            return
        obb1 = obb1_arr.markers[0]
        obb2 = obb2_arr.markers[0]

        current_time = to_sec(imu_msg.header.stamp)
        if self.start_time is None:
            self.start_time = current_time
        relative_time = current_time - self.start_time
        self.timestamps.append(relative_time)

        yaw_diff_1 = self.yaw_difference(imu_msg.orientation, obb1.pose.orientation)
        yaw_diff_2 = self.yaw_difference(imu_msg.orientation, obb2.pose.orientation)

        self.yaw_diffs_1.append(yaw_diff_1)
        self.yaw_diffs_2.append(yaw_diff_2)

        if len(self.timestamps) % 2 == 0:
            self.get_logger().info(
                f'✅ [{relative_time:.3f}s] OBB1 Gap: {yaw_diff_1:.2f}°, OBB2 Gap: {yaw_diff_2:.2f}°'
            )

    @staticmethod
    def yaw_difference(q_ref_msg: Quaternion, q_target_msg: Quaternion) -> float:
        q_ref = R.from_quat([q_ref_msg.x, q_ref_msg.y, q_ref_msg.z, q_ref_msg.w])
        q_tgt = R.from_quat([q_target_msg.x, q_target_msg.y, q_target_msg.z, q_target_msg.w])
        q_diff = q_ref.inv() * q_tgt
        return float(q_diff.as_euler('zyx', degrees=True)[0])

    def generate_plot(self):
        if not self.timestamps:
            self.get_logger().warn('No data to plot.')
            return
        avg_1, std_1 = (np.mean(self.yaw_diffs_1), np.std(self.yaw_diffs_1)) if self.yaw_diffs_1 else (0,0)
        avg_2, std_2 = (np.mean(self.yaw_diffs_2), np.std(self.yaw_diffs_2)) if self.yaw_diffs_2 else (0,0)

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 8))
        plt.plot(self.timestamps, self.yaw_diffs_1, label=f'OBB 1 vs GT (/obb_markers)\nAvg: {avg_1:.2f}°, Std: {std_1:.2f}°', alpha=0.85)
        plt.plot(self.timestamps, self.yaw_diffs_2, label=f'OBB 2 vs GT (/obb_markers_2)\nAvg: {avg_2:.2f}°, Std: {std_2:.2f}°', alpha=0.85)
        plt.axhline(0, color='r', linestyle='--', linewidth=1, label='Zero Error')
        plt.title('Orientation Yaw Difference Comparison (vs IMU GT)', fontsize=16)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Yaw Difference (degrees)', fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        out = 'orientation_comparison.png'
        plt.savefig(out)
        self.get_logger().info(f'Plot saved to {out}')

def main(args=None):
    rclpy.init(args=args)
    node = OrientationComparer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C: shutting down.')
    finally:
        node.generate_plot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
