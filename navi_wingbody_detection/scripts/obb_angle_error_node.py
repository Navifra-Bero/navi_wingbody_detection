#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseArray, PoseStamped
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # 헤드리스 저장용
import matplotlib.pyplot as plt
from collections import deque, defaultdict

def quat_to_yaw(q):
    # ZYX yaw (degrees)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    w, x, y, z = q.w, q.x, q.y, q.z
    s1 = 2.0*(w*z + x*y)
    s2 = 1.0 - 2.0*(y*y + z*z)
    yaw = math.degrees(math.atan2(s1, s2))
    # normalize to [-180, 180)
    while yaw >= 180.0: yaw -= 360.0
    while yaw <  -180.0: yaw += 360.0
    return yaw

def wrap_to_pm180(deg):
    while deg >= 180.0: deg -= 360.0
    while deg <  -180.0: deg += 360.0
    return deg

def wrap_err_to_pm40_by_90(err_deg):
    """
    예시 요구:
      60 -> -30 (90 빼기)
      -60 -> +30 (90 더하기)
      120 -> 30  (90 빼기)
      170 -> -10 (180 빼기)
    즉, 90° 단위로 이동해 |오차|를 40° 이하로 최소화.
    """
    candidates = [err_deg + k*90.0 for k in (-2, -1, 0, 1, 2)]
    candidates = [wrap_to_pm180(c) for c in candidates]
    # 40 안으로 들어오고 abs 최소인 것 우선, 없으면 abs 최소
    inside = [c for c in candidates if -40.0 <= c <= 40.0]
    if inside:
        return min(inside, key=lambda v: abs(v))
    return min(candidates, key=lambda v: abs(v))

class OBBIMUHistNode(Node):
    def __init__(self):
        super().__init__('obb_angle_error_node')

        # Topics
        self.topic_imu = '/imu_pose'
        self.topic_poses = [
            ('var1', '/obb_poses'),
            ('var2', '/obb_poses_2'),
            ('var3', '/obb_poses_3'),
            ('var4', '/obb_poses_4'),
        ]

        # QoS (퍼블리셔가 Reliable일 경우 맞춤)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # State
        self.imu_yaw = None
        # criterion별 오프셋, 샘플 리스트
        self.offset = {name: None for name, _ in self.topic_poses}
        self.errors = {name: [] for name, _ in self.topic_poses}

        # Subscribers
        self.sub_imu = self.create_subscription(
            PoseStamped, self.topic_imu, self.cb_imu, qos)

        self.subs_obb = []
        for name, topic in self.topic_poses:
            self.subs_obb.append(
                self.create_subscription(PoseArray, topic,
                                         lambda msg, nm=name: self.cb_obb(msg, nm),
                                         qos))

        self.get_logger().info('OBB-IMU histogram node started.')

    def cb_imu(self, msg: PoseStamped):
        self.imu_yaw = quat_to_yaw(msg.pose.orientation)

    def cb_obb(self, msg: PoseArray, name: str):
        # PoseArray가 비었으면 스킵
        if self.imu_yaw is None:  # IMU 먼저 필요
            return
        if not msg.poses:
            return

        p = msg.poses[0]
        obb_yaw = quat_to_yaw(p.orientation)

        raw_diff = wrap_to_pm180(obb_yaw - self.imu_yaw)

        if self.offset[name] is None:
            self.offset[name] = raw_diff
            return  # 다음 샘플부터 에러 수집

        err = wrap_to_pm180(raw_diff - self.offset[name])
        err = wrap_err_to_pm40_by_90(err)

        self.errors[name].append(err)

    def save_txts(self, base='obb_angle_errors'):
        for name, _ in self.topic_poses:
            arr = self.errors[name]
            fname = f'{base}_{name}.txt'
            with open(fname, 'w') as f:
                for v in arr:
                    f.write(f'{v:.6f}\n')
        self.get_logger().info('Saved angle error text files.')
        
    def save_histograms(self, out_png='obb_angle_error_hists.png'):
        # 4행 1열 subplot (세로로)
        fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
        bins = np.arange(-40, 40 + 5, 5)  # 5도 간격
        titles = {
            'var1': 'Area Criterion',
            'var2': 'Closness Criterion',
            'var3': 'Inlier Criterion',
            'var4': 'Variances Criterion',
        }

        any_data = False
        for ax, (name, _) in zip(axes, self.topic_poses):
            data = np.array(self.errors[name], dtype=float)
            if data.size == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', fontsize=10)
                ax.set_title(titles[name])
                ax.set_xlim([-40, 40])
                ax.set_ylim([0, 400])
                ax.set_xticks(np.arange(-40, 45, 5))
                ax.set_yticks(np.arange(0, 401, 100))
                continue

            any_data = True
            counts, _ = np.histogram(data, bins=bins)
            centers = (bins[:-1] + bins[1:]) / 2.0

            # 막대그래프 (테두리 추가, grid 없음)
            ax.bar(centers, counts, width=5.0, align='center', color='flag')

            ax.set_title(titles[name])
            ax.set_xlim([-40, 40])
            ax.set_ylim([0, 400])
            ax.set_xticks(np.arange(-40, 45, 5))   # x축 눈금 5도 단위
            ax.set_yticks(np.arange(0, 401, 100))  # y축 0~400, 100 간격
            ax.set_xlabel('Heading error (deg)')
            ax.set_ylabel('Count')

            # grid 제거
            ax.grid(False)

        fig.suptitle('Angle Error Histogram per OBB Criterion (5° bins, fixed 0–400 count)', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_png)
        plt.close(fig)

        if any_data:
            self.get_logger().info(f'Saved hist figure: {out_png}')
        else:
            self.get_logger().warn('No data received for any criterion; saved placeholders.')


def main():
    rclpy.init()
    node = OBBIMUHistNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C detected, finishing up...')
    finally:
        # 결과 저장
        node.save_txts()
        node.save_histograms()
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
