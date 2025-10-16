#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped

class IMUOrientationPub(Node):
    def __init__(self):
        super().__init__('imu_orientation_pub')
        self.declare_parameter('topic', '/imu/data_filtered')
        self.declare_parameter('fixed_frame', 'vanjee_lidar')  # RViz Fixed Frame

        self.topic = self.get_parameter('topic').get_parameter_value().string_value
        self.fixed_frame = self.get_parameter('fixed_frame').get_parameter_value().string_value

        # 구독(IMU) / 발행(Pose)
        self.sub = self.create_subscription(Imu, self.topic, self.cb, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/imu_pose', 10)

        self.get_logger().info(f"[IMU-ORIENT] Sub: {self.topic}  -> Pub: /imu_pose  (frame={self.fixed_frame})")

    def cb(self, msg: Imu):
        pose = PoseStamped()
        # stamp는 IMU와 동일, frame은 고정 프레임으로 강제
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.fixed_frame

        # 위치는 (0,0,0) 고정
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0

        # Orientation만 IMU에서 사용
        pose.pose.orientation = msg.orientation

        self.pose_pub.publish(pose)

def main(args=None):
    rclpy.init(args=args)
    node = IMUOrientationPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down imu_orientation_pub...")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
