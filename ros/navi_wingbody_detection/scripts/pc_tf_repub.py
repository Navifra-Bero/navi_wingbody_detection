#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

class PcTfRepub(Node):
    def __init__(self):
        super().__init__('pc_tf_repub')

        # Params
        self.declare_parameter('input_topic', '/camera/depth/points')
        self.declare_parameter('output_topic', '/camera/depth/tf_ch_points')
        self.declare_parameter('output_frame', 'vanjee_lidar')

        def row_param(name, default):
            return np.array(
                self.get_parameter(name).get_parameter_value().double_array_value
                if self.has_parameter(name) else default, dtype=np.float64
            )

        r0 = row_param('camera_to_lidar_row0',
                       np.array([0.03028327, -0.99951961, -0.00659278, -0.03088140]))
        r1 = row_param('camera_to_lidar_row1',
                       np.array([0.00134207,  0.00663646, -0.99997708,  0.11743733]))
        r2 = row_param('camera_to_lidar_row2',
                       np.array([0.99954046,  0.03027373,  0.00154240, -0.08846909]))
        r3 = row_param('camera_to_lidar_row3',
                       np.array([0.0,         0.0,         0.0,         1.0]))
        self.T = np.vstack([r0, r1, r2, r3])

        self.input_topic  = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.output_frame = self.get_parameter('output_frame').get_parameter_value().string_value

        sub_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.cb, sub_qos)
        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)

        self.get_logger().info(
            f"[pc_tf_repub] in='{self.input_topic}' → out='{self.output_topic}' "
            f"| out_frame='{self.output_frame}'"
        )
        self.get_logger().info(f"T_cam_to_lidar:\n{self.T}")

    def cb(self, msg: PointCloud2):
        fields = msg.fields
        field_names = [f.name for f in fields]

        if not {'x', 'y', 'z'}.issubset(field_names):
            self.get_logger().warn("Incoming PointCloud2 has no x/y/z fields; skipping.")
            return

        # 모든 포인트를 리스트로 가져오기 (NaN 유지)
        pts_iter = pc2.read_points(msg, field_names=field_names, skip_nans=False)
        pts = list(pts_iter)
        n = len(pts)

        if n == 0:
            out = PointCloud2()
            out.header = msg.header
            out.header.frame_id = self.output_frame
            out.fields = fields
            out.height = msg.height
            out.width = msg.width
            out.is_bigendian = msg.is_bigendian
            out.point_step = msg.point_step
            out.row_step = msg.row_step
            out.is_dense = msg.is_dense
            self.pub.publish(out)
            return

        ix = field_names.index('x')
        iy = field_names.index('y')
        iz = field_names.index('z')

        # ✅ 리스트 컴프리헨션으로 안전하게 (N,3) 배열 생성
        xyz = np.array([[p[ix], p[iy], p[iz]] for p in pts], dtype=np.float64)

        finite_mask = np.isfinite(xyz).all(axis=1)
        xyz_h = np.ones((n, 4), dtype=np.float64)
        xyz_h[:, :3] = xyz

        xyz_lidar = xyz.copy()
        if np.any(finite_mask):
            transformed = (self.T @ xyz_h[finite_mask].T).T
            xyz_lidar[finite_mask] = transformed[:, :3]

        # 원래 튜플의 다른 필드 유지하며 xyz만 교체
        out_points = []
        for i, p in enumerate(pts):
            t = list(p)
            if finite_mask[i]:
                t[ix], t[iy], t[iz] = xyz_lidar[i, 0], xyz_lidar[i, 1], xyz_lidar[i, 2]
            out_points.append(tuple(t))

        header = msg.header
        header.frame_id = self.output_frame

        out_msg = pc2.create_cloud(header, fields, out_points)
        # organized 유지
        if msg.height * msg.width == len(out_points):
            out_msg.height = msg.height
            out_msg.width  = msg.width
            out_msg.is_dense = msg.is_dense
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PcTfRepub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
