#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header

from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
import math
from collections import defaultdict, deque

import sys, os
from pathlib import Path

_here = Path(__file__).resolve().parent
_build = Path.home() / 'ros2_ws' / 'build' / 'navi_wingbody_detection'

for p in (_here, _build):
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))

import lshape_bindings as lb

def rot_to_quat_z(yaw_deg: float):
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    return (0.0, 0.0, sy, cy)

class OBBDetectorPy(Node):
    def __init__(self):
        super().__init__('obb_detector_py')

        # Params
        self.input_topic = self.declare_parameter('input_topic', '/patchworkpp/nonground').get_parameter_value().string_value
        self.cluster_tolerance = float(self.declare_parameter('cluster_tolerance', 0.2).value)
        self.min_cluster_size  = int(self.declare_parameter('min_cluster_size', 40).value)
        self.max_cluster_size  = int(self.declare_parameter('max_cluster_size', 20000).value)
        self.marker_lifetime   = float(self.declare_parameter('marker_lifetime', 0.8).value)
        self.min_box_xy        = float(self.declare_parameter('min_box_xy', 0.4).value)
        self.voxel_leaf        = float(self.declare_parameter('voxel_leaf', 0.1).value)
        

        # 토픽 이름
        self.marker_topic_1 = self.declare_parameter('marker_topic',   '/obb/markers_area').get_parameter_value().string_value
        self.marker_topic_2 = self.declare_parameter('marker_topic_2', '/obb/markers_closs').get_parameter_value().string_value
        self.marker_topic_3 = self.declare_parameter('marker_topic_3', '/obb/markers_inlier').get_parameter_value().string_value
        self.marker_topic_4 = self.declare_parameter('marker_topic_4', '/obb/markers_var').get_parameter_value().string_value

        self.poses_topic_1  = self.declare_parameter('poses_topic',   '/obb_poses').get_parameter_value().string_value
        self.poses_topic_2  = self.declare_parameter('poses_topic_2', '/obb_poses_2').get_parameter_value().string_value
        self.poses_topic_3  = self.declare_parameter('poses_topic_3', '/obb_poses_3').get_parameter_value().string_value
        self.poses_topic_4  = self.declare_parameter('poses_topic_4', '/obb_poses_4').get_parameter_value().string_value

        # L-shape 파라미터
        self.dtheta_deg       = float(self.declare_parameter('lshape.dtheta_deg', 1.0).value)
        self.inlier_threshold = float(self.declare_parameter('lshape.inlier_threshold', 0.1).value)
        self.min_dist_nearest = float(self.declare_parameter('lshape.min_dist_nearest', 0.01).value)

        # Pub/Sub
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.cb_cloud,  rclpy.qos.qos_profile_sensor_data)
        self.pub_m1 = self.create_publisher(MarkerArray, self.marker_topic_1, 10)
        self.pub_m2 = self.create_publisher(MarkerArray, self.marker_topic_2, 10)
        self.pub_m3 = self.create_publisher(MarkerArray, self.marker_topic_3, 10)
        self.pub_m4 = self.create_publisher(MarkerArray, self.marker_topic_4, 10)
        self.pub_p1 = self.create_publisher(PoseArray,   self.poses_topic_1,  10)
        self.pub_p2 = self.create_publisher(PoseArray,   self.poses_topic_2,  10)
        self.pub_p3 = self.create_publisher(PoseArray,   self.poses_topic_3,  10)
        self.pub_p4 = self.create_publisher(PoseArray,   self.poses_topic_4,  10)

        # pybind 객체
        self.lshape = lb.LShaped(self.dtheta_deg, self.inlier_threshold, self.min_dist_nearest)

        self.get_logger().info(f"[PY OBB] sub={self.input_topic}  leaf={self.voxel_leaf} tol={self.cluster_tolerance} min={self.min_cluster_size} max={self.max_cluster_size}")

    # -------- PointCloud2 → numpy (x,y,z) --------
    def cloud_to_xyz(self, msg: PointCloud2):
        xyz = np.asarray([(p[0], p[1], p[2]) for p in pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True)], dtype=np.float32)
        if xyz.size == 0:
            return xyz
        if self.voxel_leaf > 1e-6:
            leaf = self.voxel_leaf
            q = np.floor(xyz / leaf) * leaf
            _, idx = np.unique(q.view([('', q.dtype)]*3), return_index=True)
            xyz = xyz[idx]
        return xyz

    # -------- Euclidean clustering (XY) --------
    def cluster_xy(self, xy: np.ndarray):
        tol = self.cluster_tolerance
        if xy.shape[0] == 0:
            return []

        # 그리드 셀
        keys = np.floor(xy / tol).astype(np.int32)
        cell2idx = defaultdict(list)
        for i, k in enumerate(map(tuple, keys)):
            cell2idx[k].append(i)

        # 인접 8방향
        nbrs = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1) if not (dx==0 and dy==0)]

        visited_cells = set()
        clusters = []

        for cell in cell2idx.keys():
            if cell in visited_cells: continue
            q = deque([cell])
            visited_cells.add(cell)
            point_ids = []

            while q:
                c = q.popleft()
                point_ids.extend(cell2idx[c])
                cx, cy = c
                for dx,dy in nbrs:
                    nc = (cx+dx, cy+dy)
                    if nc in cell2idx and nc not in visited_cells:
                        visited_cells.add(nc)
                        q.append(nc)

            if self.min_cluster_size <= len(point_ids) <= self.max_cluster_size:
                clusters.append(np.asarray(point_ids, dtype=np.int32))

        return clusters

    def make_cube(self, hdr: Header, id_: int, ns: str, cx, cy, w, h, yaw_deg, z_center, size_z, rgba):
        mk = Marker()
        mk.header = hdr
        mk.ns = ns
        mk.id = id_
        mk.type = Marker.CUBE
        mk.action = Marker.ADD
        mk.pose.position.x = float(cx)
        mk.pose.position.y = float(cy)
        mk.pose.position.z = float(z_center)
        qx,qy,qz,qw = rot_to_quat_z(yaw_deg)
        mk.pose.orientation.x = qx
        mk.pose.orientation.y = qy
        mk.pose.orientation.z = qz
        mk.pose.orientation.w = qw
        mk.scale.x = float(w)
        mk.scale.y = float(h)
        mk.scale.z = float(size_z)
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = rgba
        mk.lifetime = rclpy.duration.Duration(seconds=self.marker_lifetime).to_msg()
        return mk

    def cb_cloud(self, msg: PointCloud2):
        xyz = self.cloud_to_xyz(msg)

        # 비어있는 경우에는 empty PoseArray 발행
        pa1=PoseArray(); pa2=PoseArray(); pa3=PoseArray(); pa4=PoseArray()
        pa1.header=msg.header; pa2.header=msg.header; pa3.header=msg.header; pa4.header=msg.header
        ma1=MarkerArray(); ma2=MarkerArray(); ma3=MarkerArray(); ma4=MarkerArray()

        if xyz.size == 0:
            self.pub_p1.publish(pa1); self.pub_p2.publish(pa2); self.pub_p3.publish(pa3); self.pub_p4.publish(pa4)
            self.pub_m1.publish(ma1); self.pub_m2.publish(ma2); self.pub_m3.publish(ma3); self.pub_m4.publish(ma4)
            return

        xy = xyz[:, :2]
        z  = xyz[:, 2]
        clusters = self.cluster_xy(xy)

        for i, ids in enumerate(clusters):
            pts2 = xy[ids]
            zc = z[ids]
            zmin, zmax = float(np.min(zc)), float(np.max(zc))
            size_z = max(0.01, zmax - zmin)
            z_center = 0.5*(zmin+zmax)

            # 4 criterion(size, nearest, inlier, variances)
            try:
                cx, cy, w, h, a = self.lshape.fit_area(pts2.astype(np.float32))
                cx2, cy2, w2, h2, a2 = self.lshape.fit_nearest(pts2.astype(np.float32))
                cx3, cy3, w3, h3, a3 = self.lshape.fit_inlier(pts2.astype(np.float32))
                cx4, cy4, w4, h4, a4 = self.lshape.fit_variances(pts2.astype(np.float32))
            except Exception as e:
                self.get_logger().warn(f"Lshape error: {e}")
                continue

            # min_box_xy filter
            def valid(w, h): return (w>0 and h>0) and not (w<self.min_box_xy and h<self.min_box_xy)

            if valid(w,h):
                ma1.markers.append(self.make_cube(msg.header, i, "obb_area",      cx, cy, w, h, a, z_center, size_z, (1.0,0.0,0.0,0.3)))
                p=Pose(); p.position.x=cx; p.position.y=cy; p.position.z=z_center
                qx,qy,qz,qw=rot_to_quat_z(a); p.orientation.x=qx; p.orientation.y=qy; p.orientation.z=qz; p.orientation.w=qw
                pa1.poses.append(p)

            if valid(w2,h2):
                ma2.markers.append(self.make_cube(msg.header, i, "obb_nearest",   cx2, cy2, w2, h2, a2, z_center, size_z, (0.0,1.0,0.0,0.3)))
                p=Pose(); p.position.x=cx2; p.position.y=cy2; p.position.z=z_center
                qx,qy,qz,qw=rot_to_quat_z(a2); p.orientation.x=qx; p.orientation.y=qy; p.orientation.z=qz; p.orientation.w=qw
                pa2.poses.append(p)

            if valid(w3,h3):
                ma3.markers.append(self.make_cube(msg.header, i, "obb_inlier",    cx3, cy3, w3, h3, a3, z_center, size_z, (0.0,0.0,1.0,0.3)))
                p=Pose(); p.position.x=cx3; p.position.y=cy3; p.position.z=z_center
                qx,qy,qz,qw=rot_to_quat_z(a3); p.orientation.x=qx; p.orientation.y=qy; p.orientation.z=qz; p.orientation.w=qw
                pa3.poses.append(p)

            if valid(w4,h4):
                ma4.markers.append(self.make_cube(msg.header, i, "obb_variance",  cx4, cy4, w4, h4, a4, z_center, size_z, (1.0,1.0,0.0,0.3)))
                p=Pose(); p.position.x=cx4; p.position.y=cy4; p.position.z=z_center
                qx,qy,qz,qw=rot_to_quat_z(a4); p.orientation.x=qx; p.orientation.y=qy; p.orientation.z=qz; p.orientation.w=qw
                pa4.poses.append(p)

        # publish
        self.pub_p1.publish(pa1); self.pub_p2.publish(pa2); self.pub_p3.publish(pa3); self.pub_p4.publish(pa4)
        self.pub_m1.publish(ma1); self.pub_m2.publish(ma2); self.pub_m3.publish(ma3); self.pub_m4.publish(ma4)

def main():
    rclpy.init()
    node = OBBDetectorPy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
