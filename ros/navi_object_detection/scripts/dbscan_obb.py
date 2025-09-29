#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from typing import Tuple, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs_py import point_cloud2 as pc2
from sklearn.cluster import DBSCAN


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """
    roll=pitch=0 가정, yaw만 quaternion으로 변환
    returns (x,y,z,w)
    """
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    return (0.0, 0.0, sy, cy)


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """
    아주 가벼운 numpy 기반 voxel 그리드 다운샘플.
    points: (N,3) float32
    voxel:  voxel size [m]
    """
    if points.shape[0] == 0 or voxel <= 1e-6:
        return points
    # 음수 좌표 대응을 위해 floor 대신 round 사용
    keys = np.round(points / voxel).astype(np.int64)
    # unique voxel index의 첫 포인트만 취함
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]

def _clean_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    finite = np.isfinite(points).all(axis=1)
    return points[finite]



def obb_from_xy_pca(pts_xyz: np.ndarray, robust_percentile: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    - 지면 제거된 상황 가정(Z-up). roll/pitch = 0, yaw만 추정.
    - XY 2D PCA로 yaw 계산 → 축 뒤집힘에 강함.
    - 크기(size)는 퍼센타일 기반으로 최소/최대의 튐을 억제.
    returns center(3,), size(3,), yaw
    """
    n = pts_xyz.shape[0]
    if n < 3:
        center = pts_xyz.mean(axis=0) if n > 0 else np.zeros(3, np.float32)
        size = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        yaw = 0.0
        return center.astype(np.float32), size, float(yaw)

    xy = pts_xyz[:, :2].astype(np.float64, copy=False)
    mu = xy.mean(axis=0)            # (2,)
    X = xy - mu
    cov = (X.T @ X) / max(1, X.shape[0])
    evals, evecs = np.linalg.eigh(cov)  # 2x2, 오름차순
    order = np.argsort(evals)[::-1]     # 큰 고유값이 전방축
    R2 = evecs[:, order]                # (2,2)
    # yaw: 전방축을 x로
    yaw = float(math.atan2(R2[1, 0], R2[0, 0]))

    # 로컬 좌표로 투영
    Q = X @ R2  # (N,2)
    if robust_percentile:
        lo = np.percentile(Q, 5, axis=0)
        hi = np.percentile(Q, 95, axis=0)
    else:
        lo = Q.min(axis=0)
        hi = Q.max(axis=0)

    size_xy = (hi - lo).astype(np.float64)
    # z 범위 (퍼센타일)
    z = pts_xyz[:, 2].astype(np.float64)
    if robust_percentile:
        z_lo, z_hi = np.percentile(z, 5), np.percentile(z, 95)
    else:
        z_lo, z_hi = z.min(), z.max()
    size_z = max(0.05, float(z_hi - z_lo))

    # 월드 센터 (XY는 로컬 중앙, Z는 평균)
    center_local = 0.5 * (lo + hi)
    center_xy_world = mu + center_local @ R2.T
    center_z_world = float(0.5 * (z_lo + z_hi)) 
    center = np.array([center_xy_world[0], center_xy_world[1], center_z_world], dtype=np.float32)
    size = np.array([
        max(0.05, float(size_xy[0])),
        max(0.05, float(size_xy[1])),
        size_z
    ], dtype=np.float32)
    return center, size, yaw


class BoxSmoother:
    """
    프레임 간 OBB 스무더.
    - 최근 프레임 박스들과 최근접 매칭 (간단 greedy, 거리 임계치 내)
    - center/size/yaw에 지수평활 적용
    - 외부 의존성 없이 numpy만 사용
    """
    def __init__(self, alpha_pos=0.3, alpha_size=0.4, alpha_yaw=0.3, match_radius=1.5):
        self.alpha_pos = float(alpha_pos)
        self.alpha_size = float(alpha_size)
        self.alpha_yaw = float(alpha_yaw)
        self.match_radius = float(match_radius)
        self.prev_centers = None  # (M,3)
        self.prev_boxes = []      # [{'center':(3,), 'size':(3,), 'yaw':float}, ...]

    @staticmethod
    def _wrap_yaw(y):
        return (y + math.pi) % (2 * math.pi) - math.pi

    def _pairwise_dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # A:(N,3), B:(M,3) → (N,M)
        # ||A-B||^2 = |A|^2 + |B|^2 - 2 A·B
        a2 = np.sum(A*A, axis=1, keepdims=True)        # (N,1)
        b2 = np.sum(B*B, axis=1, keepdims=True).T      # (1,M)
        dist2 = a2 + b2 - 2.0 * (A @ B.T)              # (N,M)
        dist2 = np.maximum(dist2, 0.0)
        return np.sqrt(dist2, dtype=np.float64)

    def smooth(self, centers: np.ndarray, sizes: np.ndarray, yaws: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = centers.shape[0]
        if N == 0:
            self.prev_centers = None
            self.prev_boxes = []
            return centers, sizes, yaws

        if self.prev_centers is None or len(self.prev_boxes) == 0:
            # 첫 프레임은 그대로
            self.prev_centers = centers.copy()
            self.prev_boxes = [{'center': centers[i].copy(),
                                'size': sizes[i].copy(),
                                'yaw': float(yaws[i])} for i in range(N)]
            return centers, sizes, yaws

        # 최근접 매칭
        D = self._pairwise_dist(centers, self.prev_centers)  # (N,M)
        M = D.shape[1]
        sm_centers = centers.copy()
        sm_sizes = sizes.copy()
        sm_yaws = yaws.copy()

        used_prev = set()
        # 간단 greedy 매칭: 각 현재 박스 i에 대해 가장 가까운 이전 박스 j 선택
        for i in range(N):
            j = int(np.argmin(D[i]))
            if j >= M:
                continue
            if D[i, j] > self.match_radius:
                continue
            if j in used_prev:
                continue
            used_prev.add(j)

            p = self.prev_boxes[j]
            # EMA
            sm_centers[i] = (1.0 - self.alpha_pos) * centers[i] + self.alpha_pos * p['center']
            sm_sizes[i] = (1.0 - self.alpha_size) * sizes[i] + self.alpha_size * p['size']
            dyaw = self._wrap_yaw(yaws[i] - p['yaw'])
            sm_yaws[i] = self._wrap_yaw(p['yaw'] + (1.0 - self.alpha_yaw) * dyaw)

        # 상태 갱신
        self.prev_centers = sm_centers.copy()
        self.prev_boxes = [{'center': sm_centers[i].copy(),
                            'size': sm_sizes[i].copy(),
                            'yaw': float(sm_yaws[i])} for i in range(N)]
        return sm_centers, sm_sizes, sm_yaws


class DBSCANOBBNode(Node):
    def __init__(self):
        super().__init__('dbscan_obb_node')

        # ---- Parameters (런치/CLI에서 override 가능) ----
        self.declare_parameter('input_topic', '/patchworkpp/nonground')
        self.declare_parameter('marker_topic', '/dbscan/obb_markers')
        self.declare_parameter('eps', 0.6)
        self.declare_parameter('min_samples', 20)
        self.declare_parameter('max_cluster', 1000)
        self.declare_parameter('voxel', 0.02)            # 다운샘플 크기[m]
        self.declare_parameter('percentile', True)       # OBB robust percentile 사용 여부
        self.declare_parameter('smooth.alpha_pos', 0.3)
        self.declare_parameter('smooth.alpha_size', 0.4)
        self.declare_parameter('smooth.alpha_yaw', 0.3)
        self.declare_parameter('smooth.match_radius', 1.5)
        self.declare_parameter('marker_lifetime', 0.0)
        self.declare_parameter('min_box_xy', 0.1)       # 너무 작은 박스 제거(가로/세로)
        self.declare_parameter('min_points_after_voxel', 10)  # 다운샘플 후 최소 포인트

        in_topic = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('marker_topic').value

        self.eps = float(self.get_parameter('eps').value)
        self.min_samples = int(self.get_parameter('min_samples').value)
        self.max_cluster = int(self.get_parameter('max_cluster').value)
        self.voxel = float(self.get_parameter('voxel').value)
        self.use_percentile = bool(self.get_parameter('percentile').value)
        self.marker_lifetime = float(self.get_parameter('marker_lifetime').value)
        self.min_box_xy = float(self.get_parameter('min_box_xy').value)
        self.min_pts_after_voxel = int(self.get_parameter('min_points_after_voxel').value)

        self.smoother = BoxSmoother(
            alpha_pos=float(self.get_parameter('smooth.alpha_pos').value),
            alpha_size=float(self.get_parameter('smooth.alpha_size').value),
            alpha_yaw=float(self.get_parameter('smooth.alpha_yaw').value),
            match_radius=float(self.get_parameter('smooth.match_radius').value),
        )

        # ---- IO ----
        self.sub = self.create_subscription(PointCloud2, in_topic, self.callback, 10)
        self.pub = self.create_publisher(MarkerArray, out_topic, 10)
        self._last_marker_count = 0

        self.get_logger().info(
            f"[DBSCAN-OBB] in='{in_topic}', out='{out_topic}', "
            f"eps={self.eps}, min_samples={self.min_samples}, voxel={self.voxel}, "
            f"max_cluster={self.max_cluster}"
        )

    # PointCloud2 → numpy (robust: structured/2D 모두 처리)
    def _cloud_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        arr = pc2.read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            reshape_organized_cloud=False
        )
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if arr.dtype.fields is None:
            # 일반 배열 (N,3) 또는 (3,)
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 3)
            return arr.astype(np.float32, copy=False)
        else:
            # 구조화 배열
            return np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float32, copy=False)

    def callback(self, msg: PointCloud2):
        cloud = self._cloud_to_numpy(msg)
        if cloud.shape[0] == 0:
            return
        
        cloud = _clean_points(cloud)
        if cloud.shape[0] == 0:
            return

        if self.voxel > 1e-6:
            cloud_ds = voxel_downsample(cloud, self.voxel)
        else:
            cloud_ds = cloud

        if cloud_ds.shape[0] < self.min_pts_after_voxel:
            return

        # --- DBSCAN ---
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        labels = db.fit_predict(cloud_ds)

        uniq = np.unique(labels)
        uniq = uniq[uniq >= 0]  # -1은 noise
        if uniq.size == 0:
            # 이전 마커 지우기
            if self._last_marker_count > 0:
                marr = MarkerArray()
                for kill in range(self._last_marker_count):
                    delm = Marker()
                    delm.header = msg.header
                    delm.ns = "obb"
                    delm.id = kill
                    delm.action = Marker.DELETE
                    marr.markers.append(delm)
                self.pub.publish(marr)
                self._last_marker_count = 0
            return

        centers: List[np.ndarray] = []
        sizes:   List[np.ndarray] = []
        yaws:    List[float] = []

        for cid in uniq:
            pts = cloud_ds[labels == cid]
            if pts.shape[0] < self.min_samples:
                continue

            # 2D-PCA 기반 견고 OBB
            c, s, yaw = obb_from_xy_pca(pts, robust_percentile=self.use_percentile)

            # 너무 작은 박스 제거(잡음 필터)
            if s[0] < self.min_box_xy and s[1] < self.min_box_xy:
                continue

            centers.append(c)
            sizes.append(s)
            yaws.append(yaw)

            if len(centers) >= self.max_cluster:
                break

        if len(centers) == 0:
            return

        centers = np.stack(centers, axis=0)
        sizes   = np.stack(sizes, axis=0)
        yaws    = np.asarray(yaws, dtype=np.float32)

        # --- 프레임 간 스무딩 ---
        centers, sizes, yaws = self.smoother.smooth(centers, sizes, yaws)

        # --- MarkerArray 발행 ---
        marr = MarkerArray()
        for i in range(centers.shape[0]):
            mk = Marker()
            mk.header = msg.header
            mk.ns = "obb"
            mk.id = i
            mk.type = Marker.CUBE
            mk.action = Marker.ADD

            mk.pose.position.x = float(centers[i, 0])
            mk.pose.position.y = float(centers[i, 1])
            mk.pose.position.z = float(centers[i, 2])

            qx, qy, qz, qw = yaw_to_quaternion(float(yaws[i]))
            mk.pose.orientation.x = qx
            mk.pose.orientation.y = qy
            mk.pose.orientation.z = qz
            mk.pose.orientation.w = qw

            mk.scale.x = float(max(0.01, sizes[i, 0]))
            mk.scale.y = float(max(0.01, sizes[i, 1]))
            mk.scale.z = float(max(0.01, sizes[i, 2]))

            mk.color.r = 0.1
            mk.color.g = 0.8
            mk.color.b = 0.2
            mk.color.a = 0.5

            mk.lifetime.sec = int(self.marker_lifetime)
            mk.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)

            marr.markers.append(mk)

        # 남은 마커 삭제
        for kill in range(centers.shape[0], self._last_marker_count):
            delm = Marker()
            delm.header = msg.header
            delm.ns = "obb"
            delm.id = kill
            delm.action = Marker.DELETE
            marr.markers.append(delm)

        self._last_marker_count = centers.shape[0]
        self.pub.publish(marr)


def main(args=None):
    rclpy.init(args=args)
    node = DBSCANOBBNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
