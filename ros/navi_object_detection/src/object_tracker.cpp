#include "navi_object_detection/object_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <visualization_msgs/msg/marker.hpp>

using std::placeholders::_1;
namespace nt = navi_tracking;

// ================= Hungarian (작게 구현) =================
std::vector<std::pair<int, int>> nt::hungarian_min_cost(const Eigen::MatrixXd& cost) {
  // 간단한 Munkres 구현 (정사각/직사각 모두 지원). N,M 작다고 가정.
  // 참고: 성능 최적화 목적 아님.
  const int N = cost.rows();
  const int M = cost.cols();
  const int L = std::max(N, M);

  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(L, L);
  C.setConstant(0.0);
  C.topLeftCorner(N, M) = cost;

  // Row/Col reduction
  for (int i = 0; i < L; ++i) {
    double rmin = C.row(i).minCoeff();
    C.row(i).array() -= rmin;
  }
  for (int j = 0; j < L; ++j) {
    double cmin = C.col(j).minCoeff();
    C.col(j).array() -= cmin;
  }

  std::vector<int> rowOfStar(L, -1), colOfStar(L, -1);
  std::vector<int> rowOfPrime(L, -1), colCover(L, 0), rowCover(L, 0);

  // Step2: star zeros greedily
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      if (C(i, j) == 0 && rowOfStar[i] == -1 && colOfStar[j] == -1) {
        rowOfStar[i] = j;
        colOfStar[j] = i;
        break;
      }
    }
  }

  auto coverColumnsWithStarredZero = [&]() {
    int count = 0;
    for (int j = 0; j < L; ++j) {
      if (colOfStar[j] != -1) {
        colCover[j] = 1;
        ++count;
      }
    }
    return count;
  };

  int step = 3;
  while (true) {
    if (step == 3) {
      int covered = coverColumnsWithStarredZero();
      if (covered >= L) break;
      step = 4;
    } else if (step == 4) {
      bool done = false;
      while (!done) {
        int zrow = -1, zcol = -1;
        // find uncovered zero
        for (int i = 0; i < L && zrow == -1; ++i) {
          if (rowCover[i]) continue;
          for (int j = 0; j < L; ++j) {
            if (colCover[j]) continue;
            if (C(i, j) == 0) {
              zrow = i;
              zcol = j;
              break;
            }
          }
        }
        if (zrow == -1) {
          step = 6;  // adjust matrix
          done = true;
        } else {
          rowOfPrime[zrow] = zcol;
          if (rowOfStar[zrow] == -1) {
            // augmenting path
            step = 5;
            // build path
            std::vector<std::pair<int, int>> path;
            path.emplace_back(zrow, zcol);
            bool found = true;
            while (found) {
              int r = colOfStar[path.back().second];
              if (r == -1) break;
              path.emplace_back(r, path.back().second);
              int c = rowOfPrime[r];
              path.emplace_back(r, c);
            }
            for (auto& p : path) {
              if (rowOfStar[p.first] == p.second) {
                // unstar
                rowOfStar[p.first]  = -1;
                colOfStar[p.second] = -1;
              } else {
                // star
                rowOfStar[p.first]  = p.second;
                colOfStar[p.second] = p.first;
              }
            }
            // reset covers/primes
            std::fill(rowCover.begin(), rowCover.end(), 0);
            std::fill(colCover.begin(), colCover.end(), 0);
            std::fill(rowOfPrime.begin(), rowOfPrime.end(), -1);
            step = 3;
            done = true;
          } else {
            // cover this row, uncover the star's column
            rowCover[zrow]            = 1;
            colCover[rowOfStar[zrow]] = 0;
          }
        }
      }
    } else if (step == 6) {
      // add min uncovered to covered rows; subtract from uncovered columns
      double minval = std::numeric_limits<double>::infinity();
      for (int i = 0; i < L; ++i)
        if (!rowCover[i])
          for (int j = 0; j < L; ++j)
            if (!colCover[j]) minval = std::min(minval, C(i, j));
      for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
          if (rowCover[i]) C(i, j) += minval;
          if (!colCover[j]) C(i, j) -= minval;
        }
      }
      step = 4;
    }
  }

  std::vector<std::pair<int, int>> matches;
  for (int i = 0; i < N; ++i) {
    if (rowOfStar[i] >= 0 && rowOfStar[i] < M) {
      matches.emplace_back(i, rowOfStar[i]);
    }
  }
  return matches;
}

// ================= Kalman =================
nt::SortKalman::SortKalman() {
  x = Eigen::VectorXd::Zero(7);
  P = Eigen::MatrixXd::Identity(7, 7) * 10.0;

  F_ = Eigen::MatrixXd::Identity(7, 7);
  // constant-velocity on cx, cy, s
  F_(0, 4) = 1.0;  // cx += vx
  F_(1, 5) = 1.0;  // cy += vy
  F_(2, 6) = 1.0;  // s  += vs

  H_       = Eigen::MatrixXd::Zero(4, 7);
  H_(0, 0) = 1.0;
  H_(1, 1) = 1.0;
  H_(2, 2) = 1.0;
  H_(3, 3) = 1.0;

  Q_       = Eigen::MatrixXd::Identity(7, 7);
  Q_(4, 4) = Q_(5, 5) = Q_(6, 6) = 0.01;
  R_                             = Eigen::MatrixXd::Identity(4, 4);
  R_(2, 2)                       = 10.0;  // s, r noise 조금 크게
}

void nt::SortKalman::init(const Eigen::Vector4d& z0) {
  x.setZero();
  x.head<4>() = z0;
  P.setIdentity();                // 초기 공분산
  P.block<3, 3>(4, 4) *= 1000.0;  // 속도 큰 불확실성
  P *= 10.0;
}

void nt::SortKalman::predict() {
  x = F_ * x;
  P = F_ * P * F_.transpose() + Q_;
}

void nt::SortKalman::update(const Eigen::Vector4d& z) {
  // 표준 KF
  const Eigen::Vector4d y = z - H_ * x;
  const Eigen::MatrixXd S = H_ * P * H_.transpose() + R_;
  const Eigen::MatrixXd K = P * H_.transpose() * S.inverse();
  x                       = x + K * y;
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(7, 7);
  P                       = (I - K * H_) * P;
}

// ================= Track =================
nt::Box2D nt::Track::getAABB() const {
  // x-> w,h 복원
  const double s = kf.x(2), r = std::max(1e-6, kf.x(3));
  const double w  = std::sqrt(s * r);
  const double h  = s / std::max(1e-6, w);
  const double cx = kf.x(0), cy = kf.x(1);
  return nt::Box2D{cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0};
}

void nt::Track::setFromMeasurement(const Eigen::Vector4d& z,
                                   double meas_yaw,
                                   double meas_zc,
                                   double meas_zs) {
  if (id < 0) {
    // init
    kf.init(z);
    yaw      = meas_yaw;
    z_center = meas_zc;
    z_size   = meas_zs;
  } else {
    kf.update(z);
    // EMA for yaw/z
    yaw      = yaw_alpha * yaw + (1.0 - yaw_alpha) * meas_yaw;
    z_center = z_alpha * z_center + (1.0 - z_alpha) * meas_zc;
    z_size   = z_alpha * z_size + (1.0 - z_alpha) * meas_zs;
  }
  last_aabb = getAABB();
}

// ================= Node =================
static inline double clampAngle(double a) {
  while (a > M_PI) a -= 2 * M_PI;
  while (a < -M_PI) a += 2 * M_PI;
  return a;
}
double nt::ObjectTrackerNode::quatToYaw(double qx, double qy, double qz, double qw) {
  // Z-up 가정의 yaw
  const double siny_cosp = 2.0 * (qw * qz + qx * qy);
  const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
  return std::atan2(siny_cosp, cosy_cosp);
}

Eigen::Vector4d nt::ObjectTrackerNode::detToZ(double cx, double cy, double w, double h) {
  const double s = std::max(1e-6, w * h);
  const double r = std::max(1e-6, w / std::max(1e-6, h));
  return Eigen::Vector4d(cx, cy, s, r);
}
void nt::ObjectTrackerNode::zToWH(const Eigen::Vector4d& z, double& w, double& h) {
  const double s = std::max(1e-6, z(2));
  const double r = std::max(1e-6, z(3));
  w              = std::sqrt(s * r);
  h              = s / std::max(1e-6, w);
}
nt::Box2D nt::ObjectTrackerNode::toAABB(double cx, double cy, double w, double h) {
  return nt::Box2D{cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0};
}

nt::ObjectTrackerNode::ObjectTrackerNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("obb_sort_tracker_node", options) {
  input_topic_     = this->declare_parameter<std::string>("input_markers_topic", "/obb_markers");
  output_topic_    = this->declare_parameter<std::string>("output_tracked_topic", "/tracked_obb");
  iou_threshold_   = this->declare_parameter<double>("iou_threshold", 0.2);
  max_age_         = this->declare_parameter<int>("max_age", 3);
  min_hits_        = this->declare_parameter<int>("min_hits", 2);
  marker_lifetime_ = this->declare_parameter<double>("marker_lifetime", 0.2);

  sub_markers_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
      input_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&ObjectTrackerNode::markersCallback, this, _1));
  pub_tracked_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(output_topic_, 10);

  RCLCPP_INFO(this->get_logger(),
              "[OBB-SORT] in='%s' -> out='%s' (IoU>=%.2f, max_age=%d, min_hits=%d)",
              input_topic_.c_str(),
              output_topic_.c_str(),
              iou_threshold_,
              max_age_,
              min_hits_);
}

void nt::ObjectTrackerNode::markersCallback(
    const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
  std::lock_guard<std::mutex> lk(mtx_);
  // 1) 입력 detection 파싱 (CUBE만 사용)
  struct Det {
    Eigen::Vector4d z;
    nt::Box2D aabb;
    double yaw;
    double zc;
    double zs;
    visualization_msgs::msg::Marker m;
  };
  std::vector<Det> dets;
  dets.reserve(msg->markers.size());
  for (const auto& mk : msg->markers) {
    if (mk.action != visualization_msgs::msg::Marker::ADD) continue;
    if (mk.type != visualization_msgs::msg::Marker::CUBE) continue;

    const double cx = mk.pose.position.x;
    const double cy = mk.pose.position.y;
    const double w  = std::max(1e-3, (double)mk.scale.x);
    const double h  = std::max(1e-3, (double)mk.scale.y);
    const double zc = mk.pose.position.z;
    const double zs = std::max(1e-3, (double)mk.scale.z);

    const double yaw = clampAngle(quatToYaw(mk.pose.orientation.x,
                                            mk.pose.orientation.y,
                                            mk.pose.orientation.z,
                                            mk.pose.orientation.w));
    Det d;
    d.z    = detToZ(cx, cy, w, h);
    d.aabb = toAABB(cx, cy, w, h);  // IoU는 AABB 기준
    d.yaw  = yaw;
    d.zc   = zc;
    d.zs   = zs;
    d.m    = mk;
    dets.push_back(d);
  }

  // 2) 예측
  for (auto& t : tracks_) {
    t.kf.predict();
    t.age += 1;
    t.time_since_update += 1;
    t.last_aabb = t.getAABB();
  }

  // 3) 비용행렬(1 - IoU)
  const int N = (int)dets.size();
  const int M = (int)tracks_.size();
  std::vector<int> unmatched_dets, unmatched_trks;
  std::vector<std::pair<int, int>> matches;
  if (N == 0 && M == 0) {
    // nothing
  } else if (M == 0) {
    unmatched_dets.resize(N);
    std::iota(unmatched_dets.begin(), unmatched_dets.end(), 0);
  } else if (N == 0) {
    unmatched_trks.resize(M);
    std::iota(unmatched_trks.begin(), unmatched_trks.end(), 0);
  } else {
    Eigen::MatrixXd cost(N, M);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        const double iou = IoU(dets[i].aabb, tracks_[j].last_aabb);
        cost(i, j)       = 1.0 - iou;
      }
    }
    auto pairs = hungarian_min_cost(cost);

    std::vector<int> det_assigned(N, -1), trk_assigned(M, -1);
    for (auto& p : pairs) {
      int di = p.first, tj = p.second;
      const double iou = 1.0 - cost(di, tj);
      if (iou >= iou_threshold_) {
        matches.emplace_back(di, tj);
        det_assigned[di] = tj;
        trk_assigned[tj] = di;
      }
    }
    for (int i = 0; i < N; ++i)
      if (det_assigned[i] == -1) unmatched_dets.push_back(i);
    for (int j = 0; j < M; ++j)
      if (trk_assigned[j] == -1) unmatched_trks.push_back(j);
  }

  // 4) 매칭된 트랙 갱신
  for (auto& pr : matches) {
    const int di        = pr.first;
    const int tj        = pr.second;
    auto& t             = tracks_[tj];
    t.time_since_update = 0;
    t.hit_streak += 1;
    t.setFromMeasurement(dets[di].z, dets[di].yaw, dets[di].zc, dets[di].zs);
  }

  // 5) unmatched dets → 새 트랙 생성
  for (int di : unmatched_dets) {
    Track t;
    t.id = next_id_++;
    t.kf.init(dets[di].z);
    t.yaw               = dets[di].yaw;
    t.z_center          = dets[di].zc;
    t.z_size            = dets[di].zs;
    t.age               = 1;
    t.time_since_update = 0;
    t.hit_streak        = 1;
    t.last_aabb         = t.getAABB();
    tracks_.push_back(t);
  }

  // 6) 오래된 트랙 제거
  for (int i = (int)tracks_.size() - 1; i >= 0; --i) {
    if (tracks_[i].time_since_update > max_age_) {
      tracks_.erase(tracks_.begin() + i);
    }
  }

  // 7) 퍼블리시 (min_hits 충족 or 초반)
  visualization_msgs::msg::MarkerArray out;
  int out_id = 0;
  for (auto& t : tracks_) {
    const bool publishable = (t.time_since_update < 1) && (t.hit_streak >= min_hits_);
    if (!publishable) continue;

    // 상태 → w,h
    double w, h;
    zToWH(t.kf.x.head<4>(), w, h);

    // Marker (3D OBB)
    visualization_msgs::msg::Marker mk;
    mk.header = msg->markers.empty() ? std_msgs::msg::Header() : msg->markers.front().header;
    mk.ns     = "tracked_obb";
    mk.id     = out_id++;
    mk.type   = visualization_msgs::msg::Marker::CUBE;
    mk.action = visualization_msgs::msg::Marker::ADD;

    mk.pose.position.x = t.kf.x(0);
    mk.pose.position.y = t.kf.x(1);
    mk.pose.position.z = t.z_center;

    const double half = std::sqrt(1.0 - std::min(1.0, std::max(-1.0, std::cos(t.yaw))));
    // yaw → quat(Z-up)
    const double cy       = std::cos(t.yaw * 0.5);
    const double sy       = std::sin(t.yaw * 0.5);
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = sy;
    mk.pose.orientation.w = cy;

    mk.scale.x = std::max(0.01, w);
    mk.scale.y = std::max(0.01, h);
    mk.scale.z = std::max(0.01, t.z_size);

    mk.color.r = 0.2;
    mk.color.g = 0.6;
    mk.color.b = 1.0;
    mk.color.a = 0.6;

    mk.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
    out.markers.push_back(mk);
  }

  pub_tracked_->publish(out);
} // namespace navi_tracking