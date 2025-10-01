#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <vector>
#include <string>

namespace navi_tracking {

// ---------- 유틸: BEV AABB IoU ----------
struct Box2D {
  double x1, y1, x2, y2; // axis-aligned in BEV
};
inline double IoU(const Box2D& a, const Box2D& b) {
  const double xx1 = std::max(a.x1, b.x1);
  const double yy1 = std::max(a.y1, b.y1);
  const double xx2 = std::min(a.x2, b.x2);
  const double yy2 = std::min(a.y2, b.y2);
  const double w = std::max(0.0, xx2 - xx1);
  const double h = std::max(0.0, yy2 - yy1);
  const double inter = w * h;
  const double areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const double areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  const double uni = areaA + areaB - inter + 1e-12;
  return inter / uni;
}

// ---------- 간단 Hungarian(Munkres) ----------
std::vector<std::pair<int,int>> hungarian_min_cost(const Eigen::MatrixXd& cost);

// ---------- SORT와 동일한 상태정의 ----------
/*
 state x = [cx, cy, s, r, vx, vy, vs]^T
   - s=area(w*h), r=aspect(w/h)
 Z(meas) = [cx, cy, s, r]^T
 yaw, z, z_size는 별도 EMA로 관리
*/
class SortKalman {
public:
  SortKalman();
  void init(const Eigen::Vector4d& z0);
  void predict();
  void update(const Eigen::Vector4d& z);

  Eigen::VectorXd x;   // 7x1
  Eigen::MatrixXd P;   // 7x7

private:
  Eigen::MatrixXd F_, H_, Q_, R_;
};

// ---------- 트랙 ----------
struct Track {
  int id{-1};
  int age{0};
  int time_since_update{0};
  int hit_streak{0};

  SortKalman kf;

  // 보조(3D 확장용)
  double yaw{0.0};      // rad
  double z_center{0.0};
  double z_size{0.1};

  // EMA 파라미터
  double yaw_alpha{0.4};
  double z_alpha{0.4};

  // 최신 박스 (AABB for BEV IoU)
  Box2D last_aabb;

  // from state -> AABB
  Box2D getAABB() const;
  void setFromMeasurement(const Eigen::Vector4d& z, double meas_yaw, double meas_zc, double meas_zs);
};

// ---------- 메시지를 받아 추적 ----------
class ObjectTrackerNode : public rclcpp::Node {
public:
  explicit ObjectTrackerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~ObjectTrackerNode() override = default;

private:
  void markersCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg);

  // 파라미터
  std::string input_topic_;
  std::string output_topic_;
  double iou_threshold_;
  int    max_age_;
  int    min_hits_;
  double marker_lifetime_;

  // IO
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr sub_markers_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_tracked_;

  // 상태
  int next_id_{1};
  std::vector<Track> tracks_;
  std::mutex mtx_;

  // 헬퍼
  static Eigen::Vector4d detToZ(double cx, double cy, double w, double h);
  static void zToWH(const Eigen::Vector4d& z, double& w, double& h);
  static Box2D toAABB(double cx, double cy, double w, double h);
  static double quatToYaw(double qx, double qy, double qz, double qw);
};

} // namespace navi_tracking
