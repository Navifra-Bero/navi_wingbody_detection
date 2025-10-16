#pragma once

#include "navi_wingbody_detection/lshaped_fitting.hpp"

#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core.hpp>

// 추가 include
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>


namespace navi_detection {

class ObjectDetector : public rclcpp::Node {
public:
  explicit ObjectDetector(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~ObjectDetector() override;

private:
  // === 콜백: 최신 프레임 저장만 (즉시 return) ===
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  // === 워커: 최신 프레임 1장만 꺼내 처리 ===
  void workerLoop();
  void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg);
  static inline double to_rad(double deg) { return deg * M_PI / 180.0; }

  // === 최신 프레임 보관 (queue 대신 마지막 한 장만) ===
  std::shared_ptr<const sensor_msgs::msg::PointCloud2> latest_msg_;
  std::mutex latest_mtx_;
  std::condition_variable latest_cv_;
  std::atomic<bool> stop_{false};
  std::thread worker_;

  // === L-shape Fitter ===
  double lshape_dtheta_deg_;
  double lshape_inlier_threshold_;
  double lshape_min_dist_nearest_;

  // === 파라미터 ===
  std::string input_topic_;
  std::string marker_topic_;
  std::string marker_topic_2;
  std::string marker_topic_3;
  std::string marker_topic_4;
  std::string cluster_topic_;

  double cluster_tolerance_;
  int    min_cluster_size_;
  int    max_cluster_size_;
  double marker_lifetime_;
  double min_box_xy_;
  int    max_markers_delete_batch_;
  double voxel_leaf_;        // VoxelGrid leaf size (0 → 비활성)
  bool   viz_reliable_;      // Marker pub Reliability (true: Reliable)
  int    viz_depth_;         // Marker pub KeepLast depth
  bool   cluster_reliable_;  // Cluster pub Reliability
  int    cluster_depth_;     // Cluster pub KeepLast depth

  // 클래스 멤버 추가 (토픽명)
  std::string poses_topic_;
  std::string poses_topic_2_;
  std::string poses_topic_3_;
  std::string poses_topic_4_;

  // 클래스 멤버 추가 (퍼블리셔)
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_poses_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_poses_2_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_poses_3_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_poses_4_;

  // === ROS2 I/O ===
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_2;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_3;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_4;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_clusters_;

  // === 상태 ===
  std::size_t last_marker_count_{0};
  std::mutex pub_mutex_; // 필요 시 퍼블리시 보호용
};

} // namespace navi_detection
