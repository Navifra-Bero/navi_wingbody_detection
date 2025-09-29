#pragma once

#include "navi_object_detection/lshaped_fitting.hpp" // 새로 만든 헤더파일 include

#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <mutex>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core.hpp> // cv::Point2f 사용을 위해 추가

namespace navi_detection {

class ObjectDetector : public rclcpp::Node {
public:
  explicit ObjectDetector(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~ObjectDetector() override;

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  std::unique_ptr<LShapedFIT> lsfitter_;

  // params
  std::string input_topic_;
  std::string marker_topic_;
  std::string cluster_topic_;
  double cluster_tolerance_;
  int    min_cluster_size_;
  int    max_cluster_size_;
  double marker_lifetime_;
  double min_box_xy_;
  int    max_markers_delete_batch_;

  // --- 여기부터 ROS2 메시지 타입 문법 수정 ---
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_clusters_;
  // --- 수정 끝 ---

  // housekeeping
  std::size_t last_marker_count_{0};
  std::mutex pub_mutex_;
};

} // namespace navi_detection