#pragma once

#include "navi_object_detection/sort_tracker.hpp" // 새로 만든 헤더파일 include

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <memory>
#include <vector>

// class Sort; // 이제 전방 선언이 아닌, 실제 헤더를 include 함

namespace navi_detection {

class ObjectTracker : public rclcpp::Node {
public:
  explicit ObjectTracker(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~ObjectTracker() override;

private:
  void markersCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg);

  std::unique_ptr<Sort> mot_tracker_;

  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr sub_detections_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_tracks_;

  std::string input_topic_;
  std::string output_topic_;
  int max_age_;
  int min_hits_;
  double association_threshold_;
};

} // namespace navi_detection