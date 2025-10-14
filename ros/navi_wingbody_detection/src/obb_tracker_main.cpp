#include "navi_wingbody_detection/object_tracker.hpp" // include 경로 수정
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<navi_detection::ObjectTracker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}