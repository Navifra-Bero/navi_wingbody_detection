#include <rclcpp/rclcpp.hpp>
#include "navi_object_detection/object_detector.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<navi_detection::ObjectDetector>(); // ROS2용 생성자
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
