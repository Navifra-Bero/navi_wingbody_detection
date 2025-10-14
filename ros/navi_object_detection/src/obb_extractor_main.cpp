#include "navi_object_detection/object_detector.hpp" // object_detector.hpp 경로에 맞게 수정
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  
  // 1. 멀티스레드 실행기 생성 (예: 4개의 스레드 사용)
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 8);
  
  auto node = std::make_shared<navi_detection::ObjectDetector>();
  
  // 2. 실행기에 노드 추가
  executor.add_node(node);
  
  // 3. spin 대신 executor.spin() 사용
  executor.spin();

  rclcpp::shutdown();
  return 0;
}