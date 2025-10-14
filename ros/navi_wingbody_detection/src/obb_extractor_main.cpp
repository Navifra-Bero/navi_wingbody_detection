#include "navi_wingbody_detection/object_detector.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  // 메시지 복사 줄이기
  rclcpp::NodeOptions opts;
  // opts.use_intra_process_comms(true);

  // 멀티스레드 실행기 (다른 콜백과 병렬성 확보용)
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);

  auto node = std::make_shared<navi_detection::ObjectDetector>(opts);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
