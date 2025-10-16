#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>

class PcTfRepubNode : public rclcpp::Node {
public:
  PcTfRepubNode() : Node("pc_tf_repub") {
    // 입력/출력 토픽 & 출력 프레임
    in_topic_  = declare_parameter<std::string>("input_topic",  "/camera/depth/points");
    out_topic_ = declare_parameter<std::string>("output_topic", "/camera/depth/tf_ch_points");
    out_frame_ = declare_parameter<std::string>("output_frame", "vanjee_lidar");

    // depth_to_color: 3x3 회전 + 3x1 평행이동
    std::vector<double> Rdc_flat = declare_parameter<std::vector<double>>(
      "depth_to_color.rotation",
      // row-major 9개 (예: 질문에 준 값 그대로)
      { 0.9944840669631958,  0.00024602707708254457, -0.002201989758759737,
       -0.000013758952263742685, 0.9944864511489868,  0.10486499965190887,
        0.0022156487684696913,  -0.10486471652984619, 0.994484007358551 }
    );
    std::vector<double> tdc = declare_parameter<std::vector<double>>(
      "depth_to_color.translation",
      { -0.03218218994140625, -0.0005416243076324463, 0.0025637595653533935 }
    );

    // lidar_to_camera 4x4 (질문에서 "camera_to_lidar" 라고 표기됐지만 실제는 lidar→camera 라고 했음)
    std::vector<double> Tlc_flat = declare_parameter<std::vector<double>>(
      "lidar_to_camera.row_major",
      {
        0.03028327, -0.99951961, -0.00659278, -0.03088140,
        0.00134207,  0.00663646, -0.99997708,  0.11743733,
        0.99954046,  0.03027373,  0.00154240, -0.08846909,
        0.0,         0.0,         0.0,         1.0
      }
    );

    // 행렬 구성
    Eigen::Matrix3d Rdc;
    Rdc << Rdc_flat[0], Rdc_flat[1], Rdc_flat[2],
           Rdc_flat[3], Rdc_flat[4], Rdc_flat[5],
           Rdc_flat[6], Rdc_flat[7], Rdc_flat[8];
    Eigen::Vector3d tdc_v(tdc[0], tdc[1], tdc[2]);

    Eigen::Matrix4d Tdc = Eigen::Matrix4d::Identity();     // depth->color (optical)
    Tdc.block<3,3>(0,0) = Rdc;
    Tdc.block<3,1>(0,3) = tdc_v;

    Eigen::Matrix4d Tlc;                                   // lidar->camera
    Tlc << Tlc_flat[0],  Tlc_flat[1],  Tlc_flat[2],  Tlc_flat[3],
           Tlc_flat[4],  Tlc_flat[5],  Tlc_flat[6],  Tlc_flat[7],
           Tlc_flat[8],  Tlc_flat[9],  Tlc_flat[10], Tlc_flat[11],
           Tlc_flat[12], Tlc_flat[13], Tlc_flat[14], Tlc_flat[15];

    // camera->lidar = (lidar->camera)^-1
    Eigen::Matrix4d Tcl = Tlc.inverse();

    // 최종: depth -> lidar
    T_depth_to_lidar_ = Tcl * Tdc;

    RCLCPP_INFO(get_logger(), "[pc_tf_repub] in='%s' → out='%s' | out_frame='%s'",
                in_topic_.c_str(), out_topic_.c_str(), out_frame_.c_str());
    RCLCPP_INFO(get_logger(), "T_depth_to_lidar = \n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f",
                T_depth_to_lidar_(0,0),T_depth_to_lidar_(0,1),T_depth_to_lidar_(0,2),T_depth_to_lidar_(0,3),
                T_depth_to_lidar_(1,0),T_depth_to_lidar_(1,1),T_depth_to_lidar_(1,2),T_depth_to_lidar_(1,3),
                T_depth_to_lidar_(2,0),T_depth_to_lidar_(2,1),T_depth_to_lidar_(2,2),T_depth_to_lidar_(2,3),
                T_depth_to_lidar_(3,0),T_depth_to_lidar_(3,1),T_depth_to_lidar_(3,2),T_depth_to_lidar_(3,3));

    // QoS는 센서 데이터 권장
    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(out_topic_, pub_qos);

    auto sub_qos = rclcpp::SensorDataQoS().keep_last(1).best_effort();
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      in_topic_, sub_qos,
      std::bind(&PcTfRepubNode::cb, this, std::placeholders::_1));
  }

private:
  void cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // PointCloud2 -> PCL
    pcl::PointCloud<pcl::PointXYZ> in_pc;
    pcl::fromROSMsg(*msg, in_pc);

    pcl::PointCloud<pcl::PointXYZ> out_pc;
    out_pc.reserve(in_pc.size());

    // 변환
    const Eigen::Matrix3d R = T_depth_to_lidar_.block<3,3>(0,0);
    const Eigen::Vector3d t = T_depth_to_lidar_.block<3,1>(0,3);

    for (const auto &p : in_pc.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      Eigen::Vector3d q = R * Eigen::Vector3d(p.x, p.y, p.z) + t;
      pcl::PointXYZ o;
      o.x = static_cast<float>(q.x());
      o.y = static_cast<float>(q.y());
      o.z = static_cast<float>(q.z());
      out_pc.points.push_back(o);
    }
    out_pc.width  = static_cast<uint32_t>(out_pc.points.size());
    out_pc.height = 1;
    out_pc.is_dense = false;

    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(out_pc, out_msg);
    out_msg.header = msg->header;
    out_msg.header.frame_id = out_frame_;
    pub_->publish(out_msg);
  }

  // params
  std::string in_topic_, out_topic_, out_frame_;
  Eigen::Matrix4d T_depth_to_lidar_;

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PcTfRepubNode>());
  rclcpp::shutdown();
  return 0;
}
