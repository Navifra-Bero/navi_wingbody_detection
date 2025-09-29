#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>

#include <vector>
#include <string>
#include <unordered_map>

class DbscanObbNode : public rclcpp::Node {
public:
  DbscanObbNode();

private:
  // --- Types ---
  struct Obb {
    Eigen::Vector3f center;    // world coords
    Eigen::Vector3f size;      // (dx, dy, dz)
    Eigen::Matrix3f R;         // orientation matrix
  };

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void dbscan(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
              std::vector<int>& labels);
  Obb computeOBB(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                 const std::vector<int>& indices);
  visualization_msgs::msg::Marker makeMarkerBox(const Obb& obb,
                                                const std_msgs::msg::Header& hdr,
                                                int id);


  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_;

  double eps_;
  int    min_pts_;
  int    max_clusters_;
  std::string input_topic_;
  std::string marker_topic_;
  std::string frame_override_;
  double marker_lifetime_sec_;
  int last_marker_count_ = 0;

  static constexpr int UNVISITED = -1;
  static constexpr int NOISE     = -2;
};
