#include "patchworkpp/object_detector.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/filter.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <numeric>
#include <iostream>
#include <functional>

using std::placeholders::_1;

namespace navi_detection {

static inline visualization_msgs::msg::Marker makeDeleteMarker(const std_msgs::msg::Header& hdr, int id) {
    visualization_msgs::msg::Marker m;
    m.header = hdr; m.ns = "obb"; m.id = id;
    m.action = visualization_msgs::msg::Marker::DELETE;
    return m;
}

ObjectDetector::ObjectDetector(const rclcpp::NodeOptions& options)
: rclcpp::Node("obb_extractor_node", options)
{ 
  lsfitter_ = std::make_unique<LShapedFIT>();
  input_topic_  = this->declare_parameter<std::string>("input_topic",  "/patchworkpp/nonground");
  marker_topic_ = this->declare_parameter<std::string>("marker_topic", "/obb_markers");
  cluster_topic_ = this->declare_parameter<std::string>("cluster_topic", "/cluster/points");
  cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.6);
  min_cluster_size_  = this->declare_parameter<int>("min_cluster_size", 40);
  max_cluster_size_  = this->declare_parameter<int>("max_cluster_size", 50000);
  marker_lifetime_   = this->declare_parameter<double>("marker_lifetime", 0.2);
  min_box_xy_        = this->declare_parameter<double>("min_box_xy", 0.20);
  max_markers_delete_batch_ = this->declare_parameter<int>("max_markers_delete_batch", 512);

  pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);
  pub_clusters_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cluster_topic_, 10);
  sub_cloud_   = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      // 네임스페이스 문제 수정: std::bind(..., _1) -> std::bind(..., std::placeholders::_1)
      std::bind(&ObjectDetector::cloudCallback, this, _1));

  RCLCPP_INFO(this->get_logger(),
      "[OBB-EXTRACT] in='%s' -> out='%s' (tol=%.3f, min=%d, max=%d)",
      input_topic_.c_str(), marker_topic_.c_str(),
      cluster_tolerance_, min_cluster_size_, max_cluster_size_);
}

ObjectDetector::~ObjectDetector() = default;

void ObjectDetector::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  // === Convert & clean NaNs ===
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*msg, *cloud);
  std::vector<int> idx;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, idx);
  if (cloud->empty()) {
    // clear previous markers if any
    if (last_marker_count_ > 0) {
      visualization_msgs::msg::MarkerArray arr;
      const auto& hdr = msg->header;
      const std::size_t to_del = std::min<std::size_t>(last_marker_count_, max_markers_delete_batch_);
      for (std::size_t i = 0; i < to_del; ++i) arr.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(i)));
      pub_markers_->publish(arr);
      last_marker_count_ = 0;
    }
    return;
  }

  // === Euclidean clustering ===
  auto tree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusters_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
  clusters_rgb->reserve(cloud->size());

  visualization_msgs::msg::MarkerArray marr;
  marr.markers.reserve(cluster_indices.size());

  int id = 0;
  // object_detector.cpp의 cloudCallback 함수 내부 for 루프

  for (const auto& indices : cluster_indices) {
    if (indices.indices.empty()) continue;

    // build cluster cloud & convert to cv::Point2f
    std::vector<cv::Point2f> cluster_points_cv;
    cluster_points_cv.reserve(indices.indices.size());
    float z_min = std::numeric_limits<float>::infinity();
    float z_max = -std::numeric_limits<float>::infinity();

    for (int i : indices.indices) {
      const auto& p = cloud->points[i];
      cluster_points_cv.emplace_back(p.x, p.y);
      if (p.z < z_min) z_min = p.z;
      if (p.z > z_max) z_max = p.z;
    }
    
    // 이 클러스터의 RGB 포인트 클라우드 생성 (시각화용)
    for (const auto& p_idx : indices.indices) {
        const auto& p = cloud->points[p_idx];
        pcl::PointXYZRGB pr;
        pr.x = p.x; pr.y = p.y; pr.z = p.z;
        pr.r = 255; pr.g = 0; pr.b = 0; // 빨간색
        clusters_rgb->points.push_back(pr);
    }
    
    // L-Shape Fitting 실행
    cv::RotatedRect box = lsfitter_->FitBox(cluster_points_cv);
    
    if (box.size.width <= 0 || box.size.height <= 0) continue;

    const float size_z = std::max(0.01f, z_max - z_min);
    if (box.size.width < min_box_xy_ && box.size.height < min_box_xy_) {
      continue;
    }
    const float z_center = 0.5f * (z_min + z_max);

    // === marker 생성 (수정된 부분) ===
    visualization_msgs::msg::Marker mk;
    mk.header = msg->header;
    mk.ns = "obb";
    mk.id = id++;
    mk.type = visualization_msgs::msg::Marker::CUBE;
    mk.action = visualization_msgs::msg::Marker::ADD;

    // cv::RotatedRect 결과(box)를 marker pose에 할당
    mk.pose.position.x = box.center.x;
    mk.pose.position.y = box.center.y;
    mk.pose.position.z = z_center;

    tf2::Quaternion q;
    q.setRPY(0, 0, box.angle * M_PI / 180.0); // OpenCV 각도(degree)를 ROS quaternion으로 변환
    mk.pose.orientation = tf2::toMsg(q);

    // cv::RotatedRect 결과(box)를 marker scale에 할당
    mk.scale.x = box.size.width;
    mk.scale.y = box.size.height;
    mk.scale.z = size_z;

    mk.color.r = 1.0;
    mk.color.g = 1.0;
    mk.color.b = 1.0;
    mk.color.a = 0.4;

    mk.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);

    marr.markers.push_back(mk);
  }

  // delete stale markers
  if (id < static_cast<int>(last_marker_count_)) {
    const auto& hdr = msg->header;
    for (int k = id; k < static_cast<int>(last_marker_count_); ++k) {
      marr.markers.push_back(makeDeleteMarker(hdr, k));
    }
  }
  last_marker_count_ = static_cast<std::size_t>(id);

  // publish
  {
    std::lock_guard<std::mutex> lk(pub_mutex_);
    pub_markers_->publish(marr);
  }

  if (!clusters_rgb->points.empty()) {
    clusters_rgb->width  = static_cast<uint32_t>(clusters_rgb->points.size());
    clusters_rgb->height = 1;
    clusters_rgb->is_dense = true;

    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*clusters_rgb, out);
    out.header = msg->header;  // 입력과 같은 frame_id / stamp
    pub_clusters_->publish(out);
  }
}

} // namespace navi_detection
