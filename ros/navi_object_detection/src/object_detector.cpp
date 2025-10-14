#include "navi_object_detection/object_detector.hpp"

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
#include <omp.h>

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
  marker_topic_ = this->declare_parameter<std::string>("marker_topic", "/object_detection/obb_detection");
  marker_topic_2 = this->declare_parameter<std::string>("marker_topic_2", "/object_detection/obb_detection_2");
  cluster_topic_ = this->declare_parameter<std::string>("cluster_topic", "/cluster/points");
  cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.6);
  min_cluster_size_  = this->declare_parameter<int>("min_cluster_size", 40);
  max_cluster_size_  = this->declare_parameter<int>("max_cluster_size", 50000);
  marker_lifetime_   = this->declare_parameter<double>("marker_lifetime", 0.2);
  min_box_xy_        = this->declare_parameter<double>("min_box_xy", 0.20);
  max_markers_delete_batch_ = this->declare_parameter<int>("max_markers_delete_batch", 512);

  pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);
  pub_markers_2 = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_2, 10);
  pub_clusters_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cluster_topic_, 10);
  sub_cloud_   = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
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
    if (last_marker_count_ > 0) {
      visualization_msgs::msg::MarkerArray arr;
      const auto& hdr = msg->header;
      const std::size_t to_del = std::min<std::size_t>(last_marker_count_, max_markers_delete_batch_);
      for (std::size_t i = 0; i < to_del; ++i) arr.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(i)));
      pub_markers_->publish(arr);
      pub_markers_2->publish(arr);
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

  // 결과물을 스레드 안전하게 수집하기 위한 컨테이너와 뮤텍스
  visualization_msgs::msg::MarkerArray marr;
  visualization_msgs::msg::MarkerArray marr_2;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusters_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
  std::mutex result_mutex;

  // --- OpenMP를 이용한 병렬 처리 for 루프 ---
  // range-based for 대신 인덱스 기반 for 루프로 변경
  #pragma omp parallel for
  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    const auto& indices = cluster_indices[i];
    if (indices.indices.empty()) continue; // C++17에서는 [[likely]] 사용 가능

    // --- 스레드별 로컬 변수 ---
    // LShapedFIT 객체는 멤버 변수를 가지므로 스레드별로 독립적인 인스턴스를 생성해야 함
    LShapedFIT local_lsfitter; 
    
    // 이 스레드에서 처리할 결과를 담을 로컬 컨테이너
    visualization_msgs::msg::Marker local_mk, local_mk_2;
    pcl::PointCloud<pcl::PointXYZRGB> local_clusters_rgb;
    bool box1_valid = false, box2_valid = false;
    // --- ---

    std::vector<cv::Point2f> cluster_points_cv;
    cluster_points_cv.reserve(indices.indices.size());
    float z_min = std::numeric_limits<float>::infinity();
    float z_max = -std::numeric_limits<float>::infinity();

    for (int idx : indices.indices) {
      const auto& p = cloud->points[idx];
      cluster_points_cv.emplace_back(p.x, p.y);
      if (p.z < z_min) z_min = p.z;
      if (p.z > z_max) z_max = p.z;

      pcl::PointXYZRGB pr;
      pr.x = p.x; pr.y = p.y; pr.z = p.z;
      pr.r = 255; pr.g = 0; pr.b = 0;
      local_clusters_rgb.points.push_back(pr);
    }
    
    cv::RotatedRect box = local_lsfitter.FitBox(cluster_points_cv);
    cv::RotatedRect box_2 = local_lsfitter.FitBox_2(cluster_points_cv);
    
    const float size_z = std::max(0.01f, z_max - z_min);
    const float z_center = 0.5f * (z_min + z_max);

    if (box.size.width > 0 && box.size.height > 0 && !(box.size.width < min_box_xy_ && box.size.height < min_box_xy_)) {
      local_mk.header = msg->header;
      local_mk.ns = "obb_inlier";
      local_mk.id = static_cast<int>(i); // ID를 루프 인덱스로 설정
      local_mk.type = visualization_msgs::msg::Marker::CUBE;
      local_mk.action = visualization_msgs::msg::Marker::ADD;
      local_mk.pose.position.x = box.center.x;
      local_mk.pose.position.y = box.center.y;
      local_mk.pose.position.z = z_center;
      tf2::Quaternion q;
      q.setRPY(0, 0, box.angle * M_PI / 180.0);
      local_mk.pose.orientation = tf2::toMsg(q);
      local_mk.scale.x = box.size.width;
      local_mk.scale.y = box.size.height;
      local_mk.scale.z = size_z;
      local_mk.color.r = 1.0; local_mk.color.g = 0.0; local_mk.color.b = 0.0; local_mk.color.a = 0.4; // 빨간색
      local_mk.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
      box1_valid = true;
    }

    if (box_2.size.width > 0 && box_2.size.height > 0 && !(box_2.size.width < min_box_xy_ && box_2.size.height < min_box_xy_)) {
      local_mk_2.header = msg->header;
      local_mk_2.ns = "obb_variance";
      local_mk_2.id = static_cast<int>(i); // ID를 루프 인덱스로 설정
      local_mk_2.type = visualization_msgs::msg::Marker::CUBE;
      local_mk_2.action = visualization_msgs::msg::Marker::ADD;
      local_mk_2.pose.position.x = box_2.center.x;
      local_mk_2.pose.position.y = box_2.center.y;
      local_mk_2.pose.position.z = z_center;
      tf2::Quaternion q;
      q.setRPY(0, 0, box_2.angle * M_PI / 180.0);
      local_mk_2.pose.orientation = tf2::toMsg(q);
      local_mk_2.scale.x = box_2.size.width;
      local_mk_2.scale.y = box_2.size.height;
      local_mk_2.scale.z = size_z;
      local_mk_2.color.r = 0.0; local_mk_2.color.g = 0.0; local_mk_2.color.b = 1.0; local_mk_2.color.a = 0.4; // 파란색
      local_mk_2.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
      box2_valid = true;
    }

    // --- 모든 계산이 끝난 후, 한 번만 lock을 걸어 공유 데이터에 결과 추가 ---
    {
      std::lock_guard<std::mutex> lock(result_mutex);
      if(box1_valid) marr.markers.push_back(local_mk);
      if(box2_valid) marr_2.markers.push_back(local_mk_2);
      clusters_rgb->points.insert(clusters_rgb->points.end(), local_clusters_rgb.points.begin(), local_clusters_rgb.points.end());
    }
  }

  // delete stale markers
  size_t current_marker_count = std::max(marr.markers.size(), marr_2.markers.size());
  if (current_marker_count < last_marker_count_) {
    const auto& hdr = msg->header;
    for (size_t k = current_marker_count; k < last_marker_count_; ++k) {
      marr.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(k)));
      marr_2.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(k)));
    }
  }
  last_marker_count_ = current_marker_count;

  // publish
  pub_markers_->publish(marr);
  pub_markers_2->publish(marr_2);

  if (!clusters_rgb->points.empty()) {
    clusters_rgb->width  = static_cast<uint32_t>(clusters_rgb->points.size());
    clusters_rgb->height = 1;
    clusters_rgb->is_dense = true;
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*clusters_rgb, out);
    out.header = msg->header;
    pub_clusters_->publish(out);
  }
}

} // namespace navi_detection
