#include "navi_wingbody_detection/object_detector.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <limits>
#include <chrono>

// 스레드 과구독 방지
#include <omp.h>
#include <Eigen/Core>

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
  // === OpenMP / Eigen 스레드 고정 (과구독 방지) ===
  omp_set_dynamic(0);
  omp_set_nested(0);
  omp_set_num_threads(3);     // Jetson Orin: 3~4, Nano: 2 권장
  Eigen::setNbThreads(1);

  // === Fitter ===
  lsfitter_ = std::make_unique<LShapedFIT>();

  // === 파라미터 ===
  input_topic_  = this->declare_parameter<std::string>("input_topic",  "/patchworkpp/nonground");
  marker_topic_ = this->declare_parameter<std::string>("marker_topic", "/object_detection/obb_detection");
  marker_topic_2= this->declare_parameter<std::string>("marker_topic_2","/object_detection/obb_detection_2");
  cluster_topic_= this->declare_parameter<std::string>("cluster_topic","/cluster/points");

  cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.7);
  min_cluster_size_  = this->declare_parameter<int>("min_cluster_size", 40);
  max_cluster_size_  = this->declare_parameter<int>("max_cluster_size", 20000);
  marker_lifetime_   = this->declare_parameter<double>("marker_lifetime", 0.8); // 깜빡임 완화
  min_box_xy_        = this->declare_parameter<double>("min_box_xy", 0.20);
  max_markers_delete_batch_ = this->declare_parameter<int>("max_markers_delete_batch", 512);
  voxel_leaf_        = this->declare_parameter<double>("voxel_leaf", 0.07); // 0.0이면 비활성

  viz_reliable_      = this->declare_parameter<bool>("viz_reliable", true);
  viz_depth_         = this->declare_parameter<int>("viz_depth", 2);
  cluster_reliable_  = this->declare_parameter<bool>("cluster_reliable", false);
  cluster_depth_     = this->declare_parameter<int>("cluster_depth", 1);

  // === QoS ===
  auto make_qos = [&](bool reliable, int depth) {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(std::max(1, depth))); // Volatile (intra-process 호환)
    if (reliable) return qos.reliable();
    else          return qos.best_effort();
  };

  // Subscriber: 최신 1개만 받도록 (밀림 방지)
  auto sub_qos = rclcpp::SensorDataQoS().keep_last(1).best_effort();
  sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, sub_qos,
      std::bind(&ObjectDetector::cloudCallback, this, std::placeholders::_1));

  // Publishers
  pub_markers_  = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_,  make_qos(viz_reliable_, viz_depth_));
  pub_markers_2 = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_2, make_qos(viz_reliable_, viz_depth_));
  pub_clusters_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cluster_topic_,       make_qos(cluster_reliable_, cluster_depth_));

  // 워커 시작
  worker_ = std::thread(&ObjectDetector::workerLoop, this);

  RCLCPP_INFO(this->get_logger(),
    "[OBB-EXTRACT] in='%s' -> out='%s' (tol=%.3f, min=%d, max=%d) | Markers QoS=%s/KeepLast(%d) | voxel_leaf=%.2f",
    input_topic_.c_str(), marker_topic_.c_str(),
    cluster_tolerance_, min_cluster_size_, max_cluster_size_,
    viz_reliable_ ? "Reliable" : "BestEffort", viz_depth_, voxel_leaf_);
}

ObjectDetector::~ObjectDetector() {
  stop_.store(true);
  latest_cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void ObjectDetector::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  // 최신 프레임로 교체 후 notify (이전 프레임은 폐기)
  {
    std::lock_guard<std::mutex> lk(latest_mtx_);
    latest_msg_ = msg;
  }
  latest_cv_.notify_one();
}

void ObjectDetector::workerLoop()
{
  while (rclcpp::ok() && !stop_.load()) {
    std::shared_ptr<const sensor_msgs::msg::PointCloud2> msg;
    {
      std::unique_lock<std::mutex> lk(latest_mtx_);
      latest_cv_.wait(lk, [&]{ return stop_.load() || (latest_msg_ != nullptr); });
      if (stop_.load()) break;
      msg.swap(latest_msg_); // 최신 한 장만 가져오고 비움
    }
    if (msg) {
      processPointCloud(std::const_pointer_cast<sensor_msgs::msg::PointCloud2>(msg));
    }
  }
}

void ObjectDetector::processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg)
{
  using Clock = std::chrono::steady_clock;
  auto nowms = []{ return Clock::now(); };
  auto ms = [](auto a, auto b){ return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count(); };

  auto t0 = nowms();

  // === Convert & clean NaNs ===
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*msg, *cloud);
  std::vector<int> idx_rm;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, idx_rm);

  auto t1 = nowms();

  if (cloud->empty()) {
    if (last_marker_count_ > 0) {
      visualization_msgs::msg::MarkerArray arr;
      const auto& hdr = msg->header;
      const std::size_t to_del = std::min<std::size_t>(last_marker_count_, (std::size_t)max_markers_delete_batch_);
      for (std::size_t i = 0; i < to_del; ++i)
        arr.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(i)));
      pub_markers_->publish(arr);
      pub_markers_2->publish(arr);
      last_marker_count_ = 0;
    }
    return;
  }

  // === VoxelGrid (최악 케이스 안정화) ===
  if (voxel_leaf_ > 0.0) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
    vg.setInputCloud(cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds(new pcl::PointCloud<pcl::PointXYZ>());
    vg.filter(*ds);
    cloud.swap(ds);
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

  auto t2 = nowms();

  // === 결과 컨테이너 ===
  visualization_msgs::msg::MarkerArray marr;
  visualization_msgs::msg::MarkerArray marr_2;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusters_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
  std::mutex result_mutex;

  // === OpenMP 병렬 처리 (스레드 수는 ctor에서 제한됨) ===
  #pragma omp parallel for
  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    const auto& indices = cluster_indices[i];
    if (indices.indices.empty()) continue;

    LShapedFIT local_lsfitter; // 스레드 로컬

    visualization_msgs::msg::Marker local_mk, local_mk_2;
    pcl::PointCloud<pcl::PointXYZRGB> local_clusters_rgb;
    bool box1_valid = false, box2_valid = false;

    std::vector<cv::Point2f> cluster_points_cv;
    cluster_points_cv.reserve(indices.indices.size());
    float z_min = std::numeric_limits<float>::infinity();
    float z_max = -std::numeric_limits<float>::infinity();

    for (int id : indices.indices) {
      const auto& p = cloud->points[id];
      cluster_points_cv.emplace_back(p.x, p.y);
      if (p.z < z_min) z_min = p.z;
      if (p.z > z_max) z_max = p.z;

      pcl::PointXYZRGB pr;
      pr.x = p.x; pr.y = p.y; pr.z = p.z;
      pr.r = 255; pr.g = 0; pr.b = 0;
      local_clusters_rgb.points.push_back(pr);
    }

    cv::RotatedRect box   = local_lsfitter.FitBox(cluster_points_cv);
    cv::RotatedRect box_2 = local_lsfitter.FitBox_2(cluster_points_cv);

    const float size_z   = std::max(0.01f, z_max - z_min);
    const float z_center = 0.5f * (z_min + z_max);

    if (box.size.width > 0 && box.size.height > 0 && !(box.size.width < min_box_xy_ && box.size.height < min_box_xy_)) {
      local_mk.header = msg->header;
      local_mk.ns = "obb_inlier";
      local_mk.id = static_cast<int>(i);
      local_mk.type = visualization_msgs::msg::Marker::CUBE;
      local_mk.action = visualization_msgs::msg::Marker::ADD;
      local_mk.pose.position.x = box.center.x;
      local_mk.pose.position.y = box.center.y;
      local_mk.pose.position.z = z_center;

      tf2::Quaternion q; q.setRPY(0, 0, to_rad(box.angle));
      local_mk.pose.orientation = tf2::toMsg(q);

      local_mk.scale.x = box.size.width;
      local_mk.scale.y = box.size.height;
      local_mk.scale.z = size_z;

      local_mk.color.r = 1.0; local_mk.color.g = 0.0; local_mk.color.b = 0.0; local_mk.color.a = 0.4;
      local_mk.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
      box1_valid = true;
    }

    if (box_2.size.width > 0 && box_2.size.height > 0 && !(box_2.size.width < min_box_xy_ && box_2.size.height < min_box_xy_)) {
      local_mk_2.header = msg->header;
      local_mk_2.ns = "obb_variance";
      local_mk_2.id = static_cast<int>(i);
      local_mk_2.type = visualization_msgs::msg::Marker::CUBE;
      local_mk_2.action = visualization_msgs::msg::Marker::ADD;
      local_mk_2.pose.position.x = box_2.center.x;
      local_mk_2.pose.position.y = box_2.center.y;
      local_mk_2.pose.position.z = z_center;

      tf2::Quaternion q; q.setRPY(0, 0, to_rad(box_2.angle));
      local_mk_2.pose.orientation = tf2::toMsg(q);

      local_mk_2.scale.x = box_2.size.width;
      local_mk_2.scale.y = box_2.size.height;
      local_mk_2.scale.z = size_z;

      local_mk_2.color.r = 0.0; local_mk_2.color.g = 0.0; local_mk_2.color.b = 1.0; local_mk_2.color.a = 0.4;
      local_mk_2.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
      box2_valid = true;
    }

    // 병합
    {
      std::lock_guard<std::mutex> lock(result_mutex);
      if (box1_valid) marr.markers.push_back(local_mk);
      if (box2_valid) marr_2.markers.push_back(local_mk_2);
      clusters_rgb->points.insert(clusters_rgb->points.end(),
                                  local_clusters_rgb.points.begin(),
                                  local_clusters_rgb.points.end());
    }
  }

  auto t3 = nowms();

  // === 오래된 마커 삭제 처리 ===
  size_t current_marker_count = std::max(marr.markers.size(), marr_2.markers.size());
  if (current_marker_count < last_marker_count_) {
    const auto& hdr = msg->header;
    for (size_t k = current_marker_count; k < last_marker_count_; ++k) {
      marr.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(k)));
      marr_2.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(k)));
    }
  }
  last_marker_count_ = current_marker_count;

  // === Publish ===
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

  auto t4 = nowms();

  // 간단 타이밍 로그로 변동성 확인 (필요 시 log level 조정)
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
    "ms: io=%ld, cluster=%ld, fit=%ld, pub=%ld | pts=%u | clusters=%zu",
    static_cast<long>(ms(t0,t1)),
    static_cast<long>(ms(t1,t2)),
    static_cast<long>(ms(t2,t3)),
    static_cast<long>(ms(t3,t4)),
    msg->width * msg->height,
    cluster_indices.size());
}

} // namespace navi_detection
