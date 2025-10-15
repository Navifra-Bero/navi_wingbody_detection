#include "navi_wingbody_detection/object_detector.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <limits>
#include <chrono>
#include <cmath>
#include <mutex>

#include <omp.h>
#include <Eigen/Core>

namespace navi_detection {

// =============== helpers ===============
static inline visualization_msgs::msg::Marker makeDeleteMarker(const std_msgs::msg::Header& hdr, int id) {
  visualization_msgs::msg::Marker m;
  m.header = hdr; m.ns = "obb"; m.id = id;
  m.action = visualization_msgs::msg::Marker::DELETE;
  return m;
}

static inline visualization_msgs::msg::Marker makeDeleteMarkerNs(const std_msgs::msg::Header& hdr,
                                                                 const std::string& ns, int id) {
  visualization_msgs::msg::Marker m;
  m.header = hdr; m.ns = ns; m.id = id;
  m.action = visualization_msgs::msg::Marker::DELETE;
  return m;
}

static inline double yaw_long_from_rect(const cv::RotatedRect& r) {
  double yaw_deg = static_cast<double>(r.angle);
  if (r.size.height > r.size.width) yaw_deg += 90.0;
  // [-180,180)
  while (yaw_deg >= 180.0) yaw_deg -= 360.0;
  while (yaw_deg <  -180.0) yaw_deg += 360.0;
  return yaw_deg * M_PI / 180.0;
}

static inline double to_rad(double deg) { return deg * M_PI / 180.0; }

static inline visualization_msgs::msg::Marker makeArrow(
    const std_msgs::msg::Header& hdr, int id, const std::string& ns,
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    double shaft_diam, double head_diam, double head_len,
    float r,float g,float b,float a=0.95f) {
  visualization_msgs::msg::Marker mk;
  mk.header = hdr;
  mk.ns = ns;
  mk.id = id;
  mk.type = visualization_msgs::msg::Marker::ARROW;
  mk.action = visualization_msgs::msg::Marker::ADD;
  mk.pose.orientation.w = 1.0; 
  mk.scale.x = shaft_diam;     
  mk.scale.y = head_diam;      
  mk.scale.z = head_len;     
  mk.color.r=r; mk.color.g=g; mk.color.b=b; mk.color.a=a;
  mk.points.resize(2);
  mk.points[0].x = p0.x(); mk.points[0].y = p0.y(); mk.points[0].z = p0.z();
  mk.points[1].x = p1.x(); mk.points[1].y = p1.y(); mk.points[1].z = p1.z();
  return mk;
}

static std::vector<Eigen::Vector2d> s_prev_centers_1;
static std::vector<Eigen::Vector2d> s_prev_forward_1;
static std::vector<Eigen::Vector2d> s_prev_centers_2;
static std::vector<Eigen::Vector2d> s_prev_forward_2;
static std::vector<Eigen::Vector2d> s_prev_centers_3;
static std::vector<Eigen::Vector2d> s_prev_forward_3;
static std::vector<Eigen::Vector2d> s_prev_centers_4;
static std::vector<Eigen::Vector2d> s_prev_forward_4;

static std::vector<Eigen::Vector2d> s_curr_centers_1;
static std::vector<Eigen::Vector2d> s_curr_forward_1;
static std::vector<Eigen::Vector2d> s_curr_centers_2;
static std::vector<Eigen::Vector2d> s_curr_forward_2;
static std::vector<Eigen::Vector2d> s_curr_centers_3;
static std::vector<Eigen::Vector2d> s_curr_forward_3;
static std::vector<Eigen::Vector2d> s_curr_centers_4;
static std::vector<Eigen::Vector2d> s_curr_forward_4;

static constexpr double kAssocMaxDist = 0.8;

static inline int nn_associate(const std::vector<Eigen::Vector2d>& prev_centers,
                               const Eigen::Vector2d& c_now) {
  if (prev_centers.empty()) return -1;
  int best = -1;
  double best_d2 = kAssocMaxDist * kAssocMaxDist;
  for (int j = 0; j < (int)prev_centers.size(); ++j) {
    const auto& c = prev_centers[j];
    double dx = c.x() - c_now.x();
    double dy = c.y() - c_now.y();
    double d2 = dx*dx + dy*dy;
    if (d2 <= best_d2) { best_d2 = d2; best = j; }
  }
  return best;
}

static inline Eigen::Vector2d fwd_from_yaw(double yaw) {
  return Eigen::Vector2d(std::cos(yaw), std::sin(yaw));
}
// =======================================

ObjectDetector::ObjectDetector(const rclcpp::NodeOptions& options)
: rclcpp::Node("obb_extractor_node", options)
{
  omp_set_dynamic(0);
  omp_set_nested(0);
  omp_set_num_threads(3);
  Eigen::setNbThreads(1);

  input_topic_  = this->declare_parameter<std::string>("input_topic",  "/patchworkpp/nonground");
  marker_topic_ = this->declare_parameter<std::string>("marker_topic", "/object_detection/obb_detection");
  marker_topic_2= this->declare_parameter<std::string>("marker_topic_2","/object_detection/obb_detection_2");
  marker_topic_3= this->declare_parameter<std::string>("marker_topic_3","/object_detection/obb_detection_3");
  marker_topic_4= this->declare_parameter<std::string>("marker_topic_4","/object_detection/obb_detection_4");

  cluster_topic_= this->declare_parameter<std::string>("cluster_topic","/cluster/points");
  poses_topic_   = this->declare_parameter<std::string>("poses_topic",   "/obb_poses");
  poses_topic_2_ = this->declare_parameter<std::string>("poses_topic_2", "/obb_poses_2");
  poses_topic_3_ = this->declare_parameter<std::string>("poses_topic_3", "/obb_poses_3");
  poses_topic_4_ = this->declare_parameter<std::string>("poses_topic_4", "/obb_poses_4");

  cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.7);
  min_cluster_size_  = this->declare_parameter<int>("min_cluster_size", 40);
  max_cluster_size_  = this->declare_parameter<int>("max_cluster_size", 20000);
  marker_lifetime_   = this->declare_parameter<double>("marker_lifetime", 0.8);
  min_box_xy_        = this->declare_parameter<double>("min_box_xy", 0.20);
  max_markers_delete_batch_ = this->declare_parameter<int>("max_markers_delete_batch", 512);
  voxel_leaf_        = this->declare_parameter<double>("voxel_leaf", 0.07);

  viz_reliable_      = this->declare_parameter<bool>("viz_reliable", true);
  viz_depth_         = this->declare_parameter<int>("viz_depth", 2);
  cluster_reliable_  = this->declare_parameter<bool>("cluster_reliable", false);
  cluster_depth_     = this->declare_parameter<int>("cluster_depth", 1);

  lshape_dtheta_deg_ = this->declare_parameter<double>("lshape.dtheta_deg", 1.0);
  lshape_inlier_threshold_ = this->declare_parameter<double>("lshape.inlier_threshold", 0.1);
  lshape_min_dist_nearest_ = this->declare_parameter<double>("lshape.min_dist_nearest", 0.01);

  auto make_qos = [&](bool reliable, int depth) {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(std::max(1, depth)));
    return reliable ? qos.reliable() : qos.best_effort();
  };

  auto sub_qos = rclcpp::SensorDataQoS().keep_last(1).best_effort();
  sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, sub_qos,
      std::bind(&ObjectDetector::cloudCallback, this, std::placeholders::_1));

  pub_markers_  = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_,  make_qos(viz_reliable_, viz_depth_));
  pub_markers_2 = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_2, make_qos(viz_reliable_, viz_depth_));
  pub_markers_3 = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_3, make_qos(viz_reliable_, viz_depth_));
  pub_markers_4 = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_4, make_qos(viz_reliable_, viz_depth_));
  pub_clusters_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cluster_topic_,       make_qos(cluster_reliable_, cluster_depth_));

  pub_poses_    = this->create_publisher<geometry_msgs::msg::PoseArray>(poses_topic_,   make_qos(viz_reliable_, viz_depth_));
  pub_poses_2_  = this->create_publisher<geometry_msgs::msg::PoseArray>(poses_topic_2_, make_qos(viz_reliable_, viz_depth_));
  pub_poses_3_  = this->create_publisher<geometry_msgs::msg::PoseArray>(poses_topic_3_, make_qos(viz_reliable_, viz_depth_));
  pub_poses_4_  = this->create_publisher<geometry_msgs::msg::PoseArray>(poses_topic_4_, make_qos(viz_reliable_, viz_depth_));

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
  { std::lock_guard<std::mutex> lk(latest_mtx_); latest_msg_ = msg; }
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
      msg.swap(latest_msg_);
    }
    if (msg) processPointCloud(std::const_pointer_cast<sensor_msgs::msg::PointCloud2>(msg));
  }
}

void ObjectDetector::processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg)
{
  using Clock = std::chrono::steady_clock;
  auto nowms = []{ return Clock::now(); };
  auto ms = [](auto a, auto b){ return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count(); };

  auto t0 = nowms();

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
      for (std::size_t i = 0; i < to_del; ++i) {
        arr.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(i)));
        arr.markers.push_back(makeDeleteMarkerNs(hdr, "obb_dir",  static_cast<int>(i)));
        arr.markers.push_back(makeDeleteMarkerNs(hdr, "obb2_dir", static_cast<int>(i)));
      }
      pub_markers_->publish(arr);
      pub_markers_2->publish(arr);
      pub_markers_3->publish(arr);
      pub_markers_4->publish(arr);
      last_marker_count_ = 0;
    }
    geometry_msgs::msg::PoseArray empty1, empty2, empty3, empty4;
    empty1.header = msg->header;
    empty2.header = msg->header;
    empty3.header = msg->header;
    empty4.header = msg->header;
    pub_poses_->publish(empty1);
    pub_poses_2_->publish(empty2);
    pub_poses_3_->publish(empty3);
    pub_poses_4_->publish(empty4);

    s_prev_centers_1.clear(); s_prev_forward_1.clear();
    s_prev_centers_2.clear(); s_prev_forward_2.clear();
    s_prev_centers_3.clear(); s_prev_forward_3.clear();
    s_prev_centers_4.clear(); s_prev_forward_4.clear();
    return;
  }

  if (voxel_leaf_ > 0.0) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
    vg.setInputCloud(cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds(new pcl::PointCloud<pcl::PointXYZ>());
    vg.filter(*ds);
    cloud.swap(ds);
  }

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

  visualization_msgs::msg::MarkerArray marr;
  visualization_msgs::msg::MarkerArray marr_2;
  visualization_msgs::msg::MarkerArray marr_3;
  visualization_msgs::msg::MarkerArray marr_4;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusters_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
  std::mutex result_mutex;

  geometry_msgs::msg::PoseArray poses_arr_1;
  geometry_msgs::msg::PoseArray poses_arr_2;
  geometry_msgs::msg::PoseArray poses_arr_3;
  geometry_msgs::msg::PoseArray poses_arr_4;
  poses_arr_1.header = msg->header;
  poses_arr_2.header = msg->header;
  poses_arr_3.header = msg->header;
  poses_arr_4.header = msg->header;

  s_curr_centers_1.clear(); s_curr_forward_1.clear();
  s_curr_centers_2.clear(); s_curr_forward_2.clear();
  s_curr_centers_3.clear(); s_curr_forward_3.clear();
  s_curr_centers_4.clear(); s_curr_forward_4.clear();

  const double kArrowLenExtra = 0.40;
  const double kArrowShaft    = 0.05;
  const double kArrowHeadD    = 0.10;
  const double kArrowHeadL    = 0.12;

  #pragma omp parallel for
  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    const auto& indices = cluster_indices[i];
    if (indices.indices.empty()) continue;

    LShapedFIT local_lsfitter(lshape_dtheta_deg_, lshape_inlier_threshold_, lshape_min_dist_nearest_);

    visualization_msgs::msg::Marker local_mk, local_mk_2, local_mk_3, local_mk_4;
    pcl::PointCloud<pcl::PointXYZRGB> local_clusters_rgb;
    std::vector<geometry_msgs::msg::Pose> local_poses_1;
    std::vector<geometry_msgs::msg::Pose> local_poses_2;
    std::vector<geometry_msgs::msg::Pose> local_poses_3;
    std::vector<geometry_msgs::msg::Pose> local_poses_4;

    bool box1_valid = false, box2_valid = false, box3_valid = false, box4_valid = false;

    std::vector<cv::Point2f> cluster_points_cv;
    cluster_points_cv.reserve(indices.indices.size());
    float z_min = std::numeric_limits<float>::infinity();
    float z_max = -std::numeric_limits<float>::infinity();

    for (int id : indices.indices) {
      const auto& p = cloud->points[id];
      cluster_points_cv.emplace_back(p.x, p.y);
      if (p.z < z_min) z_min = p.z;
      if (p.z > z_max) z_max = p.z;

      pcl::PointXYZRGB pr; pr.x=p.x; pr.y=p.y; pr.z=p.z; pr.r=255; pr.g=0; pr.b=0;
      local_clusters_rgb.points.push_back(pr);
    }

    cv::RotatedRect box   = local_lsfitter.FitBox_area(cluster_points_cv);
    cv::RotatedRect box_2 = local_lsfitter.FitBox_nearest(cluster_points_cv);
    cv::RotatedRect box_3 = local_lsfitter.FitBox_inlier(cluster_points_cv);
    cv::RotatedRect box_4 = local_lsfitter.FitBox_variances(cluster_points_cv);

    const float size_z   = std::max(0.01f, z_max - z_min);
    const float z_center = 0.5f * (z_min + z_max);

    // ---------- 방법1(size criterion) ----------
    if (box.size.width > 0 && box.size.height > 0 && !(box.size.width < min_box_xy_ && box.size.height < min_box_xy_)) {
      local_mk.header = msg->header;
      local_mk.ns = "obb_area";
      local_mk.id = static_cast<int>(i);
      local_mk.type = visualization_msgs::msg::Marker::CUBE;
      local_mk.action = visualization_msgs::msg::Marker::ADD;
      local_mk.pose.position.x = box.center.x;
      local_mk.pose.position.y = box.center.y;
      local_mk.pose.position.z = z_center;
      { tf2::Quaternion q; q.setRPY(0, 0, to_rad(box.angle)); local_mk.pose.orientation = tf2::toMsg(q); }
      local_mk.scale.x = box.size.width; local_mk.scale.y = box.size.height; local_mk.scale.z = size_z;
      local_mk.color.r = 1.0; local_mk.color.g = 0.0; local_mk.color.b = 0.0; local_mk.color.a = 0.2;
      local_mk.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
      box1_valid = true;

      geometry_msgs::msg::Pose p1; p1.position = local_mk.pose.position; p1.orientation = local_mk.pose.orientation;
      local_poses_1.push_back(p1);

      double yaw_long = yaw_long_from_rect(box);
      Eigen::Vector2d fwd_now = fwd_from_yaw(yaw_long);

      Eigen::Vector2d c_now(box.center.x, box.center.y);
      int j = nn_associate(s_prev_centers_1, c_now);

      if (j >= 0 && j < (int)s_prev_forward_1.size()) {
        Eigen::Vector2d fwd_fix = s_prev_forward_1[j];
        if (fwd_fix.dot(fwd_now) < 0.0) {
          yaw_long += M_PI;
          if (yaw_long >= M_PI) yaw_long -= 2.0*M_PI;
          fwd_now = fwd_from_yaw(yaw_long);
        }
      } else {

      }

      const double long_len = std::max(box.size.width, box.size.height);
      Eigen::Vector3d c3(box.center.x, box.center.y, z_center);
      Eigen::Vector3d edge = c3 + Eigen::Vector3d(fwd_now.x()*(0.5*long_len), fwd_now.y()*(0.5*long_len), 0.0);
      Eigen::Vector3d tip  = edge + Eigen::Vector3d(fwd_now.x()*kArrowLenExtra, fwd_now.y()*kArrowLenExtra, 0.0);
      auto dir = makeArrow(msg->header, static_cast<int>(i), "obb_dir", edge, tip,
                           kArrowShaft, kArrowHeadD, kArrowHeadL, 1.0f, 0.25f, 0.25f, 0.95f);
      dir.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);

      {
        std::lock_guard<std::mutex> lock(result_mutex);
        s_curr_centers_1.push_back(c_now);
        s_curr_forward_1.push_back(fwd_now);

        marr.markers.push_back(local_mk);
        marr.markers.push_back(dir);

        poses_arr_1.poses.push_back(p1);

        clusters_rgb->points.insert(clusters_rgb->points.end(),
                                    local_clusters_rgb.points.begin(),
                                    local_clusters_rgb.points.end());
      }
    }

    // ---------- 방법2(nearest criterion) ----------
    if (box_2.size.width > 0 && box_2.size.height > 0 && !(box_2.size.width < min_box_xy_ && box_2.size.height < min_box_xy_)) {
      local_mk_2.header = msg->header;
      local_mk_2.ns = "obb_nearest";
      local_mk_2.id = static_cast<int>(i);
      local_mk_2.type = visualization_msgs::msg::Marker::CUBE;
      local_mk_2.action = visualization_msgs::msg::Marker::ADD;
      local_mk_2.pose.position.x = box_2.center.x;
      local_mk_2.pose.position.y = box_2.center.y;
      local_mk_2.pose.position.z = z_center;
      { tf2::Quaternion q; q.setRPY(0, 0, to_rad(box_2.angle)); local_mk_2.pose.orientation = tf2::toMsg(q); }
      local_mk_2.scale.x = box_2.size.width; local_mk_2.scale.y = box_2.size.height; local_mk_2.scale.z = size_z;
      local_mk_2.color.r = 0.0; local_mk_2.color.g = 1.0; local_mk_2.color.b = 0.0; local_mk_2.color.a = 0.2;
      local_mk_2.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
      box2_valid = true;

      geometry_msgs::msg::Pose p2; p2.position = local_mk_2.pose.position; p2.orientation = local_mk_2.pose.orientation;
      local_poses_2.push_back(p2);

      double yaw_long2 = yaw_long_from_rect(box_2);
      Eigen::Vector2d fwd_now2 = fwd_from_yaw(yaw_long2);
      Eigen::Vector2d c_now2(box_2.center.x, box_2.center.y);
      int j2 = nn_associate(s_prev_centers_2, c_now2);
      if (j2 >= 0 && j2 < (int)s_prev_forward_2.size()) {
        Eigen::Vector2d fwd_fix2 = s_prev_forward_2[j2];
        if (fwd_fix2.dot(fwd_now2) < 0.0) {
          yaw_long2 += M_PI;
          if (yaw_long2 >= M_PI) yaw_long2 -= 2.0*M_PI;
          fwd_now2 = fwd_from_yaw(yaw_long2);
        }
      }

      const double long_len2 = std::max(box_2.size.width, box_2.size.height);
      Eigen::Vector3d c32(box_2.center.x, box_2.center.y, z_center);
      Eigen::Vector3d edge2 = c32 + Eigen::Vector3d(fwd_now2.x()*(0.5*long_len2), fwd_now2.y()*(0.5*long_len2), 0.0);
      Eigen::Vector3d tip2  = edge2 + Eigen::Vector3d(fwd_now2.x()*kArrowLenExtra, fwd_now2.y()*kArrowLenExtra, 0.0);
      auto dir2 = makeArrow(msg->header, static_cast<int>(i), "obb2_dir", edge2, tip2,
                            kArrowShaft, kArrowHeadD, kArrowHeadL, 0.25f, 1.0f, 0.25f, 0.95f);
      dir2.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);

      {
        std::lock_guard<std::mutex> lock(result_mutex);
        s_curr_centers_2.push_back(c_now2);
        s_curr_forward_2.push_back(fwd_now2);

        marr_2.markers.push_back(local_mk_2);
        marr_2.markers.push_back(dir2);

        poses_arr_2.poses.push_back(p2);

        clusters_rgb->points.insert(clusters_rgb->points.end(),
                                    local_clusters_rgb.points.begin(),
                                    local_clusters_rgb.points.end());
      }
    }

    // ---------- 방법3(inlier criterion) ----------
    if (box_3.size.width > 0 && box_3.size.height > 0 && !(box_3.size.width < min_box_xy_ && box_3.size.height < min_box_xy_)) {
      local_mk_3.header = msg->header;
      local_mk_3.ns = "obb_inlier";
      local_mk_3.id = static_cast<int>(i);
      local_mk_3.type = visualization_msgs::msg::Marker::CUBE;
      local_mk_3.action = visualization_msgs::msg::Marker::ADD;
      local_mk_3.pose.position.x = box_3.center.x;
      local_mk_3.pose.position.y = box_3.center.y;
      local_mk_3.pose.position.z = z_center;
      { tf2::Quaternion q; q.setRPY(0, 0, to_rad(box_3.angle)); local_mk_3.pose.orientation = tf2::toMsg(q); }
      local_mk_3.scale.x = box_3.size.width; local_mk_3.scale.y = box_3.size.height; local_mk_3.scale.z = size_z;
      local_mk_3.color.r = 0.0; local_mk_3.color.g = 0.0; local_mk_3.color.b = 1.0; local_mk_3.color.a = 0.2;
      local_mk_3.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
      box3_valid = true;

      geometry_msgs::msg::Pose p3; p3.position = local_mk_3.pose.position; p3.orientation = local_mk_3.pose.orientation;
      local_poses_3.push_back(p3);

      double yaw_long3 = yaw_long_from_rect(box_3);
      Eigen::Vector2d fwd_now3 = fwd_from_yaw(yaw_long3);
      Eigen::Vector2d c_now3(box_3.center.x, box_3.center.y);
      int j3 = nn_associate(s_prev_centers_3, c_now3);
      if (j3 >= 0 && j3 < (int)s_prev_forward_3.size()) {
        Eigen::Vector2d fwd_fix3 = s_prev_forward_3[j3];
        if (fwd_fix3.dot(fwd_now3) < 0.0) {
          yaw_long3 += M_PI;
          if (yaw_long3 >= M_PI) yaw_long3 -= 2.0*M_PI;
          fwd_now3 = fwd_from_yaw(yaw_long3);
        }
      }

      const double long_len3 = std::max(box_3.size.width, box_3.size.height);
      Eigen::Vector3d c33(box_3.center.x, box_3.center.y, z_center);
      Eigen::Vector3d edge3 = c33 + Eigen::Vector3d(fwd_now3.x()*(0.5*long_len3), fwd_now3.y()*(0.5*long_len3), 0.0);
      Eigen::Vector3d tip3  = edge3 + Eigen::Vector3d(fwd_now3.x()*kArrowLenExtra, fwd_now3.y()*kArrowLenExtra, 0.0);
      auto dir3 = makeArrow(msg->header, static_cast<int>(i), "obb3_dir", edge3, tip3,
                            kArrowShaft, kArrowHeadD, kArrowHeadL, 0.25f, 0.25f, 1.0f, 0.95f);
      dir3.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
      {
        std::lock_guard<std::mutex> lock(result_mutex);
        s_curr_centers_3.push_back(c_now3);
        s_curr_forward_3.push_back(fwd_now3);

        marr_3.markers.push_back(local_mk_3);
        marr_3.markers.push_back(dir3);

        poses_arr_3.poses.push_back(p3);

        clusters_rgb->points.insert(clusters_rgb->points.end(),
                                    local_clusters_rgb.points.begin(),
                                    local_clusters_rgb.points.end());
      }
    }

    // ---------- 방법4(variance criterion) ----------
    if (box_4.size.width > 0 && box_4.size.height > 0 && !(box_4.size.width < min_box_xy_ && box_4.size.height < min_box_xy_)) {
      visualization_msgs::msg::Marker local_mk_4;

      local_mk_4.header = msg->header;
      local_mk_4.ns = "obb_variance";
      local_mk_4.id = static_cast<int>(i);
      local_mk_4.type = visualization_msgs::msg::Marker::CUBE;
      local_mk_4.action = visualization_msgs::msg::Marker::ADD;
      local_mk_4.pose.position.x = box_4.center.x;
      local_mk_4.pose.position.y = box_4.center.y;
      local_mk_4.pose.position.z = z_center;
      { tf2::Quaternion q; q.setRPY(0, 0, to_rad(box_4.angle)); local_mk_4.pose.orientation = tf2::toMsg(q); }
      local_mk_4.scale.x = box_4.size.width; local_mk_4.scale.y = box_4.size.height; local_mk_4.scale.z = size_z;
      local_mk_4.color.r = 1.0; local_mk_4.color.g = 1.0; local_mk_4.color.b = 0.0; local_mk_4.color.a = 0.2;
      local_mk_4.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
      bool box4_valid = true;

      geometry_msgs::msg::Pose p4; 
      p4.position = local_mk_4.pose.position; 
      p4.orientation = local_mk_4.pose.orientation;
      std::vector<geometry_msgs::msg::Pose> local_poses_4;
      local_poses_4.push_back(p4);

      double yaw_long4 = yaw_long_from_rect(box_4);
      Eigen::Vector2d fwd_now4 = fwd_from_yaw(yaw_long4);
      Eigen::Vector2d c_now4(box_4.center.x, box_4.center.y);
      int j4 = nn_associate(s_prev_centers_4, c_now4);
      if (j4 >= 0 && j4 < (int)s_prev_forward_4.size()) {
        Eigen::Vector2d fwd_fix4 = s_prev_forward_4[j4];
        if (fwd_fix4.dot(fwd_now4) < 0.0) {
          yaw_long4 += M_PI;
          if (yaw_long4 >= M_PI) yaw_long4 -= 2.0*M_PI;
          fwd_now4 = fwd_from_yaw(yaw_long4);
        }
      }

      const double long_len4 = std::max(box_4.size.width, box_4.size.height);
      Eigen::Vector3d c34(box_4.center.x, box_4.center.y, z_center);
      Eigen::Vector3d edge4 = c34 + Eigen::Vector3d(fwd_now4.x()*(0.5*long_len4), fwd_now4.y()*(0.5*long_len4), 0.0);
      Eigen::Vector3d tip4  = edge4 + Eigen::Vector3d(fwd_now4.x()*kArrowLenExtra, fwd_now4.y()*kArrowLenExtra, 0.0);
      auto dir4 = makeArrow(msg->header, static_cast<int>(i), "obb4_dir", edge4, tip4,
                            kArrowShaft, kArrowHeadD, kArrowHeadL, 1.0f, 1.0f, 0.25f, 0.95f);
      dir4.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);

      {
        std::lock_guard<std::mutex> lock(result_mutex);
        s_curr_centers_4.push_back(c_now4);
        s_curr_forward_4.push_back(fwd_now4);

        marr_4.markers.push_back(local_mk_4);
        marr_4.markers.push_back(dir4);

        poses_arr_4.poses.push_back(p4);

        clusters_rgb->points.insert(clusters_rgb->points.end(),
                                    local_clusters_rgb.points.begin(),
                                    local_clusters_rgb.points.end());
      }
    }
  }

  auto t3 = nowms();

  auto count_cubes = [](const visualization_msgs::msg::MarkerArray& arr)->size_t {
    size_t n=0; for (auto& m: arr.markers) if (m.type == visualization_msgs::msg::Marker::CUBE) ++n; return n;
  };
  size_t cube_count1 = count_cubes(marr);
  size_t cube_count2 = count_cubes(marr_2);
  size_t cube_count3 = count_cubes(marr_3);
  size_t cube_count4 = count_cubes(marr_4);
  size_t current_cube_count_1 = std::max(cube_count1, cube_count2);
  size_t current_cube_count_2 = std::max(cube_count3, cube_count4);
  size_t current_cube_count = std::max(current_cube_count_1, current_cube_count_2);

  if (current_cube_count < last_marker_count_) {
    const auto& hdr = msg->header;
    for (size_t k = current_cube_count; k < last_marker_count_; ++k) {
      marr.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(k)));
      marr_2.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(k)));
      marr_3.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(k)));
      marr_4.markers.push_back(makeDeleteMarker(hdr, static_cast<int>(k)));

      marr.markers.push_back(makeDeleteMarkerNs(hdr, "obb_dir",  static_cast<int>(k)));
      marr_2.markers.push_back(makeDeleteMarkerNs(hdr, "obb2_dir", static_cast<int>(k)));
      marr_3.markers.push_back(makeDeleteMarkerNs(hdr, "obb3_dir", static_cast<int>(k)));
      marr_4.markers.push_back(makeDeleteMarkerNs(hdr, "obb4_dir", static_cast<int>(k)));
    }
  }
  last_marker_count_ = current_cube_count;

  // Publish
  pub_markers_->publish(marr);
  pub_markers_2->publish(marr_2);
  pub_markers_3->publish(marr_3);
  pub_markers_4->publish(marr_4);

  pub_poses_->publish(poses_arr_1);
  pub_poses_2_->publish(poses_arr_2);
  pub_poses_3_->publish(poses_arr_3);
  pub_poses_4_->publish(poses_arr_4);

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

  s_prev_centers_1.swap(s_curr_centers_1);
  s_prev_forward_1.swap(s_curr_forward_1);
  s_prev_centers_2.swap(s_curr_centers_2);
  s_prev_forward_2.swap(s_curr_forward_2);
  s_prev_centers_3.swap(s_curr_centers_3);
  s_prev_forward_3.swap(s_curr_forward_3);
  s_prev_centers_4.swap(s_curr_centers_4);
  s_prev_forward_4.swap(s_curr_forward_4);

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
