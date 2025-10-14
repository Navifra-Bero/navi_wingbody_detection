#include "navi_object_detection/object_tracker.hpp"

#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <functional>
#include <string>

using std::placeholders::_1;


namespace navi_detection {

ObjectTracker::ObjectTracker(const rclcpp::NodeOptions& options)
: rclcpp::Node("object_tracker_node", options) {
    input_topic_ = this->declare_parameter<std::string>("input_topic", "/object_detection/obb_detection");
    output_topic_ = this->declare_parameter<std::string>("output_topic", "/object_detection/obb_tracking");
    max_age_ = this->declare_parameter<int>("max_age", 3);
    min_hits_ = this->declare_parameter<int>("min_hits", 3);
    association_threshold_ = this->declare_parameter<double>("association_threshold", 2.0);

    mot_tracker_ = std::make_unique<Sort>(max_age_, min_hits_, association_threshold_);

    sub_detections_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
        input_topic_, 10, std::bind(&ObjectTracker::markersCallback, this, _1));
    
    pub_tracks_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(output_topic_, 10);

    RCLCPP_INFO(this->get_logger(), "ObjectTracker node has been initialized.");
}

ObjectTracker::~ObjectTracker() = default;

void ObjectTracker::markersCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
    if (!msg || msg->markers.empty()) {
        mot_tracker_->update({});
        visualization_msgs::msg::MarkerArray empty_markers;
        pub_tracks_->publish(empty_markers);
        return;
    }
    
    std::vector<Detection3D> detections;
    for (const auto& marker : msg->markers) {
        if (marker.action == visualization_msgs::msg::Marker::DELETE) continue;
        
        tf2::Quaternion q;
        tf2::fromMsg(marker.pose.orientation, q);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        
        detections.push_back({
            (float)marker.pose.position.x, (float)marker.pose.position.y,
            (float)marker.scale.x, (float)marker.scale.y,
            (float)yaw,
            (float)marker.pose.position.z, (float)marker.scale.z
        });
    }

    auto tracked_objects = mot_tracker_->update(detections);

    visualization_msgs::msg::MarkerArray tracked_markers;
    const auto& header = msg->markers[0].header; 
    
    for (const auto& tracked : tracked_objects) {
        const Detection3D& box = tracked.first;
        int id = tracked.second;

        visualization_msgs::msg::Marker cube_mk;
        cube_mk.header = header;
        cube_mk.ns = "tracked_box";
        cube_mk.id = id;
        cube_mk.type = visualization_msgs::msg::Marker::CUBE;
        cube_mk.action = visualization_msgs::msg::Marker::ADD;
        cube_mk.pose.position.x = box.x;
        cube_mk.pose.position.y = box.y;
        cube_mk.pose.position.z = box.z_center;
        tf2::Quaternion q_cube;
        q_cube.setRPY(0, 0, box.angle);
        cube_mk.pose.orientation = tf2::toMsg(q_cube);
        cube_mk.scale.x = box.w;
        cube_mk.scale.y = box.h;
        cube_mk.scale.z = box.height;
        cube_mk.color.a = 0.6;
        cube_mk.color.r = (id * 50 % 255) / 255.0;
        cube_mk.color.g = (id * 90 % 255) / 255.0;
        cube_mk.color.b = (id * 120 % 255) / 255.0;
        cube_mk.lifetime = rclcpp::Duration::from_seconds(0.2);
        tracked_markers.markers.push_back(cube_mk);

        visualization_msgs::msg::Marker text_mk;
        text_mk.header = header;
        text_mk.ns = "tracked_id";
        text_mk.id = id;
        text_mk.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_mk.action = visualization_msgs::msg::Marker::ADD;
        text_mk.pose.position.x = box.x;
        text_mk.pose.position.y = box.y;
        text_mk.pose.position.z = box.z_center + box.height / 2.0 + 0.5;
        text_mk.text = std::to_string(id);
        text_mk.scale.z = 1.0;
        text_mk.color.a = 1.0;
        text_mk.color.r = 1.0; text_mk.color.g = 1.0; text_mk.color.b = 1.0;
        text_mk.lifetime = rclcpp::Duration::from_seconds(0.2);
        tracked_markers.markers.push_back(text_mk);

        visualization_msgs::msg::Marker arrow_mk;
        arrow_mk.header = header;
        arrow_mk.ns = "tracked_orientation";
        arrow_mk.id = id;
        arrow_mk.type = visualization_msgs::msg::Marker::ARROW;
        arrow_mk.action = visualization_msgs::msg::Marker::ADD;
        arrow_mk.pose = cube_mk.pose;

        arrow_mk.scale.x = box.w;
        arrow_mk.scale.y = 0.15;
        arrow_mk.scale.z = 0.15;
        arrow_mk.color.a = 1.0;
        arrow_mk.color.r = 1.0;
        arrow_mk.color.g = 1.0;
        arrow_mk.color.b = 0.0;

        arrow_mk.lifetime = rclcpp::Duration::from_seconds(0.2);
        tracked_markers.markers.push_back(arrow_mk);
    }
    pub_tracks_->publish(tracked_markers);
}

} // namespace navi_detection