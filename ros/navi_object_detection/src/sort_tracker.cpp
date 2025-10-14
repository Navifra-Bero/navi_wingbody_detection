#include "navi_object_detection/sort_tracker.hpp"
#include <algorithm>
#include <cmath>

// KalmanBoxTracker의 static 멤버 변수 초기화
int KalmanBoxTracker::count_ = 0;

KalmanBoxTracker::KalmanBoxTracker(const Detection3D& det) {
    kf_ = cv::KalmanFilter(10, 5, 0);
    kf_.transitionMatrix = (cv::Mat_<float>(10, 10) << 
    1,0,0,0,0,1,0,0,0,0, 
    0,1,0,0,0,0,1,0,0,0, 
    0,0,1,0,0,0,0,1,0,0, 
    0,0,0,1,0,0,0,0,1,0, 
    0,0,0,0,1,0,0,0,0,1, 
    0,0,0,0,0,1,0,0,0,0, 
    0,0,0,0,0,0,1,0,0,0, 
    0,0,0,0,0,0,0,1,0,0, 
    0,0,0,0,0,0,0,0,1,0, 
    0,0,0,0,0,0,0,0,0,1);
    kf_.measurementMatrix = cv::Mat::eye(5, 10, CV_32F);
    setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
    kf_.statePost.at<float>(0) = det.x;
    kf_.statePost.at<float>(1) = det.y;
    kf_.statePost.at<float>(2) = det.w;
    kf_.statePost.at<float>(3) = det.h;
    kf_.statePost.at<float>(4) = det.angle;
    id_ = count_++;
}

void KalmanBoxTracker::predict() {
    kf_.predict();
    age_++;
    time_since_update_++;
}

void KalmanBoxTracker::update(const Detection3D& det) {
    cv::Mat measurement = (cv::Mat_<float>(5, 1) << det.x, det.y, det.w, det.h, det.angle);
    kf_.correct(measurement);
    time_since_update_ = 0;
    hits_++;
    hit_streak_++;
}

Detection3D KalmanBoxTracker::get_state() {
    const float* state = (float*)kf_.statePost.data;
    return {state[0], state[1], state[2], state[3], state[4], 0.0f, 0.0f};
}

int KalmanBoxTracker::get_id() const { return id_; }
int KalmanBoxTracker::get_time_since_update() const { return time_since_update_; }
int KalmanBoxTracker::get_hit_streak() const { return hit_streak_; }


Sort::Sort(int max_age, int min_hits, double association_threshold)
    : max_age_(max_age), min_hits_(min_hits), association_threshold_(association_threshold) {}

std::vector<std::pair<Detection3D, int>> Sort::update(const std::vector<Detection3D>& detections) {
    for (auto& tracker : trackers_) {
        tracker.predict();
    }

    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_detections;
    if (!trackers_.empty()) {
        std::vector<bool> det_matched(detections.size(), false);
        for (size_t t = 0; t < trackers_.size(); ++t) {
            Detection3D pred_box = trackers_[t].get_state();
            double min_dist = association_threshold_;
            int best_det_idx = -1;
            for (size_t d = 0; d < detections.size(); ++d) {
                if (det_matched[d]) continue;
                double dist = std::hypot(pred_box.x - detections[d].x, pred_box.y - detections[d].y);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_det_idx = d;
                }
            }
            if (best_det_idx != -1) {
                matches.emplace_back(best_det_idx, t);
                det_matched[best_det_idx] = true;
            }
        }
        for(size_t i = 0; i < det_matched.size(); ++i) if(!det_matched[i]) unmatched_detections.push_back(i);
    } else {
        for(size_t i = 0; i < detections.size(); ++i) unmatched_detections.push_back(i);
    }

    for (const auto& match : matches) {
        trackers_[match.second].update(detections[match.first]);
    }

    for (int idx : unmatched_detections) {
        trackers_.emplace_back(detections[idx]);
    }
    
    std::vector<std::pair<Detection3D, int>> result;
    auto it = trackers_.begin();
    while (it != trackers_.end()) {
        if (it->get_time_since_update() < 1 && it->get_hit_streak() >= min_hits_) {
            Detection3D tracked_box = it->get_state();
            int id = it->get_id();
            for(const auto& match : matches) {
                if(trackers_[match.second].get_id() == id) {
                    tracked_box.z_center = detections[match.first].z_center;
                    tracked_box.height = detections[match.first].height;
                    break;
                }
            }
            result.emplace_back(tracked_box, id);
        }
        if (it->get_time_since_update() > max_age_) {
            it = trackers_.erase(it);
        } else {
            ++it;
        }
    }
    return result;
}