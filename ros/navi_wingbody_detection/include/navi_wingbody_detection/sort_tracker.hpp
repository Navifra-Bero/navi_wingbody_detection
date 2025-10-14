#pragma once

#include <vector>
#include <utility>
#include <opencv2/video/tracking.hpp>

// 감지/추적된 객체의 상태를 담는 구조체
struct Detection3D {
    float x, y, w, h, angle; 
    float z_center, height;
};

// 칼만 필터로 개별 객체를 추적하는 클래스 (Sort 클래스의 내부 헬퍼)
class KalmanBoxTracker {
public:
    explicit KalmanBoxTracker(const Detection3D& det);
    void predict();
    void update(const Detection3D& det);
    Detection3D get_state();
    int get_id() const;
    int get_time_since_update() const;
    int get_hit_streak() const;

private:
    cv::KalmanFilter kf_;
    int id_;
    int time_since_update_ = 0;
    int hits_ = 0;
    int hit_streak_ = 0;
    int age_ = 0;
    static int count_;
};

// SORT 알고리즘의 메인 클래스
class Sort {
public:
    Sort(int max_age, int min_hits, double association_threshold);
    std::vector<std::pair<Detection3D, int>> update(const std::vector<Detection3D>& detections);

private:
    int max_age_;
    int min_hits_;
    double association_threshold_;
    std::vector<KalmanBoxTracker> trackers_;
};