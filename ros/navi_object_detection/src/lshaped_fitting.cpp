#include "navi_object_detection/lshaped_fitting.hpp" // 위에서 만든 헤더파일

#include <numeric>
#include <iostream>
#include <cmath>
#include <opencv2/imgproc.hpp> // cv::minAreaRect 사용을 위해 추가

LShapedFIT::LShapedFIT() {
    dtheta_deg_for_search_ = 1.0;
    criterion_ = Criterion::VARIANCE;
}

double LShapedFIT::calc_var(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double variance = sq_sum / v.size() - mean * mean;
    return variance;
}

double LShapedFIT::calc_variances_criterion(const cv::Mat& c1, const cv::Mat& c2) {
    double min_c1, max_c1, min_c2, max_c2;
    cv::minMaxLoc(c1, &min_c1, &max_c1);
    cv::minMaxLoc(c2, &min_c2, &max_c2);
    std::vector<double> d1, d2, e1_dist, e2_dist;
    for (int i = 0; i < c1.rows; ++i) {
        d1.push_back(std::min(c1.at<double>(i) - min_c1, max_c1 - c1.at<double>(i)));
        d2.push_back(std::min(c2.at<double>(i) - min_c2, max_c2 - c2.at<double>(i)));
        if (d1.back() < d2.back()) e1_dist.push_back(d1.back());
        else e2_dist.push_back(d2.back());
    }
    double v1 = e1_dist.empty() ? 0.0 : calc_var(e1_dist);
    double v2 = e2_dist.empty() ? 0.0 : calc_var(e2_dist);
    return -(v1 + v2);
}

void LShapedFIT::calc_cross_point(double a0, double b0, double c0, double a1, double b1, double c1, double& x, double& y) {
    double det = a0 * b1 - a1 * b0;
    if (std::abs(det) < 1e-9) { x = 0; y = 0; return; }
    x = (c0 * b1 - c1 * b0) / det;
    y = (a0 * c1 - a1 * c0) / det;
}

cv::RotatedRect LShapedFIT::calc_rect_contour() {
    vertex_pts_.clear();
    vertex_pts_.resize(4);
    double temp_x, temp_y;
    calc_cross_point(a_[0], b_[0], c_[0], a_[1], b_[1], c_[1], temp_x, temp_y);
    vertex_pts_[0] = cv::Point2f(static_cast<float>(temp_x), static_cast<float>(temp_y));
    calc_cross_point(a_[0], b_[0], c_[0], a_[3], b_[3], c_[3], temp_x, temp_y);
    vertex_pts_[1] = cv::Point2f(static_cast<float>(temp_x), static_cast<float>(temp_y));
    calc_cross_point(a_[2], b_[2], c_[2], a_[3], b_[3], c_[3], temp_x, temp_y);
    vertex_pts_[2] = cv::Point2f(static_cast<float>(temp_x), static_cast<float>(temp_y));
    calc_cross_point(a_[2], b_[2], c_[2], a_[1], b_[1], c_[1], temp_x, temp_y);
    vertex_pts_[3] = cv::Point2f(static_cast<float>(temp_x), static_cast<float>(temp_y));
    return cv::minAreaRect(vertex_pts_);
}

cv::RotatedRect LShapedFIT::FitBox(const std::vector<cv::Point2f>& points) {
    if (points.size() < 3) return cv::RotatedRect();
    cv::Mat matrix_pts(points.size(), 2, CV_64FC1);
    for (size_t i = 0; i < points.size(); ++i) {
        matrix_pts.at<double>(i, 0) = points[i].x;
        matrix_pts.at<double>(i, 1) = points[i].y;
    }
    double dtheta = dtheta_deg_for_search_ * M_PI / 180.0;
    double best_theta = 0.0;
    double max_score = -std::numeric_limits<double>::max();
    int loop_number = static_cast<int>(std::ceil((M_PI / 2.0) / dtheta));
    for (int k = 0; k < loop_number; ++k) {
        double theta = k * dtheta;
        cv::Mat e1 = (cv::Mat_<double>(1, 2) << std::cos(theta), std::sin(theta));
        cv::Mat e2 = (cv::Mat_<double>(1, 2) << -std::sin(theta), std::cos(theta));
        cv::Mat c1 = matrix_pts * e1.t();
        cv::Mat c2 = matrix_pts * e2.t();
        double score = calc_variances_criterion(c1, c2);
        if (score > max_score) {
            max_score = score;
            best_theta = theta;
        }
    }
    double sin_s = sin(best_theta);
    double cos_s = cos(best_theta);
    a_ = {cos_s, -sin_s, cos_s, -sin_s};
    b_ = {sin_s, cos_s, sin_s, cos_s};
    cv::Mat e1_s = (cv::Mat_<double>(1, 2) << a_[0], b_[0]);
    cv::Mat e2_s = (cv::Mat_<double>(1, 2) << a_[1], b_[1]);
    cv::Mat c1_s = matrix_pts * e1_s.t();
    cv::Mat c2_s = matrix_pts * e2_s.t();
    double min_c1, max_c1, min_c2, max_c2;
    cv::minMaxLoc(c1_s, &min_c1, &max_c1);
    cv::minMaxLoc(c2_s, &min_c2, &max_c2);
    c_ = {min_c1, min_c2, max_c1, max_c2};
    return calc_rect_contour();
}