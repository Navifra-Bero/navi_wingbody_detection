#pragma once

#include <vector>
#include <opencv2/core.hpp>

enum class Criterion
{
    AREA,
    NEAREST,
    VARIANCE,
    INLIER
};

class LShapedFIT
{
public:
    LShapedFIT();

    cv::RotatedRect FitBox_inlier(const std::vector<cv::Point2f>& points);
    cv::RotatedRect FitBox_variances(const std::vector<cv::Point2f>& points);
    cv::RotatedRect FitBox_area(const std::vector<cv::Point2f>& points);  
    cv::RotatedRect FitBox_nearest(const std::vector<cv::Point2f>& points);
    LShapedFIT(double dtheta_deg, double inlier_thresh, double min_dist_nearest);

private:
    double calc_area_criterion(const cv::Mat& c1, const cv::Mat& c2);
    double calc_nearest_criterion(const cv::Mat& c1, const cv::Mat& c2);
    double calc_inlier_criterion(const cv::Mat& c1, const cv::Mat& c2);
    double calc_variances_criterion(const cv::Mat& c1, const cv::Mat& c2);
    double calc_var(const std::vector<double>& v);
    void calc_cross_point(double a0, double b0, double c0, double a1, double b1, double c1, double& x, double& y);
    cv::RotatedRect calc_rect_contour();

    double dtheta_deg_for_search_;
    double inlier_threshold_;
    double min_dist_of_nearest_crit_;

    std::vector<double> a_, b_, c_;
    std::vector<cv::Point2f> vertex_pts_;
};