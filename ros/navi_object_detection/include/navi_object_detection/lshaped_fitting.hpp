#pragma once

#include <vector>
#include <opencv2/core.hpp>

// 논문의 세 가지 기준(Criterion)을 열거형으로 정의
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
    // 생성자
    LShapedFIT();

    // 메인 함수: 포인트들을 입력받아 회전된 사각형(RotatedRect)을 반환
    cv::RotatedRect FitBox(const std::vector<cv::Point2f>& points);
    cv::RotatedRect FitBox_2(const std::vector<cv::Point2f>& points);

private:
    // 새로운 기준 함수 선언 추가
    double calc_inlier_criterion(const cv::Mat& c1, const cv::Mat& c2);
    double calc_variances_criterion(const cv::Mat& c1, const cv::Mat& c2);
    double calc_var(const std::vector<double>& v);
    void calc_cross_point(double a0, double b0, double c0, double a1, double b1, double c1, double& x, double& y);
    cv::RotatedRect calc_rect_contour();

    // 멤버 변수
    Criterion criterion_;
    double dtheta_deg_for_search_;
    double inlier_threshold_; // 내점 판단을 위한 거리 임계값 추가

    std::vector<double> a_, b_, c_;
    std::vector<cv::Point2f> vertex_pts_;
};