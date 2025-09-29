#pragma once

#include <vector>
#include <opencv2/core.hpp>

// 논문의 세 가지 기준(Criterion)을 열거형으로 정의
enum class Criterion
{
    AREA,
    NEAREST,
    VARIANCE
};

class LShapedFIT
{
public:
    // 생성자
    LShapedFIT();

    // 메인 함수: 포인트들을 입력받아 회전된 사각형(RotatedRect)을 반환
    cv::RotatedRect FitBox(const std::vector<cv::Point2f>& points);

private:
    // 각 기준(criterion)에 대한 비용(cost)을 계산하는 함수들
    double calc_variances_criterion(const cv::Mat& c1, const cv::Mat& c2);

    // 벡터의 분산(variance)을 계산하는 헬퍼 함수
    double calc_var(const std::vector<double>& v);

    // 사각형의 네 꼭짓점을 계산하는 헬퍼 함수
    void calc_cross_point(double a0, double b0, double c0, double a1, double b1, double c1, double& x, double& y);
    cv::RotatedRect calc_rect_contour();

    // 멤버 변수
    Criterion criterion_;
    double dtheta_deg_for_search_;

    // 사각형의 네 변을 나타내는 파라미터 (ax + by = c)
    std::vector<double> a_, b_, c_;
    std::vector<cv::Point2f> vertex_pts_;
};