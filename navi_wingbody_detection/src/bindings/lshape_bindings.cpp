#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <stdexcept>

#include <Eigen/Core>
#include "navi_wingbody_detection/lshaped_fitting.hpp"

namespace py = pybind11;

using MatX2fRow = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;

static inline std::vector<cv::Point2f> numpy_to_cv_points(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
  if (arr.ndim() != 2 || arr.shape(1) != 2)
    throw std::runtime_error("points must be (N,2) float32/float64");
  const int N = static_cast<int>(arr.shape(0));
  auto buf = arr.unchecked<2>();
  std::vector<cv::Point2f> pts;
  pts.reserve(N);
  for (int i=0;i<N;++i) {
    pts.emplace_back(buf(i,0), buf(i,1));
  }
  return pts;
}


// cpp로 작성된 함수를 파이썬에서 사용하기 위한 binding 함수들
PYBIND11_MODULE(lshape_bindings, m) {
  m.doc() = "Pybind11 bindings for L-shaped fitting (Patchwork++ OBB helpers)";

  py::class_<LShapedFIT>(m, "LShaped")
    .def(py::init<float, float, float>(),
         py::arg("dtheta_deg")=1.0f,
         py::arg("inlier_threshold")=0.1f,
         py::arg("min_dist_nearest")=0.01f)
    .def("fit_area", [](LShapedFIT &self, py::array_t<float, py::array::c_style | py::array::forcecast> points){
        auto pts = numpy_to_cv_points(points);
        cv::RotatedRect r = self.FitBox_area(pts);
        return py::make_tuple(r.center.x, r.center.y, r.size.width, r.size.height, r.angle);
      })
    .def("fit_nearest", [](LShapedFIT &self, py::array_t<float, py::array::c_style | py::array::forcecast> points){
        auto pts = numpy_to_cv_points(points);
        cv::RotatedRect r = self.FitBox_nearest(pts);
        return py::make_tuple(r.center.x, r.center.y, r.size.width, r.size.height, r.angle);
      })
    .def("fit_inlier", [](LShapedFIT &self, py::array_t<float, py::array::c_style | py::array::forcecast> points){
        auto pts = numpy_to_cv_points(points);
        cv::RotatedRect r = self.FitBox_inlier(pts);
        return py::make_tuple(r.center.x, r.center.y, r.size.width, r.size.height, r.angle);
      })
    .def("fit_variances", [](LShapedFIT &self, py::array_t<float, py::array::c_style | py::array::forcecast> points){
        auto pts = numpy_to_cv_points(points);
        cv::RotatedRect r = self.FitBox_variances(pts);
        return py::make_tuple(r.center.x, r.center.y, r.size.width, r.size.height, r.angle);
      });
}
