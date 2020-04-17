#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "sophus/se3.hpp"

namespace dso {

class Converter {
 public:
  static cv::Mat ToCvMat(const Eigen::Matrix<double, 4, 4>& mat);
  static cv::Mat ToCvMat(const Eigen::Matrix3d& mat);
  static cv::Mat ToCvMat(const Eigen::Matrix<double, 3, 1>& mat);
  static cv::Mat ToCvSE3(const Eigen::Matrix<double, 3, 3>& R,
                         const Eigen::Matrix<double, 3, 1>& t);

  static Eigen::Matrix<double, 3, 1> ToVector3d(const cv::Mat& vector);
  static Eigen::Matrix<double, 3, 1> ToVector3d(const cv::Point3f& point);
  static Eigen::Matrix<double, 3, 3> ToMatrix3d(const cv::Mat& mat);

  static std::vector<float> ToQuaternion(const cv::Mat& mat);
  static Eigen::Matrix<double, 4, 4> ToMatrix4d(const cv::Mat& mat);

  static Sophus::SE3d ToSophusSE3(const cv::Mat& mat);
  static Sophus::SE3d ToSophusSE3(
      const std::vector<double>& pose);  // timestamp tx ty tz qx qy qz qw
  static Sophus::SE3d ToSophusSE3(const double tx, const double ty,
                                  const double tz, const double qx,
                                  const double qy, const double qz,
                                  const double qw);
};

}  // dso
