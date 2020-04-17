#include "util/converter.h"

#include <glog/logging.h>

namespace dso {

cv::Mat Converter::ToCvMat(const Eigen::Matrix<double, 4, 4>& mat) {
  cv::Mat cv_mat(4, 4, CV_32F);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      cv_mat.at<float>(i, j) = mat(i, j);
    }
  }

  return cv_mat.clone();
}

cv::Mat Converter::ToCvMat(const Eigen::Matrix3d& mat) {
  cv::Mat cvMat(3, 3, CV_32F);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cvMat.at<float>(i, j) = mat(i, j);
    }
  }

  return cvMat.clone();
}

cv::Mat Converter::ToCvMat(const Eigen::Matrix<double, 3, 1>& mat) {
  cv::Mat cvMat(3, 1, CV_32F);
  for (int i = 0; i < 3; ++i) {
    cvMat.at<float>(i) = mat(i);
  }

  return cvMat.clone();
}

cv::Mat Converter::ToCvSE3(const Eigen::Matrix<double, 3, 3>& R,
                           const Eigen::Matrix<double, 3, 1>& t) {
  cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cvMat.at<float>(i, j) = R(i, j);
    }
  }
  for (int i = 0; i < 3; ++i) {
    cvMat.at<float>(i, 3) = t(i);
  }

  return cvMat.clone();
}

Eigen::Matrix<double, 3, 1> Converter::ToVector3d(const cv::Mat& vector) {
  Eigen::Matrix<double, 3, 1> v;
  v << vector.at<float>(0), vector.at<float>(1), vector.at<float>(2);

  return v;
}

Eigen::Matrix<double, 3, 1> Converter::ToVector3d(const cv::Point3f& cvPoint) {
  Eigen::Matrix<double, 3, 1> v;
  v << cvPoint.x, cvPoint.y, cvPoint.z;

  return v;
}

Eigen::Matrix<double, 3, 3> Converter::ToMatrix3d(const cv::Mat& mat) {
  Eigen::Matrix<double, 3, 3> M;

  M << mat.at<float>(0, 0), mat.at<float>(0, 1), mat.at<float>(0, 2),
      mat.at<float>(1, 0), mat.at<float>(1, 1), mat.at<float>(1, 2),
      mat.at<float>(2, 0), mat.at<float>(2, 1), mat.at<float>(2, 2);

  return M;
}

std::vector<float> Converter::ToQuaternion(const cv::Mat& M) {
  Eigen::Matrix<double, 3, 3> eigMat = ToMatrix3d(M);
  Eigen::Quaterniond q(eigMat);

  std::vector<float> v(4);
  v[0] = q.x();
  v[1] = q.y();
  v[2] = q.z();
  v[3] = q.w();

  return v;
}

Eigen::Matrix<double, 4, 4> Converter::ToMatrix4d(const cv::Mat& mat) {
  Eigen::Matrix<double, 4, 4> M;
  M << mat.at<float>(0, 0), mat.at<float>(0, 1), mat.at<float>(0, 2),
      mat.at<float>(0, 3), mat.at<float>(1, 0), mat.at<float>(1, 1),
      mat.at<float>(1, 2), mat.at<float>(1, 3), mat.at<float>(2, 0),
      mat.at<float>(2, 1), mat.at<float>(2, 2), mat.at<float>(2, 3),
      mat.at<float>(3, 0), mat.at<float>(3, 1), mat.at<float>(3, 2),
      mat.at<float>(3, 3);

  return M;
}

Sophus::SE3d Converter::ToSophusSE3(const cv::Mat& mat) {
  const Eigen::Matrix4d mat_eigen = Converter::ToMatrix4d(mat);
  Sophus::SE3d mat_sophus;
  mat_sophus.setRotationMatrix(mat_eigen.topLeftCorner<3, 3>());
  mat_sophus.translation() = mat_eigen.topRightCorner<3, 1>();
  return mat_sophus;
}

Sophus::SE3d Converter::ToSophusSE3(const double tx, const double ty,
                                    const double tz, const double qx,
                                    const double qy, const double qz,
                                    const double qw) {
  Sophus::SE3d se3;
  se3.translation() = Eigen::Vector3d(tx, ty, tz);
  Eigen::Quaterniond q(qw, qx, qy, qz);
  q.normalize();
  se3.setQuaternion(q);
  return se3;
}

Sophus::SE3d Converter::ToSophusSE3(const std::vector<double>& pose) {
  CHECK_EQ(pose.size(), 8);
  return ToSophusSE3(pose[1], pose[2], pose[3], pose[4], pose[5], pose[6],
                     pose[7]);
}

}  // dso
