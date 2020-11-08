#pragma once

#include "undistorter/undistorter.h"

#include <Eigen/Core>

namespace dso {
class UndistortPinhole : public Undistorter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  UndistortPinhole(const char* file_config, bool noprefix);
  ~UndistortPinhole();
  void DistortCoordinates(float* const in_x, float* const in_y,
                          float* const out_x, float* const out_y,
                          const int n) const;
};

}  // namespace dso