#pragma once

#include "undistorter/undistorter.h"

#include <Eigen/Core>

namespace dso {
class UndistortKB : public Undistorter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  UndistortKB(const char* file_config, bool noprefix);
  ~UndistortKB();
  void DistortCoordinates(float* const in_x, float* const in_y,
                          float* const out_x, float* const out_y,
                          const int n) const;
};
}  // dso