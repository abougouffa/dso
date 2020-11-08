#pragma once

#include "undistorter/undistorter.h"

#include <Eigen/Core>

namespace dso {
class UndistortRadTan : public Undistorter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  UndistortRadTan(const char* file_config, bool noprefix);
  ~UndistortRadTan();
  void DistortCoordinates(float* const in_x, float* const in_y,
                          float* const out_x, float* const out_y,
                          const int n) const;
};

}  // dso