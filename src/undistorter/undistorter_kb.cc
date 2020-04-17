#include "undistorter/undistorter_kb.h"

#include <glog/logging.h>

namespace dso {
UndistortKB::UndistortKB(const char* file_config, bool noprefix) {
  LOG(INFO) << "Creating KannalaBrandt undistorter";

  if (noprefix) {
    ReadFromFile(file_config, 8);
  } else {
    ReadFromFile(file_config, 8, "KannalaBrandt ");
  }
}
UndistortKB::~UndistortKB() {}

void UndistortKB::DistortCoordinates(float* const in_x, float* const in_y,
                                     float* const out_x, float* const out_y,
                                     const int n) const {
  const float fx = argv_[0];
  const float fy = argv_[1];
  const float cx = argv_[2];
  const float cy = argv_[3];
  const float k0 = argv_[4];
  const float k1 = argv_[5];
  const float k2 = argv_[6];
  const float k3 = argv_[7];

  const float ofx = K_(0, 0);
  const float ofy = K_(1, 1);
  const float ocx = K_(0, 2);
  const float ocy = K_(1, 2);

  for (int i = 0; i < n; ++i) {
    const float x = in_x[i];
    const float y = in_y[i];

    // RADTAN
    const float ix = (x - ocx) / ofx;
    const float iy = (y - ocy) / ofy;

    const float Xsq_plus_Ysq = ix * ix + iy * iy;
    const float sqrt_Xsq_Ysq = sqrtf(Xsq_plus_Ysq);
    const float theta = atan2f(sqrt_Xsq_Ysq, 1);
    const float theta2 = theta * theta;
    const float theta3 = theta2 * theta;
    const float theta5 = theta3 * theta2;
    const float theta7 = theta5 * theta2;
    const float theta9 = theta7 * theta2;
    const float r =
        theta + k0 * theta3 + k1 * theta5 + k2 * theta7 + k3 * theta9;

    if (sqrt_Xsq_Ysq < 1e-6f) {
      out_x[i] = fx * ix + cx;
      out_y[i] = fy * iy + cy;
    } else {
      out_x[i] = (r / sqrt_Xsq_Ysq) * fx * ix + cx;
      out_y[i] = (r / sqrt_Xsq_Ysq) * fy * iy + cy;
    }
  }
}
}  // dso