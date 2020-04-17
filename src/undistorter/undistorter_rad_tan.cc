#include "undistorter/undistorter_rad_tan.h"

#include <glog/logging.h>

namespace dso {
UndistortRadTan::UndistortRadTan(const char* file_config, bool noprefix) {
  LOG(INFO) << "Creating RadTan undistorter";

  if (noprefix) {
    ReadFromFile(file_config, 8);
  } else {
    ReadFromFile(file_config, 8, "RadTan ");
  }
}
UndistortRadTan::~UndistortRadTan() {}

void UndistortRadTan::DistortCoordinates(float* const in_x, float* const in_y,
                                         float* const out_x, float* const out_y,
                                         const int n) const {
  // RADTAN
  const float fx = argv_[0];
  const float fy = argv_[1];
  const float cx = argv_[2];
  const float cy = argv_[3];
  const float k1 = argv_[4];
  const float k2 = argv_[5];
  const float r1 = argv_[6];
  const float r2 = argv_[7];

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
    const float mx2_u = ix * ix;
    const float my2_u = iy * iy;
    const float mxy_u = ix * iy;
    const float rho2_u = mx2_u + my2_u;
    const float rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    const float x_dist =
        ix + ix * rad_dist_u + 2.f * r1 * mxy_u + r2 * (rho2_u + 2.f * mx2_u);
    const float y_dist =
        iy + iy * rad_dist_u + 2.f * r2 * mxy_u + r1 * (rho2_u + 2.f * my2_u);
    const float ox = fx * x_dist + cx;
    const float oy = fy * y_dist + cy;

    out_x[i] = ox;
    out_y[i] = oy;
  }
}
}  // dso