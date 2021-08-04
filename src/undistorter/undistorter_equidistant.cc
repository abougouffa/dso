#include "undistorter/undistorter_equidistant.h"

#include <glog/logging.h>

namespace dso {
UndistortEquidistant::UndistortEquidistant(const char* file_config,
                                           const bool noprefix) {
  LOG(INFO) << "Creating Equidistant undistorter";

  if (noprefix) {
    ReadFromFile(file_config, 8);
  } else {
    ReadFromFile(file_config, 8, "EquiDistant ");
  }
}
UndistortEquidistant::~UndistortEquidistant() {}

void UndistortEquidistant::DistortCoordinates(float* const in_x,
                                              float* const in_y,
                                              float* const out_x,
                                              float* const out_y,
                                              const int n) const {
  CHECK_NOTNULL(in_x);
  CHECK_NOTNULL(in_y);
  CHECK_NOTNULL(out_x);
  CHECK_NOTNULL(out_y);

  // EQUI
  const float fx = argv_[0];
  const float fy = argv_[1];
  const float cx = argv_[2];
  const float cy = argv_[3];
  const float k1 = argv_[4];
  const float k2 = argv_[5];
  const float k3 = argv_[6];
  const float k4 = argv_[7];

  const float ofx = K_(0, 0);
  const float ofy = K_(1, 1);
  const float ocx = K_(0, 2);
  const float ocy = K_(1, 2);

  for (int i = 0; i < n; ++i) {
    float x = in_x[i];
    float y = in_y[i];

    // EQUI
    const float ix = (x - ocx) / ofx;
    const float iy = (y - ocy) / ofy;
    const float r = sqrtf(ix * ix + iy * iy);
    const float theta = atanf(r);
    const float theta2 = theta * theta;
    const float theta4 = theta2 * theta2;
    const float theta6 = theta4 * theta2;
    const float theta8 = theta4 * theta4;
    const float thetad =
        theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
    const float scaling = (r > 1e-8f) ? thetad / r : 1.f;
    const float ox = fx * ix * scaling + cx;
    const float oy = fy * iy * scaling + cy;

    out_x[i] = ox;
    out_y[i] = oy;
  }
}
}  // dso