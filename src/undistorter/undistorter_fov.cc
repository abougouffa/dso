#include "undistorter/undistorter_fov.h"

#include <glog/logging.h>

namespace dso {
UndistortFOV::UndistortFOV(const char* file_config, const bool noprefix) {
  LOG(INFO) << "Creating FOV undistorter";

  if (noprefix) {
    ReadFromFile(file_config, 5);
  } else {
    ReadFromFile(file_config, 5, "FOV ");
  }
}
UndistortFOV::~UndistortFOV() {}

void UndistortFOV::DistortCoordinates(float* const in_x, float* const in_y,
                                      float* const out_x, float* const out_y,
                                      const int n) const {
  const float dist = argv_[4];
  const float d2t = 2.f * tanf(dist / 2.0f);

  // current camera parameters
  const float fx = argv_[0];
  const float fy = argv_[1];
  const float cx = argv_[2];
  const float cy = argv_[3];

  const float ofx = K_(0, 0);
  const float ofy = K_(1, 1);
  const float ocx = K_(0, 2);
  const float ocy = K_(1, 2);

  for (int i = 0; i < n; ++i) {
    const float x = in_x[i];
    const float y = in_y[i];
    float ix = (x - ocx) / ofx;
    float iy = (y - ocy) / ofy;

    const float r = sqrtf(ix * ix + iy * iy);
    const float fac = (r == 0 || dist == 0) ? 1 : atanf(r * d2t) / (dist * r);

    ix = fx * fac * ix + cx;
    iy = fy * fac * iy + cy;

    out_x[i] = ix;
    out_y[i] = iy;
  }
}
}  // dso