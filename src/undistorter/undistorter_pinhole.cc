#include "undistorter/undistorter_pinhole.h"

namespace dso {
UndistortPinhole::UndistortPinhole(const char* file_config, bool noprefix) {
  if (noprefix) {
    ReadFromFile(file_config, 5);
  } else {
    ReadFromFile(file_config, 5, "Pinhole ");
  }
}
UndistortPinhole::~UndistortPinhole() {}

void UndistortPinhole::DistortCoordinates(float* const in_x, float* const in_y,
                                          float* const out_x,
                                          float* const out_y,
                                          const int n) const {
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
    ix = fx * ix + cx;
    iy = fy * iy + cy;
    out_x[i] = ix;
    out_y[i] = iy;
  }
}
}  // dso