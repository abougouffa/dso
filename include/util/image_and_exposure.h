#pragma once

#include <iostream>

#include <Eigen/Core>
#include "sophus/se3.hpp"

namespace dso {

class ImageAndExposure {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImageAndExposure(int w_, int h_, double timestamp_ = 0)
      : w(w_), h(h_), timestamp(timestamp_) {
    image = new float[w * h];
    exposure_time = 1.f;
  }
  ~ImageAndExposure() { delete[] image; }

  void CopyMetaTo(ImageAndExposure& other) {
    other.exposure_time = exposure_time;
  }

  ImageAndExposure* GetDeepCopy() {
    ImageAndExposure* img = new ImageAndExposure(w, h, timestamp);
    img->exposure_time = exposure_time;
    memcpy(img->image, image, w * h * sizeof(float));
    return img;
  }

 public:
  float* image;  // irradiance. between 0 and 256
  int w, h;      // width and height;
  double timestamp, init_scale;
  float exposure_time;  // exposure time in ms.
};
}
