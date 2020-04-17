#pragma once

#include <glog/logging.h>

#include "util/image_and_exposure.h"
#include "util/minimal_image.h"
#include "util/num_type.h"
#include "util/settings.h"

namespace dso {

class PhotometricUndistorter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  PhotometricUndistorter(const std::string& file_photometric_calibration,
                         const std::string& image_noise,
                         const std::string& image_vignette, const int w,
                         const int h);
  ~PhotometricUndistorter();

  float* GetG() { return valid_ ? G_ : nullptr; };

  void UnMapFloatImage(float* const image);

  // removes readout noise, and converts to irradiance.
  // affine normalizes values to 0 <= I < 256.
  // raw irradiance = a*I + b.
  // output_ will be written in [output_].
  template <typename T>
  void ProcessFrame(T* const image_in, const float exposure_time,
                    const float factor = 1) {
    int wh = w_ * h_;
    float* const data = output_->image;
    CHECK_EQ(output_->w, w_);
    CHECK_EQ(output_->h, h_);
    CHECK_NOTNULL(data);

    if (!valid_ || exposure_time <= 0.f ||
        setting_photometricCalibration == 0) {
      // disable full photometric calibration.
      for (int i = 0; i < wh; ++i) {
        data[i] = factor * image_in[i];
      }
      output_->exposure_time = exposure_time;
      output_->timestamp = 0.;
    } else {
      for (int i = 0; i < wh; ++i) {
        data[i] = G_[image_in[i]];
      }

      if (setting_photometricCalibration == 2) {
        for (int i = 0; i < wh; ++i) {
          data[i] *= vignette_map_inv_[i];
        }
      }

      output_->exposure_time = exposure_time;
      output_->timestamp = 0.;
    }

    if (!setting_useExposure) {
      output_->exposure_time = 1.f;
    }
  }

 public:
  ImageAndExposure* output_;

 private:
  float G_[256 * 256];
  int G_depth_;
  float* vignette_map_;
  float* vignette_map_inv_;
  int w_, h_;
  bool valid_;
};
}