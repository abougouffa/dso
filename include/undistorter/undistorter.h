#pragma once

#include <glog/logging.h>
#include <Eigen/Core>

#include "undistorter/photometric_undistorter.h"
#include "util/global_funcs.h"
#include "util/image_and_exposure.h"
#include "util/minimal_image.h"
#include "util/num_type.h"
#include "util/settings.h"

namespace dso {

class Undistorter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual ~Undistorter();

  virtual void DistortCoordinates(float* const in_x, float* const in_y,
                                  float* const out_x, float* const out_y,
                                  const int n) const = 0;

  const Mat33 GetK() const { return K_; };
  const Eigen::Vector2i GetSize() const { return Eigen::Vector2i(w_, h_); };
  const VecX GetOriginalParameter() const { return argv_; };
  const Eigen::Vector2i GetOriginalSize() {
    return Eigen::Vector2i(w_org_, h_org_);
  };
  bool IsValid() { return valid_; };

  static Undistorter* GetUndistorterForFile(const std::string& file_config,
                                            const std::string& file_gamma,
                                            const std::string& file_vignette);

  void LoadPhotometricCalibration(
      const std::string& file_photometric_calibration,
      const std::string& image_noise, const std::string& image_vignette);

  template <typename T>
  ImageAndExposure* Undistort(const MinimalImage<T>* image_raw,
                              float exposure = 0, double timestamp = 0,
                              float factor = 1) const {
    LOG_IF(FATAL, image_raw->w != w_org_ || image_raw->h != h_org_)
        << "Wrong image size (" << image_raw->w << ", " << image_raw->h
        << ") instead of (" << w_ << ", " << h_ << ")!";

    photometric_undistorter_->ProcessFrame<T>(image_raw->data, exposure,
                                              factor);
    ImageAndExposure* result = new ImageAndExposure(w_, h_, timestamp);
    photometric_undistorter_->output_->CopyMetaTo(*result);

    if (!pass_through_) {
      float* out_data = result->image;
      float* in_data = photometric_undistorter_->output_->image;

      float* noise_map_x = nullptr;
      float* noise_map_y = nullptr;
      if (benchmark_varNoise > 0) {
        const int num_noise =
            (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
        noise_map_x = new float[num_noise];
        noise_map_y = new float[num_noise];
        memset(noise_map_x, 0, sizeof(float) * num_noise);
        memset(noise_map_y, 0, sizeof(float) * num_noise);

        for (int i = 0; i < num_noise; ++i) {
          noise_map_x[i] =
              2.f * benchmark_varNoise *
              (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) -
               0.5f);
          noise_map_y[i] =
              2.f * benchmark_varNoise *
              (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) -
               0.5f);
        }
      }

      for (int idx = w_ * h_ - 1; idx >= 0; --idx) {
        // get interp. values
        float xx = remap_x_[idx];
        float yy = remap_y_[idx];

        if (benchmark_varNoise > 0) {
          const float delta_x = getInterpolatedElement11BiCub(
              noise_map_x,
              4.f + (xx / static_cast<float>(w_org_)) * benchmark_noiseGridsize,
              4.f + (yy / static_cast<float>(h_org_)) * benchmark_noiseGridsize,
              benchmark_noiseGridsize + 8);
          const float delta_y = getInterpolatedElement11BiCub(
              noise_map_y,
              4.f + (xx / static_cast<float>(w_org_)) * benchmark_noiseGridsize,
              4.f + (yy / static_cast<float>(h_org_)) * benchmark_noiseGridsize,
              benchmark_noiseGridsize + 8);
          float x = idx % w_ + delta_x;
          float y = idx / w_ + delta_y;
          if (x < 0.01f) {
            x = 0.01f;
          }
          if (y < 0.01f) {
            y = 0.01f;
          }
          if (x > static_cast<float>(w_) - 1.01f) {
            x = static_cast<float>(w_) - 1.01f;
          }
          if (y > static_cast<float>(h_) - 1.01f) {
            y = static_cast<float>(h_) - 1.01f;
          }

          xx = getInterpolatedElement(remap_x_, x, y, w_);
          yy = getInterpolatedElement(remap_y_, x, y, w_);
        }

        if (xx < 0) {
          out_data[idx] = 0.f;
        } else {
          // get integer and rational parts
          const int xxi = static_cast<int>(xx);
          const int yyi = static_cast<int>(yy);
          xx -= static_cast<float>(xxi);
          yy -= static_cast<float>(yyi);
          const float xxyy = xx * yy;

          // get array base pointer
          const float* src = in_data + xxi + yyi * w_org_;

          // interpolate (bilinear)
          out_data[idx] = xxyy * src[1 + w_org_] + (yy - xxyy) * src[w_org_] +
                          (xx - xxyy) * src[1] + (1 - xx - yy + xxyy) * src[0];
        }
      }

      if (benchmark_varNoise > 0) {
        delete[] noise_map_x;
        delete[] noise_map_y;
      }
    } else {
      memcpy(result->image, photometric_undistorter_->output_->image,
             sizeof(float) * w_ * h_);
    }

    ApplyBlurNoise(result->image);

    return result;
  }

 public:
  PhotometricUndistorter* photometric_undistorter_;

 protected:
  void ApplyBlurNoise(float* const img) const;

  void MakeOptimalKCrop();
  void MakeOptimalKFull();

  void ReadFromFile(const char* file_config, const int argc,
                    const std::string& prefix = "");

 protected:
  int w_, h_, w_org_, h_org_, w_up_, h_up_;
  int upsample_undist_factor_;
  Mat33 K_;
  VecX argv_;
  bool valid_;
  bool pass_through_;

  float* remap_x_;
  float* remap_y_;
};
}
