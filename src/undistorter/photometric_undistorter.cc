#include "undistorter/photometric_undistorter.h"

#include <fstream>
#include <iostream>
#include <iterator>

#include "io_wrapper/image_rw.h"

namespace dso {
PhotometricUndistorter::PhotometricUndistorter(
    const std::string& file_photometric_calibration,
    const std::string& image_noise, const std::string& image_vignette,
    const int w, const int h) {
  valid_ = false;
  vignette_map_ = 0;
  vignette_map_inv_ = 0;
  w_ = w;
  h_ = h;
  output_ = new ImageAndExposure(w, h);
  if (file_photometric_calibration == "" || image_vignette == "") {
    LOG(WARNING) << "NO PHOTOMETRIC Calibration!";
  }

  // read G.
  std::ifstream f(file_photometric_calibration.c_str());
  LOG(INFO) << "Reading Photometric Calibration from file "
            << file_photometric_calibration;
  if (!f.good()) {
    LOG(WARNING) << "PhotometricUndistorter: Could not open file!";
    return;
  }

  {
    std::string line;
    std::getline(f, line);
    std::istringstream l1i(line);
    std::vector<float> G_vec = std::vector<float>(
        std::istream_iterator<float>(l1i), std::istream_iterator<float>());

    G_depth_ = G_vec.size();

    if (G_depth_ < 256) {
      LOG(WARNING) << "PhotometricUndistorter: invalid format! got "
                   << G_vec.size()
                   << " entries in first line, expected at least 256!";
      return;
    }

    for (int i = 0; i < G_depth_; ++i) {
      G_[i] = G_vec[i];
    }

    for (int i = 0; i < G_depth_ - 1; ++i) {
      if (G_[i + 1] <= G_[i]) {
        LOG(INFO) << "PhotometricUndistorter: G invalid! it has to be strictly "
                     "increasing, but it is not!";
        return;
      }
    }

    const float min = G_[0];
    const float max = G_[G_depth_ - 1];
    for (int i = 0; i < G_depth_; ++i) {
      // make it to 0..255
      G_[i] = 255.0 * (G_[i] - min) / (max - min);
    }
  }

  if (setting_photometricCalibration == 0) {
    for (int i = 0; i < G_depth_; ++i) {
      G_[i] = 255.f * static_cast<float>(i) / static_cast<float>(G_depth_ - 1);
    }
  }

  LOG(INFO) << "Reading Vignette Image from " << image_vignette;
  MinimalImage<unsigned short>* vig_img_16u =
      IOWrap::readImageBW_16U(image_vignette.c_str());
  MinimalImageB* vig_img_8u = IOWrap::readImageBW_8U(image_vignette.c_str());
  const int wh = w_ * h_;
  vignette_map_ = new float[wh];
  vignette_map_inv_ = new float[wh];

  if (vig_img_16u != nullptr) {
    if (vig_img_16u->w != w || vig_img_16u->h != h) {
      LOG(ERROR) << "PhotometricUndistorter: Invalid vignette image size! got "
                 << vig_img_16u->w << " x " << vig_img_16u->h << ", "
                 << "expected " << w << " x " << h;
      if (vig_img_16u != nullptr) {
        delete vig_img_16u;
      }
      if (vig_img_8u != nullptr) {
        delete vig_img_8u;
      }
      return;
    }

    float max_v = 0;
    for (int i = 0; i < wh; ++i) {
      const float val = static_cast<float>(vig_img_16u->at(i));
      if (val > max_v) {
        max_v = val;
      }
    }

    for (int i = 0; i < wh; ++i) {
      const float val = static_cast<float>(vig_img_16u->at(i));
      vignette_map_[i] = val / max_v;
    }
  } else if (vig_img_8u != nullptr) {
    if (vig_img_8u->w != w || vig_img_8u->h != h) {
      LOG(ERROR) << "PhotometricUndistorter: Invalid vignette image size! got "
                 << vig_img_8u->w << " x " << vig_img_8u->h << ", "
                 << "expected " << w << " x " << h;
      if (vig_img_16u != nullptr) {
        delete vig_img_16u;
      }
      if (vig_img_8u != nullptr) {
        delete vig_img_8u;
      }
      return;
    }

    float max_v = 0;
    for (int i = 0; i < wh; ++i) {
      const float val = static_cast<float>(vig_img_8u->at(i));
      if (val > max_v) {
        max_v = val;
      }
    }

    for (int i = 0; i < wh; ++i) {
      const float val = static_cast<float>(vig_img_8u->at(i));
      vignette_map_[i] = val / max_v;
    }
  } else {
    LOG(ERROR) << "PhotometricUndistorter: Invalid vignette image";
    if (vig_img_16u != nullptr) {
      delete vig_img_16u;
    }
    if (vig_img_8u != nullptr) {
      delete vig_img_8u;
    }
    return;
  }

  if (vig_img_16u != nullptr) {
    delete vig_img_16u;
  }
  if (vig_img_8u != nullptr) {
    delete vig_img_8u;
  }

  for (int i = 0; i < wh; ++i) {
    vignette_map_inv_[i] = 1.f / vignette_map_[i];
  }

  LOG(INFO) << "Successfully read photometric calibration!";
  valid_ = true;
}

PhotometricUndistorter::~PhotometricUndistorter() {
  if (vignette_map_ != 0) {
    delete[] vignette_map_;
  }
  if (vignette_map_inv_ != 0) {
    delete[] vignette_map_inv_;
  }
  delete output_;
}

void PhotometricUndistorter::UnMapFloatImage(float* const image) {
  int wh = w_ * h_;
  const float G_depth_float = static_cast<float>(G_depth_);
  for (int i = 0; i < wh; ++i) {
    float BinvC;
    float color = image[i];
    if (color < 1e-3f) {
      BinvC = 0.f;
    } else if (color > G_depth_float - 1.01f) {
      BinvC = G_depth_float - 1.1f;
    } else {
      const int c = static_cast<int>(color);
      const float a = color - static_cast<float>(c);
      BinvC = G_[c] * (1.f - a) + G_[c + 1] * a;
    }

    image[i] = (BinvC < 0.f ? 0.f : BinvC);
  }
}
}  // dso