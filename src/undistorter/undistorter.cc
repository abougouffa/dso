#include "undistorter/undistorter.h"

#include <fstream>

#include "undistorter/undistorter_equidistant.h"
#include "undistorter/undistorter_fov.h"
#include "undistorter/undistorter_kb.h"
#include "undistorter/undistorter_pinhole.h"
#include "undistorter/undistorter_rad_tan.h"

namespace dso {

Undistorter::~Undistorter() {
  if (remap_x_ != nullptr) {
    delete[] remap_x_;
  }
  if (remap_y_ != nullptr) {
    delete[] remap_y_;
  }
}

Undistorter* Undistorter::GetUndistorterForFile(
    const std::string& file_config, const std::string& file_gamma,
    const std::string& file_vignette) {
  LOG(INFO) << "Reading Calibration from file " << file_config;

  std::ifstream f(file_config.c_str());
  if (!f.good()) {
    f.close();
    LOG(WARNING) << " ... not found. Cannot operate without calibration, "
                    "shutting down.";
    f.close();
    return 0;
  }

  LOG(INFO) << " ... found!";
  std::string l1;
  std::getline(f, l1);
  f.close();

  float ic[10];

  Undistorter* u;

  if (std::sscanf(l1.c_str(), "%f %f %f %f %f %f %f %f", &ic[0], &ic[1], &ic[2],
                  &ic[3], &ic[4], &ic[5], &ic[6], &ic[7]) == 8) {
    // for backwards-compatibility: Use RadTan model for 8 parameters.
    LOG(INFO) << "found RadTan (OpenCV) camera model, building rectifier.";
    u = new UndistortRadTan(file_config.c_str(), true);
    if (!u->IsValid()) {
      delete u;
      return 0;
    }
  } else if (std::sscanf(l1.c_str(), "%f %f %f %f %f", &ic[0], &ic[1], &ic[2],
                         &ic[3], &ic[4]) == 5) {
    // for backwards-compatibility: Use Pinhole / FoV model for 5 parameter.
    if (ic[4] == 0) {
      LOG(INFO) << "found PINHOLE camera model, building rectifier.";
      u = new UndistortPinhole(file_config.c_str(), true);
      if (!u->IsValid()) {
        delete u;
        return 0;
      }
    } else {
      LOG(INFO) << "found ATAN camera model, building rectifier.";
      u = new UndistortFOV(file_config.c_str(), true);
      if (!u->IsValid()) {
        delete u;
        return 0;
      }
    }
  } else if (std::sscanf(l1.c_str(), "KannalaBrandt %f %f %f %f %f %f %f %f",
                         &ic[0], &ic[1], &ic[2], &ic[3], &ic[4], &ic[5], &ic[6],
                         &ic[7]) == 8) {
    // clean model selection implementation.
    u = new UndistortKB(file_config.c_str(), false);
    if (!u->IsValid()) {
      delete u;
      return 0;
    }
  } else if (std::sscanf(l1.c_str(), "RadTan %f %f %f %f %f %f %f %f", &ic[0],
                         &ic[1], &ic[2], &ic[3], &ic[4], &ic[5], &ic[6],
                         &ic[7]) == 8) {
    u = new UndistortRadTan(file_config.c_str(), false);
    if (!u->IsValid()) {
      delete u;
      return 0;
    }
  } else if (std::sscanf(l1.c_str(), "EquiDistant %f %f %f %f %f %f %f %f",
                         &ic[0], &ic[1], &ic[2], &ic[3], &ic[4], &ic[5], &ic[6],
                         &ic[7]) == 8) {
    u = new UndistortEquidistant(file_config.c_str(), false);
    if (!u->IsValid()) {
      delete u;
      return 0;
    }
  } else if (std::sscanf(l1.c_str(), "FOV %f %f %f %f %f", &ic[0], &ic[1],
                         &ic[2], &ic[3], &ic[4]) == 5) {
    u = new UndistortFOV(file_config.c_str(), false);
    if (!u->IsValid()) {
      delete u;
      return 0;
    }
  } else if (std::sscanf(l1.c_str(), "Pinhole %f %f %f %f %f", &ic[0], &ic[1],
                         &ic[2], &ic[3], &ic[4]) == 5) {
    u = new UndistortPinhole(file_config.c_str(), false);
    if (!u->IsValid()) {
      delete u;
      return 0;
    }
  } else {
    LOG(FATAL) << "could not read calib file! exit.";
  }

  u->LoadPhotometricCalibration(file_gamma, "", file_vignette);

  return u;
}

void Undistorter::LoadPhotometricCalibration(
    const std::string& file_photometric_calibration,
    const std::string& image_noise, const std::string& image_vignette) {
  const Eigen::Vector2i size = GetOriginalSize();
  photometric_undistorter_ =
      new PhotometricUndistorter(file_photometric_calibration, image_noise,
                                 image_vignette, size[0], size[1]);
}

void Undistorter::ApplyBlurNoise(float* const img) const {
  CHECK_NOTNULL(img);
  if (benchmark_varBlurNoise == 0.f) {
    return;
  }

  const int num_noise =
      (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
  float* const noise_map_x = new float[num_noise];
  float* const noise_map_y = new float[num_noise];
  float* const blut_tmp = new float[w_ * h_];

  if (benchmark_varBlurNoise > 0) {
    for (int i = 0; i < num_noise; ++i) {
      noise_map_x[i] = benchmark_varBlurNoise * (static_cast<float>(rand()) /
                                                 static_cast<float>(RAND_MAX));
      noise_map_y[i] = benchmark_varBlurNoise * (static_cast<float>(rand()) /
                                                 static_cast<float>(RAND_MAX));
    }
  }

  float gauss_map[1000];
  for (int i = 0; i < 1000; ++i) {
    gauss_map[i] = expf(static_cast<float>(-i * i) / (100.f * 100.f));
  }

  // x-blur.
  for (int y = 0; y < h_; ++y) {
    for (int x = 0; x < w_; ++x) {
      const float _x = 4.f +
                       (static_cast<float>(x) / static_cast<float>(w_)) *
                           static_cast<float>(benchmark_noiseGridsize);
      const float _y = 4.f +
                       (static_cast<float>(y) / static_cast<float>(h_)) *
                           static_cast<float>(benchmark_noiseGridsize);
      float x_blur = getInterpolatedElement11BiCub(noise_map_x, _x, _y,
                                                   benchmark_noiseGridsize + 8);
      if (x_blur < 0.01f) {
        x_blur = 0.01f;
      }

      const int kernel_size = 1 + static_cast<int>(1.f + x_blur * 1.5f);
      float sum_w = 0.f;
      float sum_cw = 0.f;
      for (int dx = 0; dx <= kernel_size; ++dx) {
        int g_mid =
            static_cast<int>(100.f * static_cast<float>(dx) / x_blur + 0.5f);
        if (g_mid > 900) {
          g_mid = 900;
        }
        const float gw = gauss_map[g_mid];

        if (x + dx > 0 && x + dx < w_) {
          sum_w += gw;
          sum_cw += gw * img[x + dx + y * w_];
        }

        if (x - dx > 0 && x - dx < w_ && dx != 0) {
          sum_w += gw;
          sum_cw += gw * img[x - dx + y * w_];
        }
      }

      blut_tmp[x + y * w_] = sum_cw / sum_w;
    }
  }

  // y-blur.
  for (int x = 0; x < w_; ++x) {
    for (int y = 0; y < h_; ++y) {
      const float _x = 4.f +
                       (static_cast<float>(x) / static_cast<float>(w_)) *
                           static_cast<float>(benchmark_noiseGridsize);
      const float _y = 4.f +
                       (static_cast<float>(y) / static_cast<float>(h_)) *
                           static_cast<float>(benchmark_noiseGridsize);
      float y_blur = getInterpolatedElement11BiCub(noise_map_y, _x, _y,
                                                   benchmark_noiseGridsize + 8);

      if (y_blur < 0.01) {
        y_blur = 0.01;
      }

      const int kernel_size = 1 + static_cast<int>(0.9f + y_blur * 2.5f);
      float sum_w = 0;
      float sum_cw = 0;
      for (int dy = 0; dy <= kernel_size; ++dy) {
        int g_mid =
            static_cast<int>(100.f * static_cast<float>(dy) / y_blur + 0.5f);
        if (g_mid > 900) {
          g_mid = 900;
        }
        float gw = gauss_map[g_mid];

        if (y + dy > 0 && y + dy < h_) {
          sum_w += gw;
          sum_cw += gw * blut_tmp[x + (y + dy) * w_];
        }

        if (y - dy > 0 && y - dy < h_ && dy != 0) {
          sum_w += gw;
          sum_cw += gw * blut_tmp[x + (y - dy) * w_];
        }
      }
      img[x + y * w_] = sum_cw / sum_w;
    }
  }

  delete[] noise_map_x;
  delete[] noise_map_y;
}

void Undistorter::MakeOptimalKCrop() {
  LOG(WARNING) << "finding CROP optimal new model!";
  K_.setIdentity();

  // 1. stretch the center lines as far as possible, to get initial coarse
  // guess.
  float* tg_x = new float[100000];
  float* tg_y = new float[100000];
  float min_x = 0.f;
  float max_x = 0.f;
  float min_y = 0.f;
  float max_y = 0.f;

  for (int x = 0; x < 100000; ++x) {
    tg_x[x] = (x - 50000.f) / 10000.f;
    tg_y[x] = 0.f;
  }
  DistortCoordinates(tg_x, tg_y, tg_x, tg_y, 100000);
  for (int x = 0; x < 100000; ++x) {
    if (tg_x[x] > 0 && tg_x[x] < w_org_ - 1) {
      if (min_x == 0.f) {
        min_x = (x - 50000.f) / 10000.f;
      }
      max_x = (x - 50000.f) / 10000.f;
    }
  }
  for (int y = 0; y < 100000; ++y) {
    tg_y[y] = (y - 50000.f) / 10000.f;
    tg_x[y] = 0.f;
  }
  DistortCoordinates(tg_x, tg_y, tg_x, tg_y, 100000);
  for (int y = 0; y < 100000; ++y) {
    if (tg_y[y] > 0 && tg_y[y] < h_org_ - 1) {
      if (min_y == 0.f) {
        min_y = (y - 50000.f) / 10000.f;
      }
      max_y = (y - 50000.f) / 10000.f;
    }
  }
  delete[] tg_x;
  delete[] tg_y;

  min_x *= 1.01f;
  max_x *= 1.01f;
  min_y *= 1.01f;
  max_y *= 1.01f;

  LOG(INFO) << "Initial range: " << min_x << " - " << max_x << "; y: " << min_y
            << " - " << max_y << "!";

  // 2. while there are invalid pixels at the border: shrink square at the side
  // that has invalid pixels, if several to choose from, shrink the wider
  // dimension.
  bool oob_left = true, oob_right = true, oob_top = true, oob_bottom = true;
  int iteration = 0;
  while (oob_left || oob_right || oob_top || oob_bottom) {
    oob_left = oob_right = oob_top = oob_bottom = false;
    for (int y = 0; y < h_; ++y) {
      remap_x_[y * 2] = min_x;
      remap_x_[y * 2 + 1] = max_x;
      remap_y_[y * 2] = min_y +
                        (max_y - min_y) * static_cast<float>(y) /
                            (static_cast<float>(h_) - 1.f);
      remap_y_[y * 2 + 1] = remap_y_[y * 2];
    }
    DistortCoordinates(remap_x_, remap_y_, remap_x_, remap_y_, 2 * h_);
    for (int y = 0; y < h_; ++y) {
      if (!(remap_x_[2 * y] > 0.f &&
            remap_x_[2 * y] < static_cast<float>(w_org_ - 1))) {
        oob_left = true;
      }
      if (!(remap_x_[2 * y + 1] > 0.f &&
            remap_x_[2 * y + 1] < static_cast<float>(w_org_ - 1))) {
        oob_right = true;
      }
    }

    for (int x = 0; x < w_; ++x) {
      remap_y_[x * 2] = min_y;
      remap_y_[x * 2 + 1] = max_y;
      remap_x_[x * 2] = min_x +
                        (max_x - min_x) * static_cast<float>(x) /
                            (static_cast<float>(w_) - 1.f);
      remap_x_[x * 2 + 1] = remap_x_[x * 2];
    }
    DistortCoordinates(remap_x_, remap_y_, remap_x_, remap_y_, 2 * w_);

    for (int x = 0; x < w_; ++x) {
      if (!(remap_y_[2 * x] > 0 &&
            remap_y_[2 * x] < static_cast<float>(h_org_ - 1))) {
        oob_top = true;
      }
      if (!(remap_y_[2 * x + 1] > 0 &&
            remap_y_[2 * x + 1] < static_cast<float>(h_org_ - 1))) {
        oob_bottom = true;
      }
    }

    if ((oob_left || oob_right) && (oob_top || oob_bottom)) {
      if ((max_x - min_x) > (max_y - min_y)) {
        // only shrink left/right
        oob_bottom = oob_top = false;
      } else {
        // only shrink top/bottom
        oob_left = oob_right = false;
      }
    }

    if (oob_left) {
      min_x *= 0.995f;
    }
    if (oob_right) {
      max_x *= 0.995f;
    }
    if (oob_top) {
      min_y *= 0.995f;
    }
    if (oob_bottom) {
      max_y *= 0.995f;
    }

    ++iteration;

    LOG(INFO) << "Iteration " << iteration << ": range: " << min_x << " - "
              << max_x << "; y: " << min_y << " - " << max_y << "!";
    LOG_IF(FATAL, iteration > 500) << "FAILED TO COMPUTE GOOD CAMERA MATRIX - "
                                      "SOMETHING IS SERIOUSLY WRONG. ABORTING";
  }

  K_(0, 0) = (static_cast<float>(w_) - 1.f) / (max_x - min_x);
  K_(1, 1) = (static_cast<float>(h_) - 1.f) / (max_y - min_y);
  K_(0, 2) = -min_x * K_(0, 0);
  K_(1, 2) = -min_y * K_(1, 1);
}

void Undistorter::MakeOptimalKFull() {
  // todo
  CHECK(false);
}

void Undistorter::ReadFromFile(const char* file_config, const int argc,
                               const std::string& prefix) {
  photometric_undistorter_ = nullptr;
  valid_ = false;
  pass_through_ = false;
  remap_x_ = nullptr;
  remap_y_ = nullptr;

  float output_calibration[5];

  argv_ = VecX(argc);

  // read parameters
  std::ifstream infile(file_config);
  CHECK(infile.good());

  std::string l1, l2, l3, l4;

  std::getline(infile, l1);
  std::getline(infile, l2);
  std::getline(infile, l3);
  std::getline(infile, l4);

  // l1 & l2
  if (argc == 5) {
    // fov model
    char buf[1000];
    snprintf(buf, 1000, "%s%%lf %%lf %%lf %%lf %%lf", prefix.c_str());

    if (std::sscanf(l1.c_str(), buf, &argv_[0], &argv_[1], &argv_[2], &argv_[3],
                    &argv_[4]) == 5 &&
        std::sscanf(l2.c_str(), "%d %d", &w_org_, &h_org_) == 2) {
      LOG(INFO) << "Input resolution: " << w_org_ << " " << h_org_;
      LOG(INFO) << "In: " << argv_[0] << " " << argv_[1] << " " << argv_[2]
                << " " << argv_[3] << " " << argv_[4];
    } else {
      LOG(ERROR) << "Failed to read camera calibration (invalid format?) \n "
                    "Calibration file: "
                 << file_config;
      infile.close();
      return;
    }
  } else if (argc == 8) {
    // KB, equi & radtan model
    char buf[1000];
    snprintf(buf, 1000, "%s%%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf",
             prefix.c_str());

    if (std::sscanf(l1.c_str(), buf, &argv_[0], &argv_[1], &argv_[2], &argv_[3],
                    &argv_[4], &argv_[5], &argv_[6], &argv_[7]) == 8 &&
        std::sscanf(l2.c_str(), "%d %d", &w_org_, &h_org_) == 2) {
      LOG(INFO) << "Input resolution: " << w_org_ << " " << h_org_;
      LOG(INFO) << "In: " << prefix << " " << argv_[0] << " " << argv_[1] << " "
                << argv_[2] << " " << argv_[3] << " " << argv_[4] << " "
                << argv_[5] << " " << argv_[6] << " " << argv_[7];
    } else {
      LOG(ERROR) << "Failed to read camera calibration (invalid format?) \n"
                    "Calibration file: "
                 << file_config;
      infile.close();
      return;
    }
  } else {
    LOG(ERROR) << "called with invalid number of parameters... forgot to "
                  "implement me?";
    infile.close();
    return;
  }

  if (argv_[2] < 1 && argv_[3] < 1) {
    LOG(WARNING)
        << "\n\nFound fx= " << argv_[0] << " , fy=" << argv_[1]
        << " , cx=" << argv_[2] << ", cy=" << argv_[3]
        << ".\n I'm assuming this is the \"relative\" calibration file "
           "format, and will rescale this by image width / height to fx= "
        << argv_[0] * w_org_ << " , fy=" << argv_[1] * h_org_
        << ", cx=" << argv_[2] * w_org_ - 0.5
        << ", cy=" << argv_[3] * h_org_ - 0.5;

    // rescale and substract 0.5 offset. the 0.5 is because I'm assuming the
    // calibration is given such that the pixel at (0,0) contains the integral
    // over intensity over [0,0]-[1,1], whereas I assume the pixel (0,0) to
    // contain a sample of the intensity ot [0,0], which is best approximated by
    // the integral over [-0.5,-0.5]-[0.5,0.5]. Thus, the shift by -0.5.
    argv_[0] = argv_[0] * w_org_;
    argv_[1] = argv_[1] * h_org_;
    argv_[2] = argv_[2] * w_org_ - 0.5;
    argv_[3] = argv_[3] * h_org_ - 0.5;
  }

  // l3
  if (l3 == "crop") {
    output_calibration[0] = -1;
    LOG(WARNING) << "Out: Rectify Crop";
  } else if (l3 == "full") {
    output_calibration[0] = -2;
    LOG(WARNING) << "Out: Rectify Full";
  } else if (l3 == "none") {
    output_calibration[0] = -3;
    LOG(WARNING) << "Out: No Rectification";
  } else if (std::sscanf(l3.c_str(), "%f %f %f %f %f", &output_calibration[0],
                         &output_calibration[1], &output_calibration[2],
                         &output_calibration[3], &output_calibration[4]) == 5) {
    LOG(WARNING) << "Out: " << output_calibration[0] << " "
                 << output_calibration[1] << " " << output_calibration[2] << " "
                 << output_calibration[3] << " " << output_calibration[4];
  } else {
    LOG(ERROR) << "Out: Failed to Read Output pars... not rectifying.";
    infile.close();
    return;
  }

  // l4
  if (std::sscanf(l4.c_str(), "%d %d", &w_, &h_) == 2) {
    if (benchmarkSetting_width != 0) {
      w_ = benchmarkSetting_width;
      if (output_calibration[0] == -3) {
        // crop instead of none, since probably resolution changed.
        output_calibration[0] = -1;
      }
    }
    if (benchmarkSetting_height != 0) {
      h_ = benchmarkSetting_height;
      if (output_calibration[0] == -3) {
        // crop instead of none, since probably resolution changed.
        output_calibration[0] = -1;
      }
    }
    LOG(WARNING) << "Output resolution: " << w_ << " " << h_;
  } else {
    LOG(ERROR) << "Out: Failed to Read Output resolution... not rectifying.";
    valid_ = false;
  }

  remap_x_ = new float[w_ * h_];
  remap_y_ = new float[w_ * h_];

  if (output_calibration[0] == -1) {
    MakeOptimalKCrop();
  } else if (output_calibration[0] == -2) {
    MakeOptimalKFull();
  } else if (output_calibration[0] == -3) {
    LOG_IF(FATAL, w_ != w_org_ || h_ != h_org_)
        << "ERROR: rectification mode none "
           "requires input and output_ "
           "dimenstions to match!";
    K_.setIdentity();
    K_(0, 0) = argv_[0];
    K_(1, 1) = argv_[1];
    K_(0, 2) = argv_[2];
    K_(1, 2) = argv_[3];
    pass_through_ = true;
  } else {
    if (output_calibration[2] > 1 || output_calibration[3] > 1) {
      LOG(ERROR)
          << "Given output_ calibration " << output_calibration[0] << " "
          << output_calibration[1] << " " << output_calibration[2] << " "
          << output_calibration[3]
          << " seems wrong. It needs to be relative to image width / height!";
    }

    K_.setIdentity();
    K_(0, 0) = output_calibration[0] * w_;
    K_(1, 1) = output_calibration[1] * h_;
    K_(0, 2) = output_calibration[2] * w_ - 0.5;
    K_(1, 2) = output_calibration[3] * h_ - 0.5;
  }

  if (benchmarkSetting_fxfyfac != 0.) {
    K_(0, 0) = fmax(benchmarkSetting_fxfyfac, K_(0, 0));
    K_(1, 1) = fmax(benchmarkSetting_fxfyfac, K_(1, 1));

    // cannot pass through when fx / fy have been overwritten.
    pass_through_ = false;
  }

  for (int y = 0; y < h_; ++y) {
    for (int x = 0; x < w_; ++x) {
      remap_x_[x + y * w_] = x;
      remap_y_[x + y * w_] = y;
    }
  }

  DistortCoordinates(remap_x_, remap_y_, remap_x_, remap_y_, h_ * w_);

  // 原代码逻辑
  // for (int y = 0; y < h_; ++y)
  //   for (int x = 0; x < w_; ++x) {
  //     // make rounding resistant.
  //     float ix = remap_x_[x + y * w_];
  //     float iy = remap_y_[x + y * w_];

  //     if (ix == 0) {
  //       ix = 0.001;
  //     }
  //     if (iy == 0) {
  //       iy = 0.001;
  //     }
  //     if (ix == w_org_ - 1) {
  //       ix = w_org_ - 1.001;
  //     }
  //     if (iy == h_org_ - 1) {
  //       ix = h_org_ - 1.001;
  //     }

  //     if (ix > 0 && iy > 0 && ix < w_org_ - 1 && iy < w_org_ - 1) {
  //       remap_x_[x + y * w_] = ix;
  //       remap_y_[x + y * w_] = iy;
  //     } else {
  //       remap_x_[x + y * w_] = -1;
  //       remap_y_[x + y * w_] = -1;
  //     }
  //   }

  for (int y = 0; y < h_; ++y) {
    for (int x = 0; x < w_; ++x) {
      // make rounding resistant.
      float ix = remap_x_[x + y * w_];
      float iy = remap_y_[x + y * w_];

      if (ix == 0.f) {
        ix = 0.001f;
      }
      if (iy == 0.f) {
        iy = 0.001f;
      }
      if (ix == static_cast<float>(w_org_ - 1)) {
        ix = static_cast<float>(w_org_) - 1.001f;
      }
      if (iy == static_cast<float>(h_org_ - 1)) {
        iy = static_cast<float>(h_org_) - 1.001f;
      }

      // Note: 从这开始和原本代码有些许不同
      if (ix > 0.f && iy > 0.f && ix < static_cast<float>(w_org_ - 1) &&
          iy < static_cast<float>(h_org_ - 1)) {
        remap_x_[x + y * w_] = ix;
        remap_y_[x + y * w_] = iy;
      } else {
        remap_x_[x + y * w_] = -1.f;
        remap_y_[x + y * w_] = -1.f;
      }
    }
  }

  valid_ = true;

  LOG(INFO) << "Rectified Camera Matrix:\n" << K_;
}
}
