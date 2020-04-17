#pragma once

#include <glog/logging.h>

#include "undistorter/undistorter.h"
#include "util/global_calib.h"

#if HAS_ZIPLIB
#include "zip.h"
#endif

namespace dso {

class DatasetReader {
 public:
  DatasetReader(const std::string& path, const std::string& file_calibration,
                const std::string& file_gamma,
                const std::string& file_vignette);

  DatasetReader(const std::string& path, const std::string& file_calibration,
                const std::string& file_gamma, const std::string& file_vignette,
                const std::string& file_timestamps);
  ~DatasetReader();

  Eigen::VectorXf GetOriginalCalib() {
    return undistorter_->GetOriginalParameter().cast<float>();
  }

  Eigen::Vector2i GetOriginalDimensions() {
    return undistorter_->GetOriginalSize();
  }

  void GetCalibMono(Eigen::Matrix3f* const K, int* const w, int* const h) {
    CHECK_NOTNULL(K);
    CHECK_NOTNULL(w);
    CHECK_NOTNULL(h);
    *K = undistorter_->GetK().cast<float>();
    *w = undistorter_->GetSize()[0];
    *h = undistorter_->GetSize()[1];
  }

  void SetGlobalCalibration() {
    int w_out, h_out;
    Eigen::Matrix3f K;
    GetCalibMono(&K, &w_out, &h_out);
    SetGlobalCalib(w_out, h_out, K);
  }

  size_t GetNumImages() const { return files_.size(); }

  double GetTimestamp(const int id) const {
    if (timestamps_.size() == 0) {
      return id * 0.1;
    }
    if (id < 0) {
      return 0.;
    } else if (static_cast<size_t>(id) >= timestamps_.size()) {
      return 0.;
    }
    return timestamps_[id];
  }

  MinimalImageB* GetImageRaw(const int id) {
    return GetImageRawInternal(id, 0);
  }

  ImageAndExposure* GetImage(const int id,
                             const bool force_load_directly = false) {
    return GetImageInternal(id, 0);
  }

  float* GetPhotometricGamma() {
    if (undistorter_ == nullptr ||
        undistorter_->photometric_undistorter_ == nullptr) {
      return nullptr;
    }
    return undistorter_->photometric_undistorter_->GetG();
  }

 public:
  // Undistorter. [0] always exists, [1-2] only when MT is enabled.
  Undistorter* undistorter_;

 private:
  void SetFiles(std::string dir);

  MinimalImageB* GetImageRawInternal(const int id, const int unused);
  ImageAndExposure* GetImageInternal(const int id, const int unused);

  void LoadTimestamps();
  void LoadTimestamps(const std::string& file_timestamps);

  std::vector<ImageAndExposure*> preloaded_images_;
  std::vector<std::string> files_;
  std::vector<double> timestamps_;
  std::vector<float> exposures_;

  int width_, height_;
  int width_org_, height_org_;

  std::string path_;
  std::string file_calibration_;

  bool is_zipped_;

#if HAS_ZIPLIB
  zip_t* zip_archive_;
  char* data_buffer_;
#endif
};
}
