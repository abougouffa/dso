#include "util/dataset_reader.h"

#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <string>

#include "io_wrapper/image_rw.h"

namespace dso {

DatasetReader::DatasetReader(const std::string& path,
                             const std::string& file_calibration,
                             const std::string& file_gamma,
                             const std::string& file_vignette)
    : path_(path), file_calibration_(file_calibration) {
#if HAS_ZIPLIB
  zip_archive_ = nullptr;
  data_buffer_ = nullptr;
#endif

  is_zipped_ = (path.length() > 4 && path.substr(path.length() - 4) == ".zip");

  if (is_zipped_) {
#if HAS_ZIPLIB
    int ziperror = 0;
    zip_archive_ = zip_open(path.c_str(), ZIP_RDONLY, &ziperror);
    LOG_IF(FATAL, ziperror != 0) << "ERROR " << ziperror << "reading archive "
                                 << path << "!";

    files_.clear();
    int numEntries = zip_get_num_entries(zip_archive_, 0);
    for (int k = 0; k < numEntries; ++k) {
      const char* name = zip_get_name(zip_archive_, k, ZIP_FL_ENC_STRICT);
      std::string nstr = std::string(name);
      if (nstr == "." || nstr == "..") {
        continue;
      }
      files_.emplace_back(name);
    }
    LOG(INFO) << "Got " << numEntries << " entries and " << files_.size()
              << " files_!";
    std::sort(files_.begin(), files_.end());
#else
    LOG(FATAL) << "Cannot read .zip archive, as compile without ziplib!";
#endif
  } else {
    SetFiles(path);
  }

  undistorter_ = Undistorter::GetUndistorterForFile(file_calibration,
                                                    file_gamma, file_vignette);

  width_org_ = undistorter_->GetOriginalSize()[0];
  height_org_ = undistorter_->GetOriginalSize()[1];
  width_ = undistorter_->GetSize()[0];
  height_ = undistorter_->GetSize()[1];

  // load timestamps_ if possible.
  LoadTimestamps();
  LOG(INFO) << "Got " << files_.size() << " files in " << path;
}

DatasetReader::DatasetReader(const std::string& path,
                             const std::string& file_calibration,
                             const std::string& file_gamma,
                             const std::string& file_vignette,
                             const std::string& file_timestamps)
    : path_(path), file_calibration_(file_calibration) {
#if HAS_ZIPLIB
  zip_archive_ = nullptr;
  data_buffer_ = nullptr;
#endif

  is_zipped_ = (path.length() > 4 && path.substr(path.length() - 4) == ".zip");

  if (is_zipped_) {
#if HAS_ZIPLIB
    int ziperror = 0;
    zip_archive_ = zip_open(path.c_str(), ZIP_RDONLY, &ziperror);
    LOG_IF(FATAL, ziperror != 0) << "ERROR " << ziperror << "reading archive "
                                 << path << "!";

    files_.clear();
    int numEntries = zip_get_num_entries(zip_archive_, 0);
    for (int k = 0; k < numEntries; ++k) {
      const char* name = zip_get_name(zip_archive_, k, ZIP_FL_ENC_STRICT);
      std::string nstr = std::string(name);
      if (nstr == "." || nstr == "..") {
        continue;
      }
      files_.emplace_back(name);
    }
    LOG(INFO) << "Got " << numEntries << " entries and " << files_.size()
              << " files_!";
    std::sort(files_.begin(), files_.end());
#else
    LOG(FATAL) << "Cannot read .zip archive, as compile without ziplib!";
#endif
  } else {
    SetFiles(path);
  }

  undistorter_ = Undistorter::GetUndistorterForFile(file_calibration,
                                                    file_gamma, file_vignette);

  width_org_ = undistorter_->GetOriginalSize()[0];
  height_org_ = undistorter_->GetOriginalSize()[1];
  width_ = undistorter_->GetSize()[0];
  height_ = undistorter_->GetSize()[1];

  // load timestamps_ if possible.
  // LoadTimestamps();
  LoadTimestamps(file_timestamps);
  LOG(INFO) << "Got " << files_.size() << " files in " << path;
}

DatasetReader::~DatasetReader() {
#if HAS_ZIPLIB
  if (zip_archive_ != nullptr) {
    zip_close(zip_archive_);
  }
  if (data_buffer_ != nullptr) {
    delete data_buffer_;
  }
#endif

  delete undistorter_;
};

void DatasetReader::SetFiles(std::string dir) {
  DIR* dp;
  if ((dp = opendir(dir.c_str())) == nullptr) {
    return;
  }
  dirent* dirp;
  while ((dirp = readdir(dp)) != nullptr) {
    const std::string name(dirp->d_name);
    if (name != "." && name != "..") {
      files_.emplace_back(name);
    }
  }
  closedir(dp);

  std::sort(files_.begin(), files_.end());

  if (dir.at(dir.length() - 1) != '/') {
    dir.append("/");
  }
  for (size_t i = 0; i < files_.size(); ++i) {
    if (files_[i].at(0) != '/') {
      files_[i] = dir + files_[i];
    }
  }
}

MinimalImageB* DatasetReader::GetImageRawInternal(const int id,
                                                  const int unused) {
  if (!is_zipped_) {
    // CHANGE FOR ZIP FILE
    return IOWrap::readImageBW_8U(files_[id]);
  } else {
#if HAS_ZIPLIB
    if (data_buffer_ == 0)
      data_buffer_ = new char[width_org_ * height_org_ * 6 + 10000];
    zip_file_t* fle = zip_fopen(zip_archive_, files_[id].c_str(), 0);
    long readbytes = zip_fread(fle, data_buffer_,
                               (long)width_org_ * height_org_ * 6 + 10000);

    if (readbytes > (long)width_org_ * height_org_ * 6) {
      LOG(WARNING) << "Read " << readbytes << "/"
                   << (width_org_ * height_org_ * 6 + 10000)
                   << " bytes for file " << files_[id] << ". Increase buffer!!";
      delete[] data_buffer_;
      data_buffer_ = new char[(long)width_org_ * height_org_ * 30];
      fle = zip_fopen(zip_archive_, files_[id].c_str(), 0);
      readbytes = zip_fread(fle, data_buffer_,
                            (long)width_org_ * height_org_ * 30 + 10000);

      LOG_IF(FATAL,
             readbytes > static_cast<long>(width_org_ * height_org_ * 30))
          << "Buffer still too small (read " << readbytes << " / "
          << (width_org_ * height_org_ * 30 + 10000) << ". Abort.";
    }

    return IOWrap::readStreamBW_8U(data_buffer_, readbytes);
#else
    LOG(FATAL) << "Cannot read .zip archive, as compile without ziplib!";
#endif
  }
}

ImageAndExposure* DatasetReader::GetImageInternal(const int id,
                                                  const int unused) {
  MinimalImageB* minimg = GetImageRawInternal(id, 0);
  ImageAndExposure* ret2 = undistorter_->Undistort<unsigned char>(
      minimg, (exposures_.size() == 0 ? 1.0f : exposures_[id]),
      (timestamps_.size() == 0 ? 0.0 : timestamps_[id]));
  
  ret2->init_scale = (scales_.size() == 0) ? 1. : scales_[id];

  delete minimg;
  return ret2;
}

void DatasetReader::LoadTimestamps() {
  std::ifstream tr;
  std::string timesFile =
      path_.substr(0, path_.find_last_of('/')) + "/times.txt";
  tr.open(timesFile.c_str());
  while (!tr.eof() && tr.good()) {
    std::string line;
    char buf[1000];
    tr.getline(buf, 1000);

    int id;
    double stamp;
    float exposure = 0;

    if (3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure)) {
      timestamps_.emplace_back(stamp);
      exposures_.emplace_back(exposure);
    } else if (2 == sscanf(buf, "%d %lf", &id, &stamp)) {
      timestamps_.emplace_back(stamp);
      exposures_.emplace_back(exposure);
    }
  }
  tr.close();

  // check if exposures_ are correct, (possibly skip)
  bool exposuresGood = (exposures_.size() == GetNumImages());
  for (int i = 0; i < (int)exposures_.size(); ++i) {
    if (exposures_[i] == 0) {
      // fix!
      float sum = 0, num = 0;
      if (i > 0 && exposures_[i - 1] > 0) {
        sum += exposures_[i - 1];
        ++num;
      }
      if (i + 1 < (int)exposures_.size() && exposures_[i + 1] > 0) {
        sum += exposures_[i + 1];
        ++num;
      }

      if (num > 0) {
        exposures_[i] = sum / num;
      }
    }

    if (exposures_[i] == 0) {
      exposuresGood = false;
    }
  }

  if (GetNumImages() != timestamps_.size()) {
    LOG(WARNING) << "set timestamps and exposures to zero!";
    exposures_.clear();
    timestamps_.clear();
  }

  if (GetNumImages() != exposures_.size() || !exposuresGood) {
    LOG(WARNING) << "set exposures to zero!";
    exposures_.clear();
  }

  LOG(INFO) << "Got " << GetNumImages() << " images and " << timestamps_.size()
            << " timestamps and " << exposures_.size() << " exposures!";
}

void DatasetReader::LoadTimestamps(const std::string& file_timestamps) {
  LOG(INFO) << "Reading timestamps from file " << file_timestamps;
  std::ifstream in;
  in.open(file_timestamps.c_str());
  while (!in.eof()) {
    std::string s;
    getline(in, s);
    if (!s.empty()) {
      std::stringstream ss;
      ss << s;
      if (ss.good()) {
        double t;
        ss >> t;
        timestamps_.emplace_back(t);
      }
    }
  }

  if (GetNumImages() != timestamps_.size()) {
    LOG(WARNING) << "set timestamps to zero!";
    timestamps_.clear();
  }

  LOG(INFO) << "Got " << GetNumImages() << " images and " << timestamps_.size()
            << " timestamps!";
}

  void DatasetReader::LoadScales(const std::string &file_scales) {
    LOG(INFO) << "Reading scales from file " << file_scales;
    std::ifstream in;
    in.open(file_scales.c_str());
    if (in.is_open()) {
      while (!in.eof()) {
        std::string s;
        getline(in, s);
        if (!s.empty()) {
          std::stringstream ss;
          ss << s;
          if (ss.good()) {
            double scale;
            ss >> scale;
            scales_.emplace_back(scale);
          }
        }
      }
      in.close();
    } else {
      LOG(WARNING) << "Unable to open scales file";
    }
  }

  } // dso