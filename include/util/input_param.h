#pragma once

#include <map>
#include <string>
#include <vector>

#include "util/num_type.h"

namespace dso {
using TimestampWithSE3 =
    std::map<double, SE3, std::less<double>, Eigen::aligned_allocator<SE3>>;

struct InputParam {
  int preset = 100;
  int mode = 100;

  int start_id = 0;
  int end_id = 10000000;

  float play_speed = 0.f;
  double rescale = 0.;

  std::string path_2_timestamps = "";
  std::string path_2_images = "";
  std::string path_2_calibration = "";
  std::string path_2_vignette = "";
  std::string path_2_gamma = "";
  std::string path_2_log = "";
  std::string path_2_scales = "";

  bool use_scales = false;
  bool use_sample_output = false;
  bool use_pcl_output = false;
  bool quiet = false;
  bool no_log = false;
  bool reverse = false;
  bool prefetch = false;
  bool no_gui = false;
  bool multi_threading = true;
  bool save = false;
  bool preload = false;
  bool disable_ros = false;
  bool disable_reconfigure = false;
};

}  // dso
