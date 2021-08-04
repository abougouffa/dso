#pragma once

#include <string>

#include "util/input_param.h"

namespace dso {

class InputParser {
 public:
  static InputParam Read(const std::string& config);
  static void Config(InputParam* const input);

 private:
  static void Preset(InputParam* const input);
  static void SetMode(const int mode);
  static void ConfigLog(const bool _show_log, const std::string& _log_path);
  static void LoadTrajectory(const std::string& file_path, const SE3& Tdc,
                             TimestampWithSE3* const time_and_pose);
};

}  // dso