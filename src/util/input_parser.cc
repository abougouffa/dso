#include "util/input_parser.h"

#include <iostream>
#include <iterator>
#include <string>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "util/converter.h"
#include "util/input_param.h"
#include "util/num_type.h"
#include "util/settings.h"

namespace dso {

InputParam InputParser::Read(const std::string& config) {
  cv::FileStorage settings(config, cv::FileStorage::READ);
  LOG_IF(FATAL, !settings.isOpened()) << "Failed to open settings file at: "
                                      << config;

  InputParam param;
  if (!settings["Int.Preset"].empty()) {
    settings["Int.Preset"] >> param.preset;
  }
  if (!settings["Int.Mode"].empty()) {
    settings["Int.Mode"] >> param.mode;
  }
  if (!settings["Int.StartId"].empty()) {
    settings["Int.StartId"] >> param.start_id;
  }
  if (!settings["Int.EndId"].empty()) {
    settings["Int.EndId"] >> param.end_id;
  }

  if (!settings["Float.PlaySpeed"].empty()) {
    settings["Float.PlaySpeed"] >> param.play_speed;
  }

  if (!settings["Double.Rescale"].empty()) {
    settings["Double.Rescale"] >> param.rescale;
  }

  if (!settings["String.Timestamps"].empty()) {
    settings["String.Timestamps"] >> param.path_2_timestamps;
  }
  if (!settings["String.Images"].empty()) {
    settings["String.Images"] >> param.path_2_images;
  }
  if (!settings["String.Calib"].empty()) {
    settings["String.Calib"] >> param.path_2_calibration;
  }
  if (!settings["String.Vignette"].empty()) {
    settings["String.Vignette"] >> param.path_2_vignette;
  }
  if (!settings["String.Gamma"].empty()) {
    settings["String.Gamma"] >> param.path_2_gamma;
  }
  if (!settings["String.LogPath"].empty()) {
    settings["String.LogPath"] >> param.path_2_log;
  }
  if (!settings["String.Scales"].empty()) {
    settings["String.Scales"] >> param.path_2_scales;
  }

  if (!settings["Bool.UseScales"].empty()) {
    settings["Bool.UseScales"] >> param.use_scales;
  }
  if (!settings["Bool.UseSampleOutput"].empty()) {
    settings["Bool.UseSampleOutput"] >> param.use_sample_output;
  }
  if (!settings["Bool.UsePCLOutput"].empty()) {
    settings["Bool.UsePCLOutput"] >> param.use_pcl_output;
  }
  if (!settings["Bool.Quiet"].empty()) {
    settings["Bool.Quiet"] >> param.quiet;
  }
  if (!settings["Bool.NoLog"].empty()) {
    settings["Bool.NoLog"] >> param.no_log;
  }
  if (!settings["Bool.Reverse"].empty()) {
    settings["Bool.Reverse"] >> param.reverse;
  }
  if (!settings["Bool.Prefetch"].empty()) {
    settings["Bool.Prefetch"] >> param.prefetch;
  }
  if (!settings["Bool.NoGui"].empty()) {
    settings["Bool.NoGui"] >> param.no_gui;
  }
  if (!settings["Bool.MultiThreading"].empty()) {
    settings["Bool.MultiThreading"] >> param.multi_threading;
  }
  if (!settings["Bool.Save"].empty()) {
    settings["Bool.Save"] >> param.save;
  }
  if (!settings["Bool.Preload"].empty()) {
    settings["Bool.Preload"] >> param.preload;
  }
  if (!settings["Bool.DisableRos"].empty()) {
    settings["Bool.DisableRos"] >> param.disable_ros;
  }
  if (!settings["Bool.DisableReconfigure"].empty()) {
    settings["Bool.DisableReconfigure"] >> param.disable_ros;
  }

  return param;
}

void InputParser::Config(InputParam* const param) {
  CHECK_NOTNULL(param);

  if (param->no_log) {
    setting_logStuff = false;
    LOG(WARNING) << "DISABLE LOGGING";
  }
  ConfigLog(!param->no_log, param->path_2_log);

  Preset(param);
  SetMode(param->mode);
  if (param->quiet) {
    setting_debugout_runquiet = true;
    LOG(WARNING) << "QUIET MODE, I'll shut up!";
  }
  if (!param->disable_reconfigure) {
    disableReconfigure = true;
    LOG(WARNING) << "DISABLE RECONFIGURE!";
  }
  if (param->disable_ros) {
    disableReconfigure = true;
    LOG(WARNING) << "DISABLE ROS (AND RECONFIGURE)";
  }

  if (param->no_gui) {
    disableAllDisplay = true;
  }

  if (!param->multi_threading) {
    multiThreading = false;
    LOG(WARNING) << "NO Multi Threading!";
  }

  if (param->save) {
    debugSaveImages = true;
    if (42 == system("rm -rf images_out")) {
      LOG(WARNING) << "system call returned 42 - what are the odds?. This is "
                      "only here to shut up the compiler.";
    }
    if (42 == system("mkdir images_out")) {
      LOG(WARNING) << "system call returned 42 - what are the odds?. This is "
                      "only here to shut up the compiler.";
    }
    if (42 == system("rm -rf images_out")) {
      LOG(WARNING) << "system call returned 42 - what are the odds?. This is "
                      "only here to shut up the compiler.";
    }
    if (42 == system("mkdir images_out")) {
      LOG(WARNING) << "system call returned 42 - what are the odds?. This is "
                      "only here to shut up the compiler.";
    }
    LOG(WARNING) << "SAVE IMAGES!";
  }
}

void InputParser::Preset(InputParam* const param) {
  CHECK_NOTNULL(param);

  if (param->preset == 0 || param->preset == 1) {
    LOG(INFO) << "\n=============== PRESET Settings: ===============\nDEFAULT "
                 "settings:\n- "
              << (param->preset == 0 ? "no " : "1x")
              << " real-time enforcing\n"
                 "- 2000 active points\n"
                 "- 5-7 active frames\n"
                 "- 1-6 LM iteration each KF\n"
                 "- original image "
                 "resolution\n==============================================";
    param->play_speed = (param->preset == 0 ? 0 : 1);
    param->preload = (param->preset == 1);
    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;
    setting_logStuff = false;
  } else if (param->preset == 2 || param->preset == 3) {
    LOG(INFO) << "\n=============== PRESET Settings: ===============\nFAST "
                 "settings:\n- "
              << (param->preset == 2 ? "no " : "5x")
              << " real-time enforcing\n"
                 "- 800 active points\n"
                 "- 4-6 active frames\n"
                 "- 1-4 LM iteration each KF\n"
                 "- 424 x 320 image "
                 "resolution\n==============================================";

    param->play_speed = (param->preset == 2 ? 0 : 5);
    param->preload = (param->preset == 3);
    setting_desiredImmatureDensity = 600;
    setting_desiredPointDensity = 800;
    setting_minFrames = 4;
    setting_maxFrames = 6;
    setting_maxOptIterations = 4;
    setting_minOptIterations = 1;
    benchmarkSetting_width = 424;
    benchmarkSetting_height = 320;
    setting_logStuff = false;
  }
}

void InputParser::SetMode(const int mode) {
  switch (mode) {
    case 0:
      LOG(WARNING) << "PHOTOMETRIC MODE WITH CALIBRATION!";
      break;
    case 1:
      LOG(WARNING) << "PHOTOMETRIC MODE WITHOUT CALIBRATION!";
      setting_photometricCalibration = 0;
      setting_affineOptModeA = 0;
      setting_affineOptModeB = 0;
      break;
    case 2:
      LOG(WARNING) << "PHOTOMETRIC MODE WITH PERFECT IMAGES!";
      setting_photometricCalibration = 0;
      setting_affineOptModeA = -1;
      setting_affineOptModeB = -1;
      setting_minGradHistAdd = 3;
      break;
    default:
      break;
  }
}

void InputParser::ConfigLog(const bool _show_log,
                            const std::string& _log_path) {
  // 设置glog
  const std::string name = "DSO";

  const std::string log_path =
      _log_path.substr(0, _log_path.find_last_of('/')) + '/';

  FLAGS_alsologtostderr = _show_log;
  FLAGS_colorlogtostderr = true;  //设置记录到标准输出的颜色消息（如果终端支持）

  //设置是否在磁盘已满时避免日志记录到磁盘
  FLAGS_stop_logging_if_full_disk = true;

  FLAGS_max_log_size = 100;  //设置最大日志文件大小（以MB为单位）
  FLAGS_logbufsecs = 0;
  FLAGS_stderrthreshold = 1;

  google::InitGoogleLogging(name.c_str());
  if (!log_path.empty()) {
    std::cout << "Log file path: " << log_path << std::endl;
    google::SetLogDestination(google::GLOG_INFO,
                              (log_path + name + ".INFO.").c_str());
    google::SetLogDestination(google::GLOG_WARNING,
                              (log_path + name + ".WARNING.").c_str());
    google::SetLogDestination(google::GLOG_ERROR,
                              (log_path + name + ".ERROR.").c_str());
    google::SetLogDestination(google::GLOG_FATAL,
                              (log_path + name + ".FATAL.").c_str());
  } else {
    std::cout << "Log file path: /tmp/" << std::endl;
    std::cout << "\033[31mYou can local your own log path in yaml file.\033[0m"
              << std::endl;
  }
}

}  // dso
