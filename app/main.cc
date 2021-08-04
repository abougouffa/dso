#include <signal.h>
#include <thread>

#include <glog/logging.h>

#include "full_system/full_system.h"
#include "io_wrapper/output_wrapper/pointcloud_output_wrapper.h"
#include "io_wrapper/output_wrapper/sample_output_wrapper.h"
#include "io_wrapper/pangolin/pangolin_dso_viewer.h"
#include "util/dataset_reader.h"
#include "util/input_parser.h"

using namespace dso;

void ExitHandler(int s) {
  LOG(WARNING) << "Caught signal " << s;
  exit(1);
}

bool first_ros_spin = false;

void ExitThread() {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = ExitHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  first_ros_spin = true;
  while (true) {
    pause();
  }
}

int main(int argc, char **argv) {
  LOG_IF(FATAL, argc < 2) << "Usage: ./dso path_to_configuration";

  InputParam param = InputParser::Read(argv[1]);
  InputParser::Config(&param);

  // hook crtl+C.
  boost::thread exit_thread = boost::thread(ExitThread);

  DatasetReader* reader;
  if (param.path_2_timestamps != "") {
    reader = new DatasetReader(
      param.path_2_images, param.path_2_calibration, param.path_2_gamma,
      param.path_2_vignette, param.path_2_timestamps);
  } else {
  reader = new DatasetReader(
      param.path_2_images, param.path_2_calibration, param.path_2_gamma,
      param.path_2_vignette);
  }

  reader->SetGlobalCalibration();

  LOG_IF(FATAL, setting_photometricCalibration > 0 &&
                    reader->GetPhotometricGamma() == 0)
      << "Don't have photometric calibation. Need to use commandline options "
         "mode=1 or mode=2";

  const int num_of_images = static_cast<int>(reader->GetNumImages());
  int lstart = param.start_id;
  int lend = param.end_id;
  int linc = 1;
  if (param.reverse) {
    LOG(WARNING) << "REVERSE!!!!";
    lstart = param.end_id - 1;
    if (lstart >= num_of_images) {
      lstart = num_of_images - 1;
    }
    lend = param.start_id;
    linc = -1;
  }

  FullSystem* full_system = new FullSystem();
  full_system->setGammaFunction(reader->GetPhotometricGamma());
  full_system->linearizeOperation = (param.play_speed == 0.f);

  IOWrap::PangolinDSOViewer* viewer = 0;
  if (!disableAllDisplay) {
    viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
    full_system->outputWrapper.emplace_back(viewer);
  }

  if (param.use_sample_output) {
    full_system->outputWrapper.emplace_back(new IOWrap::SampleOutputWrapper());
  }

  if (param.use_pcl_output) {
    full_system->outputWrapper.emplace_back(
        new IOWrap::PointCloudOutputWrapper());
  }

  // to make MacOS happy: run this in dedicated thread -- and use this one to
  // run the GUI.
  std::thread runthread([&]() {
    std::vector<int> ids_to_play;
    std::vector<double> times_to_play_at;
    for (int i = lstart; i >= 0 && i < num_of_images && linc * i < linc * lend;
         i += linc) {
      ids_to_play.emplace_back(i);
      const double ts = reader->GetTimestamp(i);
      times_to_play_at.emplace_back(ts);
    }

    std::vector<ImageAndExposure*> preloaded_images;
    if (param.preload) {
      LOG(WARNING) << "LOADING ALL IMAGES!";
      for (size_t ii = 0; ii < ids_to_play.size(); ++ii) {
        int i = ids_to_play[ii];
        preloaded_images.emplace_back(reader->GetImage(i));
      }
    }

    timeval tv_start;
    gettimeofday(&tv_start, nullptr);
    clock_t started = clock();
    double s_initializer_offset = 0;

    for (int ii = 0; ii < static_cast<int>(ids_to_play.size()); ++ii) {
      if (!full_system->initialized) {
        // if not initialized: reset start time.
        gettimeofday(&tv_start, nullptr);
        started = clock();
        s_initializer_offset = times_to_play_at[ii];
      }

      int i = ids_to_play[ii];

      ImageAndExposure* img;
      if (param.preload) {
        img = preloaded_images[ii];
      } else {
        img = reader->GetImage(i);
      }

      full_system->addActiveFrame(img, i);

      delete img;

      if (full_system->initFailed || setting_fullResetRequested) {
        if (ii < 250 || setting_fullResetRequested) {
          LOG(WARNING) << "RESETTING!";

          std::vector<IOWrap::Output3DWrapper*> wraps =
              full_system->outputWrapper;
          delete full_system;

          for (IOWrap::Output3DWrapper* ow : wraps) {
            ow->reset();
          }

          full_system = new FullSystem();
          full_system->setGammaFunction(reader->GetPhotometricGamma());
          full_system->linearizeOperation = (param.play_speed == 0.f);
          full_system->outputWrapper = wraps;

          setting_fullResetRequested = false;
        }
      }

      if (full_system->isLost) {
        LOG(ERROR) << "LOST!!";
        break;
      }
    }
    full_system->blockUntilMappingIsFinished();
    clock_t ended = clock();
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);

    full_system->printResult("result.txt");

    const int frames_processed = abs(ids_to_play[0] - ids_to_play.back());
    const double seconds_processed =
        fabs(reader->GetTimestamp(ids_to_play[0]) -
             reader->GetTimestamp(ids_to_play.back()));
    const double milliseconds_taken_single =
        1000.f * (ended - started) / static_cast<float>(CLOCKS_PER_SEC);
    const double milliseconds_taken_mt =
        s_initializer_offset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.f +
                                (tv_end.tv_usec - tv_start.tv_usec) / 1000.f);
    LOG(INFO) << "\n======================\n"
              << frames_processed << " Frames ("
              << frames_processed / seconds_processed << " fps)\n"
              << milliseconds_taken_single / frames_processed
              << " ms per frame (single core);\n"
              << milliseconds_taken_mt / static_cast<float>(frames_processed)
              << "ms per frame (multi core); \n"
              << 1000 / (milliseconds_taken_single / seconds_processed)
              << "x (single core); \n"
              << 1000 / (milliseconds_taken_mt / seconds_processed)
              << "x (multi core);\n======================\n\n";
    // full_system->printFrameLifetimes();
    if (setting_logStuff) {
      std::ofstream tmp_log;
      tmp_log.open("logs/time.txt", std::ios::trunc | std::ios::out);
      tmp_log << 1000.f * (ended - started) /
                     static_cast<float>(CLOCKS_PER_SEC * num_of_images)
              << " "
              << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                  (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) /
                     static_cast<float>(num_of_images)
              << "\n";
      tmp_log.flush();
      tmp_log.close();
    }

  });

  if (viewer != nullptr) {
    viewer->run();
  }

  runthread.join();

  for (IOWrap::Output3DWrapper* ow : full_system->outputWrapper) {
    ow->join();
    delete ow;
  }

  LOG(WARNING) << "DELETE FULLSYSTEM!";
  delete full_system;

  LOG(WARNING) << "DELETE READER!";
  delete reader;

  LOG(WARNING) << "EXIT NOW!";
  return 0;
}
