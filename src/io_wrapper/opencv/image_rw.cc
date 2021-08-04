#include "io_wrapper/image_rw.h"

#include <glog/logging.h>
#include <opencv2/highgui.hpp>

namespace dso {

namespace IOWrap {
MinimalImageB* readImageBW_8U(std::string filename) {
  cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  if (m.rows * m.cols == 0) {
    LOG(ERROR) << "cv::imread could not read image " << filename
               << "! This may segmentation fault.";
    return 0;
  }
  if (m.type() != CV_8U) {
    LOG(ERROR)
        << "cv::imread did something strange! This may segmentation fault.";
    return 0;
  }
  MinimalImageB* img = new MinimalImageB(m.cols, m.rows);
  memcpy(img->data, m.data, m.rows * m.cols);
  return img;
}

MinimalImageB3* readImageRGB_8U(std::string filename) {
  cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  if (m.rows * m.cols == 0) {
    LOG(ERROR) << "cv::imread could not read image " << filename
               << "! This may segmentation fault.";
    return 0;
  }
  if (m.type() != CV_8UC3) {
    LOG(ERROR)
        << "cv::imread did something strange! This may segmentation fault.";
    return 0;
  }
  MinimalImageB3* img = new MinimalImageB3(m.cols, m.rows);
  memcpy(img->data, m.data, 3 * m.rows * m.cols);
  return img;
}

MinimalImage<unsigned short>* readImageBW_16U(std::string filename) {
  cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
  if (m.rows * m.cols == 0) {
    LOG(ERROR) << "cv::imread could not read image " << filename
               << "! This may segmentation fault.";
    return 0;
  }
  if (m.type() != CV_16U) {
    LOG(ERROR) << "readImageBW_16U called on image that is not a 16bit "
                  "grayscale image. This may segmentation fault.";
    return 0;
  }
  MinimalImage<unsigned short>* img =
      new MinimalImage<unsigned short>(m.cols, m.rows);
  memcpy(img->data, m.data, 2 * m.rows * m.cols);
  return img;
}

MinimalImageB* readStreamBW_8U(char* data, int numBytes) {
  cv::Mat m =
      cv::imdecode(cv::Mat(numBytes, 1, CV_8U, data), CV_LOAD_IMAGE_GRAYSCALE);
  if (m.rows * m.cols == 0) {
    if (m.rows * m.cols == 0) {
      LOG(ERROR) << "cv::imdecode could not read stream (" << numBytes
                 << " bytes)! This may segmentation fault.";
      return 0;
    }
  }
  if (m.type() != CV_8U) {
    LOG(ERROR)
        << "cv::imdecode did something strange! This may segmentation fault.";
    return 0;
  }
  MinimalImageB* img = new MinimalImageB(m.cols, m.rows);
  memcpy(img->data, m.data, m.rows * m.cols);
  return img;
}

void writeImage(std::string filename, MinimalImageB* img) {
  cv::imwrite(filename, cv::Mat(img->h, img->w, CV_8U, img->data));
}
void writeImage(std::string filename, MinimalImageB3* img) {
  cv::imwrite(filename, cv::Mat(img->h, img->w, CV_8UC3, img->data));
}
void writeImage(std::string filename, MinimalImageF* img) {
  cv::imwrite(filename, cv::Mat(img->h, img->w, CV_32F, img->data));
}
void writeImage(std::string filename, MinimalImageF3* img) {
  cv::imwrite(filename, cv::Mat(img->h, img->w, CV_32FC3, img->data));
}
}
}
