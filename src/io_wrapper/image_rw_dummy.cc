#include "io_wrapper/image_rw.h"

#include <glog/logging.h>

namespace dso {

namespace IOWrap {

MinimalImageB* readImageBW_8U(std::string filename) {
  LOG(WARNING) << "Not implemented. Bye!";
  return nullptr;
};
MinimalImageB3* readImageRGB_8U(std::string filename) {
  LOG(WARNING) << "Not implemented. Bye!";
  return nullptr;
};
MinimalImage<unsigned short>* readImageBW_16U(std::string filename) {
  LOG(WARNING) << "Not implemented. Bye!";
  return nullptr;
};
MinimalImageB* readStreamBW_8U(char* data, int numBytes) {
  LOG(WARNING) << "Not implemented. Bye!";
  return nullptr;
};
void writeImage(std::string filename, MinimalImageB* img){};
void writeImage(std::string filename, MinimalImageB3* img){};
void writeImage(std::string filename, MinimalImageF* img){};
void writeImage(std::string filename, MinimalImageF3* img){};
}
}
