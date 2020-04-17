#pragma once

#include <vector>

#include "util/minimal_image.h"
#include "util/num_type.h"

namespace dso {

namespace IOWrap {

void displayImage(const char* windowName, const MinimalImageB* img,
                  bool autoSize = false);
void displayImage(const char* windowName, const MinimalImageB3* img,
                  bool autoSize = false);
void displayImage(const char* windowName, const MinimalImageF* img,
                  bool autoSize = false);
void displayImage(const char* windowName, const MinimalImageF3* img,
                  bool autoSize = false);
void displayImage(const char* windowName, const MinimalImageB16* img,
                  bool autoSize = false);

void displayImageStitch(const char* windowName,
                        const std::vector<MinimalImageB*> images, int cc = 0,
                        int rc = 0);
void displayImageStitch(const char* windowName,
                        const std::vector<MinimalImageB3*> images, int cc = 0,
                        int rc = 0);
void displayImageStitch(const char* windowName,
                        const std::vector<MinimalImageF*> images, int cc = 0,
                        int rc = 0);
void displayImageStitch(const char* windowName,
                        const std::vector<MinimalImageF3*> images, int cc = 0,
                        int rc = 0);

int waitKey(int milliseconds);
void closeAllWindows();
}
}
