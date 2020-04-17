#include "util/global_calib.h"

#include <glog/logging.h>

namespace dso {

// widths and heights in all pyramid levels (global variables)
int wG[PYR_LEVELS], hG[PYR_LEVELS];

// intrinsic parameters in all pyramid levels (global variables)
float fxG[PYR_LEVELS], fyG[PYR_LEVELS], cxG[PYR_LEVELS], cyG[PYR_LEVELS];

// inverse intrinsic parameters in all pyramid levels (global variables)
float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS], cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

// Calibration matrix and its inverse in all pyramid levels (global variables)
Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];

// w - 3 (global variable)
float wM3G;

// h - 3 (global variable)
float hM3G;

void SetGlobalCalib(const int w, const int h, const Eigen::Matrix3f& K) {
  int wlvl = w;
  int hlvl = h;
  PYR_LEVELS_USED = 1;
  while (wlvl % 2 == 0 && hlvl % 2 == 0 && wlvl * hlvl > 5000 &&
         PYR_LEVELS_USED < PYR_LEVELS) {
    wlvl /= 2;
    hlvl /= 2;
    ++PYR_LEVELS_USED;
  }
  LOG(INFO) << "using pyramid levels 0 to " << PYR_LEVELS_USED - 1
            << ". coarsest resolution: " << wlvl << " x " << hlvl << "!";
  if (wlvl > 100 && hlvl > 100) {
    LOG(WARNING) << "===============WARNING!===============\n "
                    "Using not enough pyramid levels. Consider scaling to a "
                    "resolution that is a multiple of a power of 2.";
  }
  if (PYR_LEVELS_USED < 3) {
    LOG(WARNING)
        << "===============WARNING!===============\n "
           "I need higher resolution, or I will probably segmentation fault.";
  }

  wM3G = w - 3;
  hM3G = h - 3;

  wG[0] = w;
  hG[0] = h;
  KG[0] = K;
  fxG[0] = K(0, 0);
  fyG[0] = K(1, 1);
  cxG[0] = K(0, 2);
  cyG[0] = K(1, 2);
  KiG[0] = KG[0].inverse();
  fxiG[0] = KiG[0](0, 0);
  fyiG[0] = KiG[0](1, 1);
  cxiG[0] = KiG[0](0, 2);
  cyiG[0] = KiG[0](1, 2);

  for (int level = 1; level < PYR_LEVELS_USED; ++level) {
    wG[level] = w >> level;
    hG[level] = h >> level;

    fxG[level] = fxG[level - 1] * 0.5f;
    fyG[level] = fyG[level - 1] * 0.5f;
    cxG[level] = (cxG[0] + 0.5f) / (1 << level) - 0.5f;
    cyG[level] = (cyG[0] + 0.5f) / (1 << level) - 0.5f;

    // synthetic
    KG[level] << fxG[level], 0.f, cxG[level], 0.f, fyG[level], cyG[level], 0.f,
        0.f, 1.f;
    KiG[level] = KG[level].inverse();

    fxiG[level] = KiG[level](0, 0);
    fyiG[level] = KiG[level](1, 1);
    cxiG[level] = KiG[level](0, 2);
    cyiG[level] = KiG[level](1, 2);
  }
}
}
