#pragma once

#include "util/num_type.h"

namespace dso {

class FrameHessian;
class CalibHessian;

class FrameFramePrecalc {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FrameFramePrecalc() { host = target = nullptr; }
  ~FrameFramePrecalc() {}

  void set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib);

 public:
  // static values
  static int instanceCounter;
  FrameHessian* host;    // defines row
  FrameHessian* target;  // defines column

  // precalc values
  // linearized point: Tth_0 = [Rth_0 tth_0]
  // current point: Tth = [Rth tth]

  Mat33f PRE_RTll;     // Rth
  Mat33f PRE_KRKiTll;  // K * Rth * K^{-1}
  Mat33f PRE_RKiTll;   // Rth * K^{-1}
  Mat33f PRE_RTll_0;   // Rth_0

  Vec2f PRE_aff_mode;  // [exp(a_{th}) b_{th}]
  float PRE_b0_mode;   // b_{h}

  Vec3f PRE_tTll;    // tth
  Vec3f PRE_KtTll;   // K * tth
  Vec3f PRE_tTll_0;  // tth_0

  float distanceLL;
};

}  // namespace dso