#pragma once

#include "full_system/hessian_blocks/hessian_blocks.h"
#include "full_system/residuals.h"
#include "util/num_type.h"

namespace dso {

struct ImmaturePointTemporaryResidual {
 public:
  ResState state_state;
  double state_energy;
  ResState state_NewState;
  double state_NewEnergy;
  FrameHessian* target;
};

enum ImmaturePointStatus {
  IPS_GOOD = 0,      // traced well and good
  IPS_OOB,           // OOB: end tracking & marginalize!
  IPS_OUTLIER,       // energy too high: if happens again: outlier!
  IPS_SKIPPED,       // traced well and good (but not actually traced).
  IPS_BADCONDITION,  // not traced because of bad condition.
  IPS_UNINITIALIZED
};  // not even traced once.

class ImmaturePoint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImmaturePoint(int u_, int v_, FrameHessian* host_, float type,
                CalibHessian* HCalib);
  ~ImmaturePoint();

  ImmaturePointStatus traceOn(FrameHessian* frame,
                              const Mat33f& hostToFrame_KRKi,
                              const Vec3f& hostToFrame_Kt,
                              const Vec2f& hostToFrame_affine,
                              CalibHessian* HCalib, bool debugPrint = false);

  double linearizeResidual(CalibHessian* HCalib, const float outlierTHSlack,
                           ImmaturePointTemporaryResidual* tmpRes, float& Hdd,
                           float& bd, float idepth);
  float getdPixdd(CalibHessian* HCalib, ImmaturePointTemporaryResidual* tmpRes,
                  float idepth);

  float calcResidual(CalibHessian* HCalib, const float outlierTHSlack,
                     ImmaturePointTemporaryResidual* tmpRes, float idepth);

 public:
  float color[MAX_RES_PER_POINT];
  float weights[MAX_RES_PER_POINT];

  Mat22f gradH;  // 一个pattern累加的[gx * gx, gx * gy; gy * gx, gy * gy]
  Vec2f gradH_ev;
  Mat22f gradH_eig;
  float energyTH;
  float u, v;
  FrameHessian* host;
  int idxInImmaturePoints;

  float quality;

  float my_type;

  float idepth_min;
  float idepth_max;

  ImmaturePointStatus lastTraceStatus;
  Vec2f lastTraceUV;
  float lastTracePixelInterval;

  float idepth_GT;
};
}  // namespace dso
