#include "full_system/hessian_blocks/frame_frame_pre_calc.h"

#include "full_system/hessian_blocks/calib_hessian.h"
#include "full_system/hessian_blocks/frame_hessian.h"

namespace dso {

void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target,
                            CalibHessian* HCalib) {
  this->host = host;
  this->target = target;

  // Tth: from host to target (linearized point x0)
  SE3 leftToLeft_0 =
      target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

  // Rth: Rotation from host to target (linearized point x0)
  PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();

  // tth: Translation from host to target (linearized point x0)
  PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();

  // Current Tth
  SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
  PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
  PRE_tTll = (leftToLeft.translation()).cast<float>();
  distanceLL = leftToLeft.translation().norm();  // length of baseline

  Mat33f K = Mat33f::Zero();
  K(0, 0) = HCalib->fxl();
  K(1, 1) = HCalib->fyl();
  K(0, 2) = HCalib->cxl();
  K(1, 2) = HCalib->cyl();
  K(2, 2) = 1;
  PRE_KRKiTll = K * PRE_RTll * K.inverse();
  PRE_RKiTll = PRE_RTll * K.inverse();
  PRE_KtTll = K * PRE_tTll;

  PRE_aff_mode =
      AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure,
                                  host->aff_g2l(), target->aff_g2l())
          .cast<float>();
  PRE_b0_mode = host->aff_g2l_0().b;
}

}  // dso