#pragma once

#include <glog/logging.h>

#include "full_system/residuals.h"
#include "util/num_type.h"
#include "util/settings.h"

#define SCALE_IDEPTH 1.0f  // scales internal value to idepth.
#define SCALE_IDEPTH_INVERSE (1.0f / SCALE_IDEPTH)

namespace dso {

class FrameHessian;
class EFPoint;
class ImmaturePoint;

/** \brief Hessian component associated with one point. */
class PointHessian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum PtStatus { ACTIVE = 0, INACTIVE, OUTLIER, OOB, MARGINALIZED };

  PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib);

  ~PointHessian() {
    CHECK(efPoint == nullptr);
    release();
    --instanceCounter;
  }

  void release() {
    for (size_t i = 0; i < residuals.size(); ++i) {
      delete residuals[i];
    }
    residuals.clear();
  }

  void setPointStatus(PtStatus s) { status = s; }

  void setIdepth(float idepth) {
    this->idepth = idepth;
    this->idepth_scaled = SCALE_IDEPTH * idepth;
  }

  void setIdepthScaled(float idepth_scaled) {
    this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
    this->idepth_scaled = idepth_scaled;
  }

  void setIdepthZero(float idepth) {
    idepth_zero = idepth;
    idepth_zero_scaled = SCALE_IDEPTH * idepth;
    nullspaces_scale = -(idepth * 1.001 - idepth / 1.001) * 500;
  }

  bool isInlierNew() {
    return static_cast<int>(residuals.size()) >=
               setting_minGoodActiveResForMarg &&
           numGoodResiduals >= setting_minGoodResForMarg;
  }

  bool isOOB(const std::vector<FrameHessian*>& toKeep,
             const std::vector<FrameHessian*>& toMarg) const;

 public:
  static int instanceCounter;
  EFPoint* efPoint;

  /** \brief Colors in host frame */
  float color[MAX_RES_PER_POINT];

  /** \brief Host-weights for respective residuals. */
  float weights[MAX_RES_PER_POINT];

  float u, v;
  int idx;
  float energyTH;
  FrameHessian* host;
  bool hasDepthPrior;

  float my_type;

  float idepth_scaled;
  float idepth_zero_scaled;
  float idepth_zero;
  float idepth;
  float step;
  float step_backup;
  float idepth_backup;

  float nullspaces_scale;
  float idepth_hessian;
  float maxRelBaseline;
  int numGoodResiduals;

  PtStatus status;

  /** \brief Container of good residuals (NO OOB and NO OUTLIER) */
  std::vector<PointFrameResidual*> residuals;

  /** \brief Information about residuals in the last two frames.
   *
   *  0: latest.
   *  1: the one before
   */
  std::pair<PointFrameResidual*, ResState> lastResiduals[2];
};

}  // namespace dso