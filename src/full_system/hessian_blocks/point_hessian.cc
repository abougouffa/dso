#include "full_system/hessian_blocks/point_hessian.h"

#include <glog/logging.h>

#include "full_system/immature_point.h"

namespace dso {

PointHessian::PointHessian(const ImmaturePoint* const rawPoint,
                           CalibHessian* Hcalib) {
  ++instanceCounter;
  host = rawPoint->host;
  hasDepthPrior = false;

  idepth_hessian = 0;
  maxRelBaseline = 0;
  numGoodResiduals = 0;

  // set static values & initialization.
  u = rawPoint->u;
  v = rawPoint->v;
  CHECK(std::isfinite(rawPoint->idepth_max));
  // idepth_init = rawPoint->idepth_GT;

  my_type = rawPoint->my_type;

  setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min) * 0.5);
  setPointStatus(PointHessian::INACTIVE);

  int n = patternNum;
  memcpy(color, rawPoint->color, sizeof(float) * n);
  memcpy(weights, rawPoint->weights, sizeof(float) * n);
  energyTH = rawPoint->energyTH;

  efPoint = 0;
}

bool PointHessian::isOOB(const std::vector<FrameHessian*>& toKeep,
                         const std::vector<FrameHessian*>& toMarg) const {
  int visInToMarg = 0;
  for (PointFrameResidual* r : residuals) {
    if (r->state_state != ResState::IN) {
      continue;
    }
    for (FrameHessian* k : toMarg) {
      if (r->target == k) {
        ++visInToMarg;
      }
    }
  }
  if (static_cast<int>(residuals.size()) >= setting_minGoodActiveResForMarg &&
      numGoodResiduals > setting_minGoodResForMarg + 10 &&
      static_cast<int>(residuals.size()) - visInToMarg <
          setting_minGoodActiveResForMarg) {
    return true;
  }

  if (lastResiduals[0].second == ResState::OOB) {
    return true;
  }
  if (residuals.size() < 2) {
    return false;
  }
  if (lastResiduals[0].second == ResState::OUTLIER &&
      lastResiduals[1].second == ResState::OUTLIER) {
    return true;
  }
  return false;
}

}  // dso