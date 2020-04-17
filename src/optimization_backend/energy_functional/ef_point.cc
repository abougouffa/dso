#include "optimization_backend/energy_functional/ef_point.h"

#include "full_system/hessian_blocks/point_hessian.h"
#include "optimization_backend/energy_functional/ef_frame.h"
#include "optimization_backend/energy_functional/ef_residual.h"
#include "util/settings.h"

namespace dso {
void EFPoint::takeData() {
  priorF = data->hasDepthPrior
               ? setting_idepthFixPrior * SCALE_IDEPTH * SCALE_IDEPTH
               : 0;
  if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR) {
    priorF = 0;
  }

  deltaF = data->idepth - data->idepth_zero;
}
}  // dso