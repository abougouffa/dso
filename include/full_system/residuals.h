#pragma once

#include <fstream>
#include <iostream>
#include <vector>

#include "optimization_backend/raw_residual_jacobian.h"
#include "util/global_calib.h"
#include "util/global_funcs.h"
#include "util/num_type.h"

namespace dso {
class PointHessian;
class FrameHessian;
class CalibHessian;

class EFResidual;

enum ResLocation { ACTIVE = 0, LINEARIZED, MARGINALIZED, NONE };
enum ResState { IN = 0, OOB, OUTLIER };

struct FullJacRowT {
  Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
};

class PointFrameResidual {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ~PointFrameResidual();
  PointFrameResidual();
  PointFrameResidual(PointHessian* point_, FrameHessian* host_,
                     FrameHessian* target_);

  Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
  Vec3f centerProjectedTo;

  //! Compute the residual of the point.
  /*!
    This function computes the total residual of the whole pattern centered on
    the pixel. If the residual is too large, the point will be seen as an
    outlier.

    @param[in] HCalib intrinsic paramters
    @return residual of this point (including the whole pattern)
  */
  double linearize(CalibHessian* const HCalib);

  void resetOOB() {
    state_NewEnergy = state_energy = 0;
    state_NewState = ResState::OUTLIER;

    setState(ResState::IN);
  };

  //! Update state and energy of this point
  /*!
    @param[in] copyJacobians flag to decide whether we should update jacobians
    and related intermediate variables for Hessian
  */
  void applyRes(const bool copyJacobians);

  void debugPlot();

  void printRows(std::vector<VecX>& v, VecX& r, int nFrames, int nPoints, int M,
                 int res);

 public:
  static int instanceCounter;

  EFResidual* efResidual;

  ResState state_state;
  double state_energy;
  ResState state_NewState;

  //! residual of this point (truncated to threshold if too large)
  double state_NewEnergy;

  //! original computed residual of this point
  double state_NewEnergyWithOutlier;

  void setState(ResState s) { state_state = s; }

  PointHessian* point;
  FrameHessian* host;
  FrameHessian* target;
  RawResidualJacobian* J;

  bool isNew;
};
}
