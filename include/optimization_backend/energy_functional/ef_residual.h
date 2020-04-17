#pragma once

#include <glog/logging.h>

#include "optimization_backend/raw_residual_jacobian.h"
#include "util/num_type.h"

namespace dso {

class PointFrameResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;

class EFResidual {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  inline EFResidual(PointFrameResidual* org, EFPoint* point_, EFFrame* host_,
                    EFFrame* target_)
      : data(org), point(point_), host(host_), target(target_) {
    isLinearized = false;
    isActiveAndIsGoodNEW = false;
    J = new RawResidualJacobian();
    CHECK(((long)this) % 16 == 0);
    CHECK(((long)J) % 16 == 0);
  }

  inline ~EFResidual() { delete J; }

  //! Use the newest Jacobians to update JpJdF
  void takeDataF();

  void fixLinearizationF(EnergyFunctional* ef);

  inline const bool& isActive() const { return isActiveAndIsGoodNEW; }

 public:
  //! structural pointers
  PointFrameResidual* data;
  int hostIDX, targetIDX;
  EFPoint* point;
  EFFrame* host;
  EFFrame* target;
  int idxInAll;

  RawResidualJacobian* J;

  VecNRf res_toZeroF;

  //! Intermedian variable for Hessian matrix \f$\mathbf{H}_{12}\f$
  Vec8f JpJdF;

  //! status.
  bool isLinearized;

  //! if residual is not OOB & not OUTLIER & should be used during accumulations
  bool isActiveAndIsGoodNEW;
};

}  // dso