#pragma once

#include "util/num_type.h"

namespace dso {

class EFFrame;
class EFResidual;
class PointHessian;

enum EFPointStatus { PS_GOOD = 0, PS_MARGINALIZE, PS_DROP };

class EFPoint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EFPoint(PointHessian* d, EFFrame* host_) : data(d), host(host_) {
    takeData();
    stateFlag = EFPointStatus::PS_GOOD;
  }
  void takeData();

 public:
  PointHessian* data;

  float priorF;
  float deltaF;

  // constant info (never changes in-between).
  int idxInPoints;
  EFFrame* host;

  // contains all residuals.
  std::vector<EFResidual*> residualsAll;

  float bdSumF;
  float HdiF;
  float Hdd_accLF;
  VecCf Hcd_accLF;
  float bd_accLF;
  float Hdd_accAF;
  VecCf Hcd_accAF;
  float bd_accAF;

  EFPointStatus stateFlag;
};

}  // dso