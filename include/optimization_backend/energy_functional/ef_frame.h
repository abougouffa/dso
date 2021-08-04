#pragma once

#include "util/num_type.h"

namespace dso {

class FrameHessian;
class EFPoint;

class EFFrame {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EFFrame(FrameHessian* d) : data(d) { takeData(); }
  void takeData();

 public:
  Vec8 prior;        // prior hessian (diagonal)
  Vec8 delta_prior;  // = state-state_prior (E_prior = (delta_prior)' *
                     // diag(prior) * (delta_prior)
  Vec8 delta;        // state - state_zero.

  std::vector<EFPoint*> points;
  FrameHessian* data;
  int idx;  // idx in frames.

  int frameID;
};
}  // dso