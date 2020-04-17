#pragma once

#include "util/num_type.h"

namespace dso {

struct Pnt {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  // index in jacobian. never changes (actually, there is no reason why).
  float u, v;

  /** \brief Inverse depth (initial: 1) */
  float idepth;

  /**  \brief True for points with high gradients */
  bool isGood;
  Vec2f energy;  // (UenergyPhotometric, energyRegularizer)
  bool isGood_new;
  float idepth_new;
  Vec2f energy_new;

  /** \brief Smoothed inverse depth */
  float iR;
  float iRSumNum;

  /** \brief Squared derivative of residual wrt. inverse depth */
  float lastHessian;
  float lastHessian_new;

  /** \brief Max stepsize for inverse depth
   *
   *  Corresponding to max. movement in pixel-space
  */
  float maxstep;

  /** \brief Closest point in one pyramid level above
   *
   *  idx = x + y * w
  */
  int parent;
  float parentDist;

  /** \brief Up to 10 nearest points in pixel space
   *
   *  idx = x + y * w
  */
  int neighbours[10];
  float neighboursDist[10];

  float my_type;
  float outlierTH;
};

}  // dso