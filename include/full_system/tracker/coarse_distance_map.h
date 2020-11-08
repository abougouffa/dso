#pragma once

#include "util/num_type.h"
#include "util/settings.h"

namespace dso {

class CalibHessian;
class FrameHessian;
class PointFrameResidual;

class CoarseDistanceMap {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CoarseDistanceMap(int w, int h);
  ~CoarseDistanceMap();

  void makeDistanceMap(std::vector<FrameHessian*> frameHessians,
                       FrameHessian* frame);

  void makeInlierVotes(std::vector<FrameHessian*> frameHessians);

  void makeK(CalibHessian* HCalib);

  void addIntoDistFinal(int u, int v);

 private:
  void growDistBFS(int bfsNum);

 public:
  float* fwdWarpedIDDistFinal;

  Mat33f K[PYR_LEVELS];
  Mat33f Ki[PYR_LEVELS];
  float fx[PYR_LEVELS];
  float fy[PYR_LEVELS];
  float fxi[PYR_LEVELS];
  float fyi[PYR_LEVELS];
  float cx[PYR_LEVELS];
  float cy[PYR_LEVELS];
  float cxi[PYR_LEVELS];
  float cyi[PYR_LEVELS];
  int w[PYR_LEVELS];
  int h[PYR_LEVELS];

 private:
  PointFrameResidual** coarseProjectionGrid;
  int* coarseProjectionGridNum;
  Eigen::Vector2i* bfsList1;
  Eigen::Vector2i* bfsList2;
};

}  // namespace dso