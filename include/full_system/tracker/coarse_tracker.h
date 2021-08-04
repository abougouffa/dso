#pragma once

#include <math.h>
#include <vector>

#include "full_system/tracker/coarse_distance_map.h"
#include "io_wrapper/output_3d_wrapper.h"
#include "optimization_backend/accumulators/matrix_accumulators.h"
#include "util/num_type.h"
#include "util/settings.h"

namespace dso {
class CalibHessian;
class FrameHessian;

class CoarseTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CoarseTracker(int w, int h);
  ~CoarseTracker();

  bool trackNewestCoarse(FrameHessian* newFrameHessian, SE3& lastToNew_out,
                         AffLight& aff_g2l_out, int coarsestLvl,
                         Vec5 minResForAbort,
                         IOWrap::Output3DWrapper* wrap = nullptr);

  void setCoarseTrackingRef(std::vector<FrameHessian*> frameHessians);

  void makeK(CalibHessian* HCalib);

  void debugPlotIDepthMap(float* minID, float* maxID,
                          std::vector<IOWrap::Output3DWrapper*>& wraps);
  void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*>& wraps);

  bool debugPrint, debugPlot;

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

  FrameHessian* lastRef;
  AffLight lastRef_aff_g2l;
  FrameHessian* newFrame;
  int refFrameID;

  // act as pure ouptut
  Vec5 lastResiduals;
  Vec3 lastFlowIndicators;
  double firstCoarseRMSE;

 private:
  void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians);
  float* idepth[PYR_LEVELS];
  float* weightSums[PYR_LEVELS];
  float* weightSums_bak[PYR_LEVELS];

  Vec6 calcResAndGS(int lvl, Mat88& H_out, Vec8& b_out, const SE3& refToNew,
                    AffLight aff_g2l, float cutoffTH);
  Vec6 calcRes(int lvl, const SE3& refToNew, AffLight aff_g2l, float cutoffTH);
  void calcGSSSE(int lvl, Mat88& H_out, Vec8& b_out, const SE3& refToNew,
                 AffLight aff_g2l);
  void calcGS(int lvl, Mat88& H_out, Vec8& b_out, const SE3& refToNew,
              AffLight aff_g2l);

 private:
  // pc buffers
  float* pc_u[PYR_LEVELS];
  float* pc_v[PYR_LEVELS];
  float* pc_idepth[PYR_LEVELS];
  float* pc_color[PYR_LEVELS];
  int pc_n[PYR_LEVELS];

  // warped buffers
  float* buf_warped_idepth;
  float* buf_warped_u;
  float* buf_warped_v;
  float* buf_warped_dx;
  float* buf_warped_dy;
  float* buf_warped_residual;
  float* buf_warped_weight;
  float* buf_warped_refColor;
  int buf_warped_n;

  std::vector<float*> ptrToDelete;

  Accumulator9 acc;
};
}
