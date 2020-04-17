#pragma once

#include <math.h>
#include <map>
#include <vector>

#include "util/index_thread_reduce.h"
#include "util/num_type.h"

namespace dso {

class PointFrameResidual;
class CalibHessian;
class FrameHessian;
class PointHessian;

class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;
class AccumulatedTopHessian;
class AccumulatedTopHessianSSE;
class AccumulatedSCHessian;
class AccumulatedSCHessianSSE;

extern bool EFAdjointsValid;
extern bool EFIndicesValid;
extern bool EFDeltaValid;

class EnergyFunctional {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  friend class EFFrame;
  friend class EFPoint;
  friend class EFResidual;
  friend class AccumulatedTopHessian;
  friend class AccumulatedTopHessianSSE;
  friend class AccumulatedSCHessian;
  friend class AccumulatedSCHessianSSE;

  EnergyFunctional();
  ~EnergyFunctional();

  EFResidual* insertResidual(PointFrameResidual* r);
  EFFrame* insertFrame(FrameHessian* fh, CalibHessian* Hcalib);
  EFPoint* insertPoint(PointHessian* ph);

  void dropResidual(EFResidual* r);

  /** \brief Marginalize a frame using Schur complement.
   *
   *
   *  @param[in] fh - frame to marginalize
  */
  void marginalizeFrame(EFFrame* fh);
  void removePoint(EFPoint* ph);

  /** \brief Marginalize a frame using Schur complement.
   *
   *  1. Compute contribution of marginalized points to the Hessian.
   *  2. Drop residuals of all marginalized points
  */
  void marginalizePointsF();
  void dropPointsF();

  /** \brief TODO
   *
   *  @param[in] iteration - times of optimization iteration
   *  @param[in] lambda    - coefficient used in LM
   *  @param[in] HCalib    - intrinsic parameters
  */
  void solveSystemF(const int iteration, double lambda,
                    CalibHessian* const HCalib);

  /** \brief Uncalled function
   *
   *  This function is used in FullSystem::calcMEnergy(), but it will never be
   *  called for now due to the global variable setting_forceAceptStep
  */
  double calcMEnergyF();

  /** \brief Uncalled function
   *
   *  This function is used in FullSystem::calcLEnergy(), but it will never be
   *  called for now due to the global variable setting_forceAceptStep
  */
  double calcLEnergyF_MT();

  void makeIDX();

  void setDeltaF(CalibHessian* HCalib);

  void setAdjointsF(CalibHessian* Hcalib);

 public:
  std::vector<EFFrame*> frames;

  //! Number of points in energy function
  int nPoints;

  //! Number of frames in energy function
  int nFrames;

  //! Number of residuals in energy function
  int nResiduals;

  //! Marginalized Hessian
  MatXX HM;

  //! Marginalized b
  VecX bM;

  int resInA, resInL, resInM;
  MatXX lastHS;
  VecX lastbS;
  VecX lastX;
  std::vector<VecX> lastNullspaces_forLogging;
  std::vector<VecX> lastNullspaces_pose;
  std::vector<VecX> lastNullspaces_scale;
  std::vector<VecX> lastNullspaces_affA;
  std::vector<VecX> lastNullspaces_affB;

  IndexThreadReduce<Vec10>* red;

  std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>,
           Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>>
      connectivityMap;

 private:
  VecX getStitchedDeltaF() const;

  void resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT);
  void resubstituteFPt(const VecCf& xc, Mat18f* xAd, int min, int max,
                       Vec10* stats, int tid);

  /** \brief Accumulate active residuals
   *
   *  @param[in]  MT - whether use multi threads
   *  @param[out] H  -
   *  @param[out] b  -
  */
  void accumulateAF_MT(MatXX& H, VecX& b, const bool MT);

  /** \brief Accumulate linearized residuals
   *
   *  @param[in]  MT - whether use multi threads
   *  @param[out] H  -
   *  @param[out] b  -
  */
  void accumulateLF_MT(MatXX& H, VecX& b, const bool MT);

  /** \brief Accumulate residuals related to schur complement
   *
   *  @param[in]  MT - whether use multi threads
   *  @param[out] H  -
   *  @param[out] b  -
  */
  void accumulateSCF_MT(MatXX& H, VecX& b, const bool MT);

  void calcLEnergyPt(int min, int max, Vec10* stats, int tid);

  void orthogonalize(VecX* b, MatXX* H);

 private:
  Mat18f* adHTdeltaF;

  Mat88* adHost;
  Mat88* adTarget;

  Mat88f* adHostF;
  Mat88f* adTargetF;

  VecC cPrior;
  VecCf cDeltaF;
  VecCf cPriorF;

  AccumulatedTopHessianSSE* accSSE_top_L;
  AccumulatedTopHessianSSE* accSSE_top_A;

  AccumulatedSCHessianSSE* accSSE_bot;

  std::vector<EFPoint*> allPoints;
  std::vector<EFPoint*> allPointsToMarg;

  float currentLambda;
};

}  // dso
