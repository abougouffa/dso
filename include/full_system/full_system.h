#pragma once

#include <glog/logging.h>
#include <math.h>

#include <deque>
#include <fstream>
#include <iostream>
#include <vector>

#include "full_system/hessian_blocks/hessian_blocks.h"
#include "full_system/pixel_selector2.h"
#include "full_system/residuals.h"
#include "optimization_backend/energy_functional/energy_functional.h"
#include "util/frame_shell.h"
#include "util/global_calib.h"
#include "util/index_thread_reduce.h"
#include "util/num_type.h"

#define MAX_ACTIVE_FRAMES 100

namespace dso {
namespace IOWrap {
class Output3DWrapper;
}

class PixelSelector;
class PCSyntheticPoint;
class CoarseTracker;
struct FrameHessian;
struct PointHessian;
class CoarseInitializer;
struct ImmaturePointTemporaryResidual;
class ImageAndExposure;
class CoarseDistanceMap;

class EnergyFunctional;

template <typename T>
inline void deleteOut(std::vector<T*>& v, const int i) {
  delete v[i];
  v[i] = v.back();
  v.pop_back();
}

template <typename T>
inline void deleteOutPt(std::vector<T*>& v, const T* i) {
  delete i;

  for (size_t k = 0; k < v.size(); ++k)
    if (v[k] == i) {
      v[k] = v.back();
      v.pop_back();
    }
}

template <typename T>
inline void deleteOutOrder(std::vector<T*>& v, const int i) {
  delete v[i];
  for (unsigned int k = i + 1; k < v.size(); ++k) {
    v[k - 1] = v[k];
  }
  v.pop_back();
}

template <typename T>
inline void deleteOutOrder(std::vector<T*>& v, const T* element) {
  int i = -1;
  for (size_t k = 0; k < v.size(); ++k) {
    if (v[k] == element) {
      i = k;
      break;
    }
  }
  CHECK_GT(i, -1);

  for (size_t k = i + 1; k < v.size(); ++k) {
    v[k - 1] = v[k];
  }
  v.pop_back();

  delete element;
}

inline bool eigenTestNan(const MatXX& m, const std::string& msg) {
  bool foundNan = false;
  for (int y = 0; y < m.rows(); ++y) {
    for (int x = 0; x < m.cols(); ++x) {
      if (!std::isfinite(m(y, x))) {
        foundNan = true;
      }
    }
  }

  LOG_IF(WARNING, foundNan) << "NAN in " << msg << "\n" << m;

  return foundNan;
}

class FullSystem {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FullSystem();
  virtual ~FullSystem();

  /** \brief Interface to add an image.
   *
   *  This function accepts an image and initialize / track.
   *
   *  @param[in] image - image container
   *  @param[in] id    - image id
   */
  void addActiveFrame(ImageAndExposure* image, int id);

  /** \brief Marginalize a frame.
   *
   *  Marginalize a frame. Drop / marginalize points & residuals.
   *
   *  @param[in] frame - frame to marginalize
   */
  void marginalizeFrame(FrameHessian* frame);
  void blockUntilMappingIsFinished();

  float optimize(int mnumOptIts);

  void printResult(std::string file);

  void debugPlot(std::string name);

  void printFrameLifetimes();

  void setGammaFunction(float* const BInv);
  void setOriginalCalib(const VecXf& originalCalib, const int originalW,
                        const int originalH);

 private:
  /** \brief Prerocess a new coming frame */
  FrameHessian* PreprocessNewFrame(ImageAndExposure* const image, const int id);

  /** \brief Optimize a single point */
  int optimizePoint(PointHessian* point, int minObs, bool flagOOB);

  PointHessian* optimizeImmaturePoint(
      ImmaturePoint* const point, const int minObs,
      ImmaturePointTemporaryResidual* const residuals);

  double linAllPointSinle(PointHessian* point, float outlierTHSlack, bool plot);

  // main pipeline functions
  Vec4 trackNewCoarse(FrameHessian* fh);
  void traceNewCoarse(FrameHessian* fh);
  void activatePoints();
  void activatePointsMT();
  void activatePointsOldFirst();

  /** \brief Flag points to drop or marginalize. */
  void flagPointsForRemoval();

  void makeNewTraces(FrameHessian* newFrame, float* gtDepth);
  void initializeFromInitializer(FrameHessian* newFrame);
  void flagFramesForMarginalization(FrameHessian* newFH);

  void removeOutliers();

  // set precalc values.
  void setPrecalcValues();

  // solve. eventually migrate to ef.
  void solveSystem(const int iteration, const double lambda);

  //! Apply step to linearization point.
  bool doStepFromBackup(float stepfacC, float stepfacT, float stepfacR,
                        float stepfacA, float stepfacD);

  /** \brief Set linearization point.
   *
   *  Back up the current state (Tcw, a, b, inverse depth of all points etc.)
   *
   *  @param[in] backupLastStep - not used
   */
  void backupState(const bool backupLastStep);

  /** \brief Set linearization point. */
  void loadSateBackup();

  /** \brief A function always returns 0 for now
   *
   *  Since the global variable setting_forceAceptStep is true by default, this
   *  function always returns 0 for now.
   */
  double calcLEnergy();

  /** \brief A function always returns 0 for now
   *
   *  Since the global variable setting_forceAceptStep is true by default, this
   *  function always returns 0 for now.
   */
  double calcMEnergy();

  /** \brief
   *
   *  @param[in] fixLinearization flag to fix linearization
   *  @return [0]: sum of all active residuals (Note: [1], [2] not used)
   */
  Vec3 linearizeAll(const bool fixLinearization);

  /** TODO
   *
   *  @param[in]  fixLinearization  flag to fix linearization
   *  @param[in]  min               min id of residual to process, usually 0
   *  @param[in]  max               bound of residual to process, usually size()
   *  @param[in]  tid               index of toRemove to use
   *  @param[out] toRemove          container to store residual to remove
   *  @param[out] stats
   */
  void linearizeAll_Reductor(const bool fixLinearization,
                             std::vector<PointFrameResidual*>* const toRemove,
                             const int min, const int max, Vec10* const stats,
                             const int tid);

  void activatePointsMT_Reductor(std::vector<PointHessian*>* optimized,
                                 std::vector<ImmaturePoint*>* toOptimize,
                                 int min, int max, Vec10* stats, int tid);

  /** \brief Update the state and energy of every active residual
   *
   *  @param[in] min           min id of residual to process, usually 0
   *  @param[in] max           bound of residual to process, usually size()
   *  @param[in] copyJacobians flag to decide whether we should update jacobians
   *  @param stats             not used (placeholder)
   *  @param tid               not used (placeholder)
   */
  void applyRes_Reductor(bool copyJacobians, const int min, const int max,
                         Vec10* stats, int tid);

  void printOptRes(const Vec3& res, double resL, double resM, double resPrior,
                   double LExact, float a, float b);

  void debugPlotTracking();

  std::vector<VecX> getNullspaces(std::vector<VecX>& nullspaces_pose,
                                  std::vector<VecX>& nullspaces_scale,
                                  std::vector<VecX>& nullspaces_affA,
                                  std::vector<VecX>& nullspaces_affB);

  /** Set new threshold for residual of points in the newest frame. */
  void setNewFrameEnergyTH();

  void printLogLine();
  void printEvalLine();
  void printEigenValLine();

  // tracking always uses the newest KF as reference.
  void makeKeyFrame(FrameHessian* const fh);
  void makeNonKeyFrame(FrameHessian* const fh);
  void deliverTrackedFrame(FrameHessian* const fh, const bool needKF);
  void mappingLoop();

 public:
  std::vector<IOWrap::Output3DWrapper*> outputWrapper;

  bool isLost;
  bool initFailed;
  bool initialized;
  bool linearizeOperation;

 private:
  CalibHessian Hcalib;

  std::ofstream* calibLog;
  std::ofstream* numsLog;
  std::ofstream* errorsLog;
  std::ofstream* eigenAllLog;
  std::ofstream* eigenPLog;
  std::ofstream* eigenALog;
  std::ofstream* DiagonalLog;
  std::ofstream* variancesLog;
  std::ofstream* nullspacesLog;

  std::ofstream* coarseTrackingLog;

  // statistics
  long int statistics_lastNumOptIts;
  long int statistics_numDroppedPoints;
  long int statistics_numActivatedPoints;
  long int statistics_numCreatedPoints;
  long int statistics_numForceDroppedResBwd;
  long int statistics_numForceDroppedResFwd;
  long int statistics_numMargResFwd;
  long int statistics_numMargResBwd;
  float statistics_lastFineTrackRMSE;

  // ========== changed by tracker-thread. protected by trackMutex ==========
  boost::mutex trackMutex;
  std::vector<FrameShell*> allFrameHistory;
  CoarseInitializer* coarseInitializer;
  Vec5 lastCoarseRMSE;

  // ============ changed by mapper-thread. protected by mapMutex ============
  boost::mutex mapMutex;
  std::vector<FrameShell*> allKeyFramesHistory;

  EnergyFunctional* ef;
  IndexThreadReduce<Vec10> treadReduce;

  float* selectionMap;
  PixelSelector* pixelSelector;
  CoarseDistanceMap* coarseDistanceMap;

  // ONLY changed in marginalizeFrame and addFrame.
  std::vector<FrameHessian*> frameHessians;

  /** \brief Active residuals for optimization
   *
   *  Residuals of those still not linearized points
   */
  std::vector<PointFrameResidual*> activeResiduals;

  float currentMinActDist;

  std::vector<float> allResVec;

  // mutex etc. for tracker exchange.

  // if tracker sees that there is a new reference, tracker locks
  // [coarseTrackerSwapMutex] and swaps the two.
  boost::mutex coarseTrackerSwapMutex;

  // set as as reference. protected by [coarseTrackerSwapMutex].
  CoarseTracker* coarseTracker_forNewKF;

  // always used to track new frames. protected by [trackMutex].
  CoarseTracker* coarseTracker;

  float minIdJetVisTracker, maxIdJetVisTracker;
  float minIdJetVisDebug, maxIdJetVisDebug;

  // mutex for camToWorl's in shells (these are always in a good configuration).
  boost::mutex shellPoseMutex;

  // tracking / mapping synchronization. All protected by [trackMapSyncMutex].
  boost::mutex trackMapSyncMutex;
  boost::condition_variable trackedFrameSignal;
  boost::condition_variable mappedFrameSignal;
  std::deque<FrameHessian*> unmappedTrackedFrames;

  // Otherwise, a new KF is *needed that has ID bigger than [needNewKFAfter]*.
  int needNewKFAfter;

  boost::thread mappingThread;
  bool runMapping;
  bool needToKetchupMapping;

  int lastRefStopID;
  SE3 T_c0w;
};
}  // namespace dso
