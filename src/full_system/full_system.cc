#include "full_system/full_system.h"

#include <glog/logging.h>
#include <stdio.h>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

#include "full_system/immature_point.h"
#include "full_system/initializer/coarse_initializer.h"
#include "full_system/pixel_selector.h"
#include "full_system/pixel_selector2.h"
#include "full_system/residual_projections.h"
#include "full_system/tracker/coarse_tracker.h"
#include "io_wrapper/image_display.h"
#include "io_wrapper/output_3d_wrapper.h"
#include "optimization_backend/energy_functional/energy_functional.h"
#include "optimization_backend/energy_functional/energy_functional_structs.h"
#include "util/global_calib.h"
#include "util/global_funcs.h"
#include "util/image_and_exposure.h"

namespace dso {
int FrameHessian::instanceCounter = 0;
int PointHessian::instanceCounter = 0;
int CalibHessian::instanceCounter = 0;

FullSystem::FullSystem() {
  int retstat = 0;
  if (setting_logStuff) {
    retstat += system("rm -rf logs");
    retstat += system("mkdir logs");

    retstat += system("rm -rf mats");
    retstat += system("mkdir mats");

    calibLog = new std::ofstream();
    calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
    calibLog->precision(12);

    numsLog = new std::ofstream();
    numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
    numsLog->precision(10);

    coarseTrackingLog = new std::ofstream();
    coarseTrackingLog->open("logs/coarseTrackingLog.txt",
                            std::ios::trunc | std::ios::out);
    coarseTrackingLog->precision(10);

    eigenAllLog = new std::ofstream();
    eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
    eigenAllLog->precision(10);

    eigenPLog = new std::ofstream();
    eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
    eigenPLog->precision(10);

    eigenALog = new std::ofstream();
    eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
    eigenALog->precision(10);

    DiagonalLog = new std::ofstream();
    DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
    DiagonalLog->precision(10);

    variancesLog = new std::ofstream();
    variancesLog->open("logs/variancesLog.txt",
                       std::ios::trunc | std::ios::out);
    variancesLog->precision(10);

    nullspacesLog = new std::ofstream();
    nullspacesLog->open("logs/nullspacesLog.txt",
                        std::ios::trunc | std::ios::out);
    nullspacesLog->precision(10);
  } else {
    nullspacesLog = 0;
    variancesLog = 0;
    DiagonalLog = 0;
    eigenALog = 0;
    eigenPLog = 0;
    eigenAllLog = 0;
    numsLog = 0;
    calibLog = 0;
  }

  CHECK_NE(retstat, 293847);

  selectionMap = new float[wG[0] * hG[0]];

  coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
  coarseTracker = new CoarseTracker(wG[0], hG[0]);
  coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
  coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
  pixelSelector = new PixelSelector(wG[0], hG[0]);

  statistics_lastNumOptIts = 0;
  statistics_numDroppedPoints = 0;
  statistics_numActivatedPoints = 0;
  statistics_numCreatedPoints = 0;
  statistics_numForceDroppedResBwd = 0;
  statistics_numForceDroppedResFwd = 0;
  statistics_numMargResFwd = 0;
  statistics_numMargResBwd = 0;

  lastCoarseRMSE.setConstant(100);

  currentMinActDist = 2;
  initialized = false;

  ef = new EnergyFunctional();
  ef->red = &this->treadReduce;

  isLost = false;
  initFailed = false;

  needNewKFAfter = -1;

  linearizeOperation = true;
  runMapping = true;
  mappingThread = boost::thread(&FullSystem::mappingLoop, this);
  lastRefStopID = 0;

  minIdJetVisDebug = -1;
  maxIdJetVisDebug = -1;
  minIdJetVisTracker = -1;
  maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem() {
  blockUntilMappingIsFinished();

  if (setting_logStuff) {
    calibLog->close();
    delete calibLog;
    numsLog->close();
    delete numsLog;
    coarseTrackingLog->close();
    delete coarseTrackingLog;
    eigenAllLog->close();
    delete eigenAllLog;
    eigenPLog->close();
    delete eigenPLog;
    eigenALog->close();
    delete eigenALog;
    DiagonalLog->close();
    delete DiagonalLog;
    variancesLog->close();
    delete variancesLog;
    nullspacesLog->close();
    delete nullspacesLog;
  }

  delete[] selectionMap;

  for (FrameShell *s : allFrameHistory) {
    delete s;
  }
  for (FrameHessian *fh : unmappedTrackedFrames) {
    delete fh;
  }

  delete coarseDistanceMap;
  delete coarseTracker;
  delete coarseTracker_forNewKF;
  delete coarseInitializer;
  delete pixelSelector;
  delete ef;
}

void FullSystem::setGammaFunction(float *const BInv) {
  if (BInv == nullptr) {
    return;
  }

  // copy BInv.
  memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);

  // invert.
  for (int i = 1; i < 255; ++i) {
    // find val, such that Binv[val] = i.
    // I dont care about speed for this, so do it the stupid way.

    for (int s = 1; s < 255; ++s) {
      if (BInv[s] <= i && BInv[s + 1] >= i) {
        Hcalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
        break;
      }
    }
  }
  Hcalib.B[0] = 0;
  Hcalib.B[255] = 255;
}

void FullSystem::printResult(std::string file) {
  boost::unique_lock<boost::mutex> lock(trackMutex);
  boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

  std::ofstream myfile;
  myfile.open(file.c_str());
  myfile << std::setprecision(15);

  for (FrameShell *s : allFrameHistory) {
    if (!s->poseValid) {
      continue;
    }

    if (setting_onlyLogKFPoses && s->marginalizedAt == s->id) {
      continue;
    }

    myfile << s->timestamp << " " << s->camToWorld.translation().transpose()
           << " " << s->camToWorld.so3().unit_quaternion().x() << " "
           << s->camToWorld.so3().unit_quaternion().y() << " "
           << s->camToWorld.so3().unit_quaternion().z() << " "
           << s->camToWorld.so3().unit_quaternion().w() << "\n";
  }
  myfile.close();
}

void FullSystem::addActiveFrame(ImageAndExposure *image, int id) {
  if (isLost) {
    return;
  }
  boost::unique_lock<boost::mutex> lock(trackMutex);

  FrameHessian *fh = PreprocessNewFrame(image, id);

  if (!initialized) {
    // use initializer!
    if (coarseInitializer->frameID < 0) {
      // first frame set. fh is kept by coarseInitializer.
      coarseInitializer->setFirst(&Hcalib, fh);
    } else if (coarseInitializer->trackFrame(fh, outputWrapper)) {
      // if SNAPPED
      initializeFromInitializer(fh);
      lock.unlock();
      deliverTrackedFrame(fh, true);
    } else {
      // if still initializing
      fh->shell->poseValid = false;
      delete fh;
    }
    return;
  } else {
    // do front-end operation.
    // ============== SWAP tracking reference?. ==============
    if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
      boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
      CoarseTracker *tmp = coarseTracker;
      coarseTracker = coarseTracker_forNewKF;
      coarseTracker_forNewKF = tmp;
    }

    Vec4 tres = trackNewCoarse(fh);
    if (!std::isfinite(tres[0]) || !std::isfinite(tres[1]) ||
        !std::isfinite(tres[2]) || !std::isfinite(tres[3])) {
      LOG(WARNING) << "Initial Tracking failed: LOST!";
      isLost = true;
      return;
    }

    bool needToMakeKF = false;
    if (setting_keyframesPerSecond > 0) {
      needToMakeKF =
          allFrameHistory.size() == 1 ||
          (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) >
              0.95f / setting_keyframesPerSecond;
    } else {
      Vec2 refToFh = AffLight::fromToVecExposure(
          coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
          coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

      // BRIGHTNESS CHECK
      const bool condition_1 = (allFrameHistory.size() == 1);
      const bool condition_2 =
          (setting_kfGlobalWeight * setting_maxShiftWeightT * sqrt(tres[1]) /
                   (wG[0] + hG[0]) +
               setting_kfGlobalWeight * setting_maxShiftWeightR *
                   sqrt(tres[2]) / (wG[0] + hG[0]) +
               setting_kfGlobalWeight * setting_maxShiftWeightRT *
                   sqrt(tres[3]) / (wG[0] + hG[0]) +
               setting_kfGlobalWeight * setting_maxAffineWeight *
                   fabs(log(refToFh[0])) >
           1);
      const bool condition_3 = (2 * coarseTracker->firstCoarseRMSE < tres[0]);
      needToMakeKF = condition_1 || condition_2 || condition_3;
    }

    for (IOWrap::Output3DWrapper *ow : outputWrapper) {
      ow->publishCamPose(fh->shell, &Hcalib);
    }

    lock.unlock();
    deliverTrackedFrame(fh, needToMakeKF);
    return;
  }
}

Vec4 FullSystem::trackNewCoarse(FrameHessian *fh) {
  CHECK_GT(allFrameHistory.size(), 0);
  // set pose initialization.

  for (IOWrap::Output3DWrapper *ow : outputWrapper) {
    ow->pushLiveFrame(fh);
  }

  FrameHessian *lastF = coarseTracker->lastRef;

  AffLight aff_last_2_l = AffLight(0, 0);

  std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
  if (allFrameHistory.size() == 2) {
    for (unsigned int i = 0; i < lastF_2_fh_tries.size(); ++i) {
      lastF_2_fh_tries.emplace_back(SE3());
    }
  } else {
    FrameShell *slast = allFrameHistory[allFrameHistory.size() - 2];
    FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3];
    SE3 slast_2_sprelast;
    SE3 lastF_2_slast;
    {
      // lock on global pose consistency!
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
      lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
      aff_last_2_l = slast->aff_g2l;
    }
    SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast.

    lastF_2_fh_tries.emplace_back(fh_2_slast.inverse() *
                                  lastF_2_slast); // assume constant motion.
    lastF_2_fh_tries.emplace_back(
        fh_2_slast.inverse() * fh_2_slast.inverse() *
        lastF_2_slast); // assume double motion (frame skipped)
    lastF_2_fh_tries.emplace_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() *
                                  lastF_2_slast); // assume half motion.
    lastF_2_fh_tries.emplace_back(lastF_2_slast); // assume zero motion.
    lastF_2_fh_tries.emplace_back(SE3());         // assume zero motion FROM KF.

    // Just try a TON of different initializations (all rotations). In the end,
    // if they don't work they will only be tried on the coarsest level, which
    // is super fast anyway. Also, if tracking rails here we loose, so we
    // really, really want to avoid that.
    for (float rotDelta = 0.02; rotDelta < 0.05; ++rotDelta) {
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
              Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
              Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
              Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
              Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
              Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
              Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
              Vec3(0, 0, 0)));
      lastF_2_fh_tries.emplace_back(
          fh_2_slast.inverse() * lastF_2_slast *
          SE3(Eigen::Quaterniond(1, rotDelta, rotDelta, rotDelta),
              Vec3(0, 0, 0)));
    }

    if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
      lastF_2_fh_tries.clear();
      lastF_2_fh_tries.emplace_back(SE3());
    }
  }

  Vec3 flowVecs = Vec3(100, 100, 100);
  SE3 lastF_2_fh = SE3();
  AffLight aff_g2l = AffLight(0, 0);

  // as long as maxResForImmediateAccept is not reached, I'll continue through
  // the options.
  // I'll keep track of the so-far best achieved residual for each level in
  // achievedRes.
  // If on a coarse level, tracking is WORSE than achievedRes, we will not
  // continue to save time.

  Vec5 achievedRes = Vec5::Constant(NAN);
  bool haveOneGood = false;
  int tryIterations = 0;
  for (unsigned int i = 0; i < lastF_2_fh_tries.size(); ++i) {
    AffLight aff_g2l_this = aff_last_2_l;
    SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

    // in each level has to be at least as good as the last try.
    bool trackingIsGood = coarseTracker->trackNewestCoarse(
        fh, lastF_2_fh_this, aff_g2l_this, PYR_LEVELS_USED - 1, achievedRes);
    ++tryIterations;

    if (i != 0) {
      LOG(WARNING) << "RE-TRACK ATTEMPT " << i << " with initOption " << i
                   << " and start-lvl " << PYR_LEVELS_USED - 1 << " (ab "
                   << aff_g2l_this.a << " " << aff_g2l_this.b
                   << "): " << achievedRes[0] << " " << achievedRes[1] << " "
                   << achievedRes[2] << " " << achievedRes[3] << " "
                   << achievedRes[4] << " -> "
                   << coarseTracker->lastResiduals[0] << " "
                   << coarseTracker->lastResiduals[1] << " "
                   << coarseTracker->lastResiduals[2] << " "
                   << coarseTracker->lastResiduals[3] << " "
                   << coarseTracker->lastResiduals[4];
    }

    // do we have a new winner?
    if (trackingIsGood &&
        std::isfinite((float)coarseTracker->lastResiduals[0]) &&
        !(coarseTracker->lastResiduals[0] >= achievedRes[0])) {
      // printf("take over. minRes %f -> %f!\n", achievedRes[0],
      // coarseTracker->lastResiduals[0]);
      flowVecs = coarseTracker->lastFlowIndicators;
      aff_g2l = aff_g2l_this;
      lastF_2_fh = lastF_2_fh_this;
      haveOneGood = true;
    }

    // take over achieved res (always).
    if (haveOneGood) {
      for (int i = 0; i < 5; ++i) {
        if (!std::isfinite((float)achievedRes[i]) ||
            achievedRes[i] > coarseTracker->lastResiduals[i]) {
          // take over if achievedRes is either bigger or NAN.
          achievedRes[i] = coarseTracker->lastResiduals[i];
        }
      }
    }

    if (haveOneGood &&
        achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold) {
      break;
    }
  }

  if (!haveOneGood) {
    LOG(ERROR) << "BIG ERROR! tracking failed entirely. Take predictred pose "
                  "and hope we may somehow recover.";
    flowVecs = Vec3(0, 0, 0);
    aff_g2l = aff_last_2_l;
    lastF_2_fh = lastF_2_fh_tries[0];
  }

  lastCoarseRMSE = achievedRes;

  // no lock required, as fh is not used anywhere yet.
  fh->shell->camToTrackingRef = lastF_2_fh.inverse();
  fh->shell->trackingRef = lastF->shell;
  fh->shell->aff_g2l = aff_g2l;
  fh->shell->camToWorld =
      fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

  if (coarseTracker->firstCoarseRMSE < 0) {
    coarseTracker->firstCoarseRMSE = achievedRes[0];
  }

  if (!setting_debugout_runquiet) {
    LOG(INFO) << "Coarse Tracker tracked ab = " << aff_g2l.a << " " << aff_g2l.b
              << "  (exp " << fh->ab_exposure << "). Res " << achievedRes[0]
              << "!";
  }

  if (setting_logStuff) {
    (*coarseTrackingLog) << std::setprecision(16) << fh->shell->id << " "
                         << fh->shell->timestamp << " " << fh->ab_exposure
                         << " " << fh->shell->camToWorld.log().transpose()
                         << " " << aff_g2l.a << " " << aff_g2l.b << " "
                         << achievedRes[0] << " " << tryIterations << "\n";
  }

  return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::traceNewCoarse(FrameHessian *fh) {
  boost::unique_lock<boost::mutex> lock(mapMutex);

  int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0,
      trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

  Mat33f K = Mat33f::Identity();
  K(0, 0) = Hcalib.fxl();
  K(1, 1) = Hcalib.fyl();
  K(0, 2) = Hcalib.cxl();
  K(1, 2) = Hcalib.cyl();

  // go through all active frames
  for (FrameHessian *host : frameHessians) {
    // Tcur_host = Tcur_w * Tw_host
    SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
    Vec3f Kt = K * hostToNew.translation().cast<float>();

    Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure,
                                            host->aff_g2l(), fh->aff_g2l())
                    .cast<float>();

    for (ImmaturePoint *ph : host->immaturePoints) {
      ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);
      switch (ph->lastTraceStatus) {
      case ImmaturePointStatus::IPS_GOOD:
        ++trace_good;
        break;
      case ImmaturePointStatus::IPS_BADCONDITION:
        ++trace_badcondition;
        break;
      case ImmaturePointStatus::IPS_OOB:
        ++trace_oob;
        break;
      case ImmaturePointStatus::IPS_OUTLIER:
        ++trace_out;
        break;
      case ImmaturePointStatus::IPS_SKIPPED:
        ++trace_skip;
        break;
      case ImmaturePointStatus::IPS_UNINITIALIZED:
        ++trace_uninitialized;
        break;
      default:
        break;
      }
      ++trace_total;
    }
  }
}

void FullSystem::activatePointsMT_Reductor(
    std::vector<PointHessian *> *optimized,
    std::vector<ImmaturePoint *> *toOptimize, int min, int max, Vec10 *stats,
    int tid) {
  ImmaturePointTemporaryResidual *tr =
      new ImmaturePointTemporaryResidual[frameHessians.size()];
  for (int k = min; k < max; ++k) {
    (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
  }
  delete[] tr;
}

void FullSystem::activatePointsMT() {
  if (ef->nPoints < setting_desiredPointDensity * 0.66) {
    currentMinActDist -= 0.8;
  } else if (ef->nPoints < setting_desiredPointDensity * 0.8) {
    currentMinActDist -= 0.5;
  } else if (ef->nPoints < setting_desiredPointDensity * 0.9) {
    currentMinActDist -= 0.2;
  } else if (ef->nPoints < setting_desiredPointDensity) {
    currentMinActDist -= 0.1;
  }

  if (ef->nPoints > setting_desiredPointDensity * 1.5) {
    currentMinActDist += 0.8;
  } else if (ef->nPoints > setting_desiredPointDensity * 1.3) {
    currentMinActDist += 0.5;
  } else if (ef->nPoints > setting_desiredPointDensity * 1.15) {
    currentMinActDist += 0.2;
  } else if (ef->nPoints > setting_desiredPointDensity) {
    currentMinActDist += 0.1;
  }

  if (currentMinActDist < 0) {
    currentMinActDist = 0;
  } else if (currentMinActDist > 4) {
    currentMinActDist = 4;
  }

  if (!setting_debugout_runquiet) {
    LOG(INFO) << "SPARSITY:  MinActDist " << currentMinActDist << " (need "
              << setting_desiredPointDensity << " points, have " << ef->nPoints
              << " points)!";
  }

  FrameHessian *newestHs = frameHessians.back();

  // make dist map.
  coarseDistanceMap->makeK(&Hcalib);
  coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

  // coarseTracker->debugPlotDistMap("distMap");

  std::vector<ImmaturePoint *> toOptimize;
  toOptimize.reserve(20000);

  for (FrameHessian *host : frameHessians) {
    // go through all active frames
    if (host == newestHs) {
      continue;
    }

    SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi =
        (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() *
         coarseDistanceMap->Ki[0]);
    Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

    for (unsigned int i = 0; i < host->immaturePoints.size(); ++i) {
      ImmaturePoint *ph = host->immaturePoints[i];
      ph->idxInImmaturePoints = i;

      // delete points that have never been traced successfully, or that are
      // outlier on the last trace.
      if (!std::isfinite(ph->idepth_max) ||
          ph->lastTraceStatus == IPS_OUTLIER) {
        // remove point.
        delete ph;
        host->immaturePoints[i] = 0;
        continue;
      }

      // can activate only if this is true.
      bool canActivate = (ph->lastTraceStatus == IPS_GOOD ||
                          ph->lastTraceStatus == IPS_SKIPPED ||
                          ph->lastTraceStatus == IPS_BADCONDITION ||
                          ph->lastTraceStatus == IPS_OOB) &&
                         ph->lastTracePixelInterval < 8 &&
                         ph->quality > setting_minTraceQuality &&
                         (ph->idepth_max + ph->idepth_min) > 0;

      // if I cannot activate the point, skip it. Maybe also delete it.
      if (!canActivate) {
        // if point will be out afterwards, delete it instead.
        if (ph->host->flaggedForMarginalization ||
            ph->lastTraceStatus == IPS_OOB) {
          delete ph;
          host->immaturePoints[i] = nullptr;
        }
        continue;
      }

      // see if we need to activate point due to distance map.
      Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) +
                  Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
      int u = ptp[0] / ptp[2] + 0.5f;
      int v = ptp[1] / ptp[2] + 0.5f;

      if (u > 0 && v > 0 && u < wG[1] && v < hG[1]) {
        float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] +
                     (ptp[0] - floorf(ptp[0]));

        if (dist >= currentMinActDist * ph->my_type) {
          coarseDistanceMap->addIntoDistFinal(u, v);
          toOptimize.emplace_back(ph);
        }
      } else {
        delete ph;
        host->immaturePoints[i] = 0;
      }
    }
  }

  std::vector<PointHessian *> optimized;
  optimized.resize(toOptimize.size());

  if (multiThreading) {
    treadReduce.reduce(
        boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized,
                    &toOptimize, boost::placeholders::_1,
                    boost::placeholders::_2, boost::placeholders::_3,
                    boost::placeholders::_4),
        0, toOptimize.size(), 50);
  } else {
    activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0,
                              0);
  }

  for (unsigned k = 0; k < toOptimize.size(); ++k) {
    PointHessian *newpoint = optimized[k];
    ImmaturePoint *ph = toOptimize[k];

    if (newpoint != 0 && newpoint != (PointHessian *)((long)(-1))) {
      newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;
      newpoint->host->pointHessians.emplace_back(newpoint);
      ef->insertPoint(newpoint);
      for (PointFrameResidual *r : newpoint->residuals) {
        ef->insertResidual(r);
      }
      CHECK_NOTNULL(newpoint->efPoint);
      delete ph;
    } else if (newpoint == (PointHessian *)((long)(-1)) ||
               ph->lastTraceStatus == IPS_OOB) {
      delete ph;
      ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
    } else {
      CHECK(newpoint == 0 || newpoint == (PointHessian *)((long)(-1)));
    }
  }

  for (FrameHessian *host : frameHessians) {
    for (int i = 0; i < (int)host->immaturePoints.size(); ++i) {
      if (host->immaturePoints[i] == 0) {
        host->immaturePoints[i] = host->immaturePoints.back();
        host->immaturePoints.pop_back();
        --i;
      }
    }
  }
}

void FullSystem::flagPointsForRemoval() {
  CHECK(EFIndicesValid);

  std::vector<FrameHessian *> fhsToKeepPoints;
  std::vector<FrameHessian *> fhsToMargPoints;

  {
    for (int i = ((int)frameHessians.size()) - 1;
         i >= 0 && i >= ((int)frameHessians.size()); --i) {
      if (!frameHessians[i]->flaggedForMarginalization) {
        fhsToKeepPoints.emplace_back(frameHessians[i]);
      }
    }

    for (size_t i = 0; i < frameHessians.size(); ++i) {
      if (frameHessians[i]->flaggedForMarginalization) {
        fhsToMargPoints.emplace_back(frameHessians[i]);
      }
    }
  }

  int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

  for (FrameHessian *host : frameHessians) {
    // go through all active frames
    for (size_t i = 0; i < host->pointHessians.size(); ++i) {
      PointHessian *ph = host->pointHessians[i];
      if (ph == nullptr) {
        continue;
      }

      if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) {
        host->pointHessiansOut.emplace_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        host->pointHessians[i] = nullptr;
        ++flag_nores;
      } else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) ||
                 host->flaggedForMarginalization) {
        ++flag_oob;
        if (ph->isInlierNew()) {
          flag_in++;
          int ngoodRes = 0;
          for (PointFrameResidual *r : ph->residuals) {
            r->resetOOB();
            r->linearize(&Hcalib);
            r->efResidual->isLinearized = false;
            r->applyRes(true);
            if (r->efResidual->isActive()) {
              r->efResidual->fixLinearizationF(ef);
              ++ngoodRes;
            }
          }
          if (ph->idepth_hessian > setting_minIdepthH_marg) {
            ++flag_inin;
            ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
            host->pointHessiansMarginalized.emplace_back(ph);
          } else {
            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
            host->pointHessiansOut.emplace_back(ph);
          }
        } else {
          host->pointHessiansOut.emplace_back(ph);
          ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        }

        host->pointHessians[i] = 0;
      }
    }

    for (int i = 0; i < (int)host->pointHessians.size(); ++i) {
      if (host->pointHessians[i] == 0) {
        host->pointHessians[i] = host->pointHessians.back();
        host->pointHessians.pop_back();
        --i;
      }
    }
  }
}

void FullSystem::deliverTrackedFrame(FrameHessian *const fh,
                                     const bool needKF) {
  if (linearizeOperation) {
    if (goStepByStep && lastRefStopID != coarseTracker->refFrameID) {
      MinimalImageF3 img(wG[0], hG[0], fh->dI);
      IOWrap::displayImage("frameToTrack", &img);
      while (true) {
        char k = IOWrap::waitKey(0);
        if (k == ' ') {
          break;
        }
        handleKey(k);
      }
      lastRefStopID = coarseTracker->refFrameID;
    } else {
      handleKey(IOWrap::waitKey(1));
    }

    if (needKF) {
      makeKeyFrame(fh);
    } else {
      makeNonKeyFrame(fh);
    }
  } else {
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
    unmappedTrackedFrames.emplace_back(fh);
    if (needKF) {
      needNewKFAfter = fh->shell->trackingRef->id;
    }
    trackedFrameSignal.notify_all();

    while (coarseTracker_forNewKF->refFrameID == -1 &&
           coarseTracker->refFrameID == -1) {
      mappedFrameSignal.wait(lock);
    }

    lock.unlock();
  }
}

void FullSystem::mappingLoop() {
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

  while (runMapping) {
    while (unmappedTrackedFrames.size() == 0) {
      trackedFrameSignal.wait(lock);
      if (!runMapping) {
        return;
      }
    }

    FrameHessian *fh = unmappedTrackedFrames.front();
    unmappedTrackedFrames.pop_front();

    // guaranteed to make a KF for the very first two tracked frames.
    if (allKeyFramesHistory.size() <= 2) {
      lock.unlock();
      makeKeyFrame(fh);
      lock.lock();
      mappedFrameSignal.notify_all();
      continue;
    }

    if (unmappedTrackedFrames.size() > 3) {
      needToKetchupMapping = true;
    }

    if (unmappedTrackedFrames.size() > 0) {
      // if there are other frames to tracke, do that first.
      lock.unlock();
      makeNonKeyFrame(fh);
      lock.lock();

      if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
        FrameHessian *fh = unmappedTrackedFrames.front();
        unmappedTrackedFrames.pop_front();
        {
          boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
          assert(fh->shell->trackingRef != 0);
          fh->shell->camToWorld =
              fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
          fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),
                               fh->shell->aff_g2l);
        }
        delete fh;
      }

    } else {
      if (setting_realTimeMaxKF ||
          needNewKFAfter >= frameHessians.back()->shell->id) {
        lock.unlock();
        makeKeyFrame(fh);
        needToKetchupMapping = false;
        lock.lock();
      } else {
        lock.unlock();
        makeNonKeyFrame(fh);
        lock.lock();
      }
    }
    mappedFrameSignal.notify_all();
  }
  LOG(INFO) << "MAPPING FINISHED!";
}

void FullSystem::blockUntilMappingIsFinished() {
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
  runMapping = false;
  trackedFrameSignal.notify_all();
  lock.unlock();

  mappingThread.join();
}

void FullSystem::makeNonKeyFrame(FrameHessian *const fh) {
  // needs to be set by mapping thread. no lock required since we are in mapping
  // thread.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    CHECK_NOTNULL(fh->shell->trackingRef);
    fh->shell->camToWorld =
        fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
  }

  traceNewCoarse(fh);
  delete fh;
}

void FullSystem::makeKeyFrame(FrameHessian *const fh) {
  // needs to be set by mapping thread
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    CHECK_NOTNULL(fh->shell->trackingRef);

    // Tw_cur = Tw_ref * Tref_cur
    fh->shell->camToWorld =
        fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
  }

  traceNewCoarse(fh);

  boost::unique_lock<boost::mutex> lock(mapMutex);

  // ============== Flag Frames to be Marginalized. ==============
  flagFramesForMarginalization(fh);

  // ============== add New Frame to Hessian Struct. ==============
  fh->idx = frameHessians.size();
  frameHessians.emplace_back(fh);
  fh->frameID = allKeyFramesHistory.size();
  allKeyFramesHistory.emplace_back(fh->shell);
  ef->insertFrame(fh, &Hcalib);

  setPrecalcValues();

  // ============== add new residuals for old points ==============
  int numFwdResAdde = 0;
  for (FrameHessian *fh1 : frameHessians) {
    // go through all active frames
    if (fh1 == fh) {
      continue;
    }
    for (PointHessian *ph : fh1->pointHessians) {
      PointFrameResidual *r = new PointFrameResidual(ph, fh1, fh);
      r->setState(ResState::IN);
      ph->residuals.emplace_back(r);
      ef->insertResidual(r);
      ph->lastResiduals[1] = ph->lastResiduals[0];
      ph->lastResiduals[0] =
          std::pair<PointFrameResidual *, ResState>(r, ResState::IN);
      ++numFwdResAdde;
    }
  }

  // ============== Activate Points (& flag for marginalization). ==============
  activatePointsMT();
  ef->makeIDX();

  // ============== OPTIMIZE ALL ==============
  fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
  float rmse = optimize(setting_maxOptIterations);

  // ============== Figure Out if INITIALIZATION FAILED ==============
  if (allKeyFramesHistory.size() <= 4) {
    const bool condition1 = (allKeyFramesHistory.size() == 2 &&
                             rmse > 20 * benchmark_initializerSlackFactor);
    const bool condition2 = (allKeyFramesHistory.size() == 3 &&
                             rmse > 13 * benchmark_initializerSlackFactor);
    const bool condition3 = (allKeyFramesHistory.size() == 4 &&
                             rmse > 9 * benchmark_initializerSlackFactor);
    if (condition1 || condition2 || condition3) {
      LOG(WARNING) << "I THINK INITIALIZATION FAILED! Resetting.";
      initFailed = true;
    }
  }

  if (isLost) {
    return;
  }

  // ============== REMOVE OUTLIER ==============
  removeOutliers();

  {
    boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
    coarseTracker_forNewKF->makeK(&Hcalib);
    coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);

    coarseTracker_forNewKF->debugPlotIDepthMap(
        &minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
    coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
  }

  debugPlot("post Optimize");

  // ============== (Activate-)Marginalize Points ==============
  flagPointsForRemoval();
  ef->dropPointsF();
  getNullspaces(ef->lastNullspaces_pose, ef->lastNullspaces_scale,
                ef->lastNullspaces_affA, ef->lastNullspaces_affB);
  ef->marginalizePointsF();

  // ============== add new Immature points & new residuals ==============
  makeNewTraces(fh, 0);

  for (IOWrap::Output3DWrapper *ow : outputWrapper) {
    ow->publishGraph(ef->connectivityMap);
    ow->publishKeyframes(frameHessians, false, &Hcalib);
  }

  // ============== Marginalize Frames ==============
  for (size_t i = 0; i < frameHessians.size(); ++i) {
    if (frameHessians[i]->flaggedForMarginalization) {
      marginalizeFrame(frameHessians[i]);
      i = 0;
    }
  }

  printLogLine();
}

void FullSystem::initializeFromInitializer(FrameHessian *newFrame) {
  LOG(WARNING) << "Initalize";
  boost::unique_lock<boost::mutex> lock(mapMutex);

  // add first frame.
  FrameHessian *firstFrame = coarseInitializer->firstFrame;
  firstFrame->idx = frameHessians.size();
  frameHessians.emplace_back(firstFrame);
  firstFrame->frameID = allKeyFramesHistory.size();
  allKeyFramesHistory.emplace_back(firstFrame->shell);
  ef->insertFrame(firstFrame, &Hcalib);
  setPrecalcValues();

  firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
  firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
  firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);

  // sum of inverse depths, number of inverse depths
  float sumID = 1e-5, numID = 1e-5;
  for (int i = 0; i < coarseInitializer->numPoints[0]; ++i) {
    sumID += coarseInitializer->points[0][i].iR;
    ++numID;
  }
  // inverse of mean inverse depth
  float rescaleFactor = 1 / (sumID / numID);

  // randomly sub-select the points I need.
  float keepPercentage =
      setting_desiredPointDensity / coarseInitializer->numPoints[0];

  if (!setting_debugout_runquiet) {
    LOG(INFO) << "Initialization: keep " << 100 * keepPercentage << "% (need "
              << setting_desiredPointDensity << ", have "
              << coarseInitializer->numPoints[0] << ")!";
  }

  for (int i = 0; i < coarseInitializer->numPoints[0]; ++i) {
    if (rand() / (float)RAND_MAX > keepPercentage) {
      continue;
    }

    Pnt *point = coarseInitializer->points[0] + i;
    ImmaturePoint *pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f,
                                          firstFrame, point->my_type, &Hcalib);

    if (!std::isfinite(pt->energyTH)) {
      delete pt;
      continue;
    }

    pt->idepth_max = pt->idepth_min = 1;
    PointHessian *ph = new PointHessian(pt, &Hcalib);
    delete pt;
    if (!std::isfinite(ph->energyTH)) {
      delete ph;
      continue;
    }

    ph->setIdepthScaled(point->iR * rescaleFactor);
    ph->setIdepthZero(ph->idepth);
    ph->hasDepthPrior = true;
    ph->setPointStatus(PointHessian::ACTIVE);

    firstFrame->pointHessians.emplace_back(ph);
    ef->insertPoint(ph);
  }

  SE3 firstToNew = coarseInitializer->thisToNext;
  firstToNew.translation() /= rescaleFactor;
  LOG(INFO) << "DSO computed t: " << firstToNew.translation().transpose();

  // really no lock required, as we are initializing.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    firstFrame->shell->camToWorld = SE3();
    firstFrame->shell->aff_g2l = AffLight(0, 0);
    firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),
                                 firstFrame->shell->aff_g2l);
    firstFrame->shell->trackingRef = 0;
    firstFrame->shell->camToTrackingRef = SE3();

    newFrame->shell->camToWorld = firstToNew.inverse();
    newFrame->shell->aff_g2l = AffLight(0, 0);
    newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),
                               newFrame->shell->aff_g2l);
    newFrame->shell->trackingRef = firstFrame->shell;
    newFrame->shell->camToTrackingRef = firstToNew.inverse();
  }

  initialized = true;
  LOG(INFO) << "INITIALIZE FROM INITIALIZER ("
            << firstFrame->pointHessians.size() << " pts)!";
}

void FullSystem::makeNewTraces(FrameHessian *newFrame, float *gtDepth) {
  pixelSelector->allowFast = true;
  int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,
                                               setting_desiredImmatureDensity);

  newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
  newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
  newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

  for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; ++y) {
    for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; ++x) {
      int i = x + y * wG[0];
      if (selectionMap[i] == 0) {
        continue;
      }

      ImmaturePoint *impt =
          new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib);
      if (!std::isfinite(impt->energyTH)) {
        delete impt;
      } else {
        newFrame->immaturePoints.emplace_back(impt);
      }
    }
  }
}

void FullSystem::setPrecalcValues() {
  for (FrameHessian *fh : frameHessians) {
    fh->targetPrecalc.resize(frameHessians.size());
    for (size_t i = 0; i < frameHessians.size(); ++i) {
      fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
    }
  }

  ef->setDeltaF(&Hcalib);
}

void FullSystem::printLogLine() {
  if (frameHessians.size() == 0) {
    return;
  }

  if (!setting_debugout_runquiet) {
    LOG(INFO) << "LOG " << allKeyFramesHistory.back()->id << ": "
              << statistics_lastFineTrackRMSE << " fine. Res: " << ef->resInA
              << " A, " << ef->resInL << " L, " << ef->resInM << " M; ("
              << statistics_numForceDroppedResFwd << " / "
              << statistics_numForceDroppedResBwd
              << ") forceDrop. a=" << allKeyFramesHistory.back()->aff_g2l.a
              << ", b=" << allKeyFramesHistory.back()->aff_g2l.b << ". Window "
              << frameHessians.back()->shell->id -
                     frameHessians.front()->shell->id
              << " (" << frameHessians.size() << ")";
  }

  if (!setting_logStuff) {
    return;
  }

  if (numsLog != 0) {
    (*numsLog) << allKeyFramesHistory.back()->id << " "
               << statistics_lastFineTrackRMSE << " "
               << (int)statistics_numCreatedPoints << " "
               << (int)statistics_numActivatedPoints << " "
               << (int)statistics_numDroppedPoints << " "
               << (int)statistics_lastNumOptIts << " " << ef->resInA << " "
               << ef->resInL << " " << ef->resInM << " "
               << statistics_numMargResFwd << " " << statistics_numMargResBwd
               << " " << statistics_numForceDroppedResFwd << " "
               << statistics_numForceDroppedResBwd << " "
               << frameHessians.back()->aff_g2l().a << " "
               << frameHessians.back()->aff_g2l().b << " "
               << frameHessians.back()->shell->id -
                      frameHessians.front()->shell->id
               << " " << (int)frameHessians.size() << " "
               << "\n";
    numsLog->flush();
  }
}

void FullSystem::printEigenValLine() {
  if (!setting_logStuff) {
    return;
  }
  if (ef->lastHS.rows() < 12) {
    return;
  }

  MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS,
                                          ef->lastHS.cols() - CPARS);
  MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS,
                                          ef->lastHS.cols() - CPARS);
  int n = Hp.cols() / 8;
  assert(Hp.cols() % 8 == 0);

  // sub-select
  for (int i = 0; i < n; ++i) {
    MatXX tmp6 = Hp.block(i * 8, 0, 6, n * 8);
    Hp.block(i * 6, 0, 6, n * 8) = tmp6;

    MatXX tmp2 = Ha.block(i * 8 + 6, 0, 2, n * 8);
    Ha.block(i * 2, 0, 2, n * 8) = tmp2;
  }
  for (int i = 0; i < n; ++i) {
    MatXX tmp6 = Hp.block(0, i * 8, n * 8, 6);
    Hp.block(0, i * 6, n * 8, 6) = tmp6;

    MatXX tmp2 = Ha.block(0, i * 8 + 6, n * 8, 2);
    Ha.block(0, i * 2, n * 8, 2) = tmp2;
  }

  VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
  VecX eigenP = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
  VecX eigenA = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
  VecX diagonal = ef->lastHS.diagonal();

  std::sort(eigenvaluesAll.data(),
            eigenvaluesAll.data() + eigenvaluesAll.size());
  std::sort(eigenP.data(), eigenP.data() + eigenP.size());
  std::sort(eigenA.data(), eigenA.data() + eigenA.size());

  int nz = std::max(100, setting_maxFrames * 10);

  if (eigenAllLog != 0) {
    VecX ea = VecX::Zero(nz);
    ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
    (*eigenAllLog) << allKeyFramesHistory.back()->id << " " << ea.transpose()
                   << "\n";
    eigenAllLog->flush();
  }
  if (eigenALog != 0) {
    VecX ea = VecX::Zero(nz);
    ea.head(eigenA.size()) = eigenA;
    (*eigenALog) << allKeyFramesHistory.back()->id << " " << ea.transpose()
                 << "\n";
    eigenALog->flush();
  }
  if (eigenPLog != 0) {
    VecX ea = VecX::Zero(nz);
    ea.head(eigenP.size()) = eigenP;
    (*eigenPLog) << allKeyFramesHistory.back()->id << " " << ea.transpose()
                 << "\n";
    eigenPLog->flush();
  }

  if (DiagonalLog != 0) {
    VecX ea = VecX::Zero(nz);
    ea.head(diagonal.size()) = diagonal;
    (*DiagonalLog) << allKeyFramesHistory.back()->id << " " << ea.transpose()
                   << "\n";
    DiagonalLog->flush();
  }

  if (variancesLog != 0) {
    VecX ea = VecX::Zero(nz);
    ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
    (*variancesLog) << allKeyFramesHistory.back()->id << " " << ea.transpose()
                    << "\n";
    variancesLog->flush();
  }

  std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
  (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
  for (unsigned int i = 0; i < nsp.size(); ++i)
    (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " "
                     << nsp[i].dot(ef->lastbS) << " ";
  (*nullspacesLog) << "\n";
  nullspacesLog->flush();
}

void FullSystem::printFrameLifetimes() {
  if (!setting_logStuff) {
    return;
  }

  boost::unique_lock<boost::mutex> lock(trackMutex);

  std::ofstream *lg = new std::ofstream();
  lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
  lg->precision(15);

  for (FrameShell *s : allFrameHistory) {
    (*lg) << s->id << " " << s->marginalizedAt << " "
          << s->statistics_goodResOnThis << " "
          << s->statistics_outlierResOnThis << " " << s->movedByOpt;

    (*lg) << "\n";
  }

  lg->close();
  delete lg;
}

void FullSystem::printEvalLine() { return; }

FrameHessian *FullSystem::PreprocessNewFrame(ImageAndExposure *const image,
                                             const int id) {
  // ============== add into allFrameHistory ==============
  FrameHessian *fh = new FrameHessian();
  FrameShell *shell = new FrameShell();

  // no lock required, as fh is not used anywhere yet.
  shell->camToWorld = SE3();
  shell->aff_g2l = AffLight(0, 0);
  shell->marginalizedAt = shell->id = allFrameHistory.size();
  shell->timestamp = image->timestamp;
  shell->incoming_id = id;
  shell->init_scale = image->init_scale;
  fh->shell = shell;
  allFrameHistory.emplace_back(shell);

  // ============== make Images / derivatives etc. ==============
  fh->ab_exposure = image->exposure_time;
  fh->makeImages(image->image, &Hcalib);

  return fh;
}

} // namespace dso
