#include "full_system/full_system.h"

#include <stdio.h>
#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "full_system/residual_projections.h"
#include "io_wrapper/image_display.h"
#include "optimization_backend/energy_functional/energy_functional.h"
#include "optimization_backend/energy_functional/energy_functional_structs.h"
#include "util/global_calib.h"
#include "util/global_funcs.h"

namespace dso {

void FullSystem::linearizeAll_Reductor(
    const bool fixLinearization,
    std::vector<PointFrameResidual*>* const toRemove, const int min,
    const int max, Vec10* const stats, const int tid) {
  CHECK_GE(min, 0);
  CHECK_LE(max, activeResiduals.size());
  CHECK_LT(min, max);
  for (int k = min; k < max; ++k) {
    PointFrameResidual* r = activeResiduals[k];
    (*stats)[0] += r->linearize(&Hcalib);  // add the residual of this point

    if (fixLinearization) {
      r->applyRes(true);

      if (r->efResidual->isActive()) {
        if (r->isNew) {
          PointHessian* p = r->point;

          // projected point assuming infinite depth.
          Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll *
                          Vec3f(p->u, p->v, 1);

          // projected point with real depth.
          Vec3f ptp = ptp_inf +
                      r->host->targetPrecalc[r->target->idx].PRE_KtTll *
                          p->idepth_scaled;

          // 0.01 = one pixel.
          float relBS =
              0.01 *
              ((ptp_inf.head<2>() / ptp_inf[2]) - (ptp.head<2>() / ptp[2]))
                  .norm();

          if (relBS > p->maxRelBaseline) {
            p->maxRelBaseline = relBS;
          }

          ++(p->numGoodResiduals);
        }
      } else {
        toRemove[tid].emplace_back(activeResiduals[k]);
      }
    }
  }
}

void FullSystem::applyRes_Reductor(const bool copyJacobians, const int min,
                                   const int max, Vec10* stats, int tid) {
  CHECK_GE(min, 0);
  CHECK_LE(max, activeResiduals.size());
  CHECK_LT(min, max);
  for (int k = min; k < max; ++k) {
    activeResiduals[k]->applyRes(true);
  }
}

void FullSystem::setNewFrameEnergyTH() {
  // collect all residuals and make decision on TH.
  allResVec.clear();
  allResVec.reserve(activeResiduals.size() * 2);
  FrameHessian* newFrame = frameHessians.back();

  for (PointFrameResidual* r : activeResiduals) {
    if (r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame) {
      allResVec.emplace_back(r->state_NewEnergyWithOutlier);
    }
  }

  if (allResVec.size() == 0) {
    // should never happen, but lets make sure.
    newFrame->frameEnergyTH = 12 * 12 * patternNum;
    return;
  }

  const int nthIdx = setting_frameEnergyTHN * allResVec.size();

  CHECK_LT(nthIdx, allResVec.size());
  CHECK_LT(setting_frameEnergyTHN, 1);

  std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx,
                   allResVec.end());
  const float nthElement = sqrtf(allResVec[nthIdx]);

  newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
  newFrame->frameEnergyTH =
      26.0f * setting_frameEnergyTHConstWeight +
      newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
  newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
  newFrame->frameEnergyTH *=
      setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
}

Vec3 FullSystem::linearizeAll(const bool fixLinearization) {
  double lastEnergyP = 0;  // energy of all active points
  double lastEnergyR = 0;
  double num = 0;

  // If multi threads, every toRemove[i] will be used.
  // Otherwise, only toRemove[0] will be used.
  std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; ++i) {
    toRemove[i].clear();
  }

  if (multiThreading) {
    treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this,
                                   fixLinearization, toRemove, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4),
                       0, activeResiduals.size(), 0);
    lastEnergyP = treadReduce.stats[0];
  } else {
    Vec10 stats;
    linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(),
                          &stats, 0);
    lastEnergyP = stats[0];
  }

  setNewFrameEnergyTH();

  if (fixLinearization) {
    for (PointFrameResidual* r : activeResiduals) {
      PointHessian* ph = r->point;
      if (ph->lastResiduals[0].first == r) {
        ph->lastResiduals[0].second = r->state_state;
      } else if (ph->lastResiduals[1].first == r) {
        ph->lastResiduals[1].second = r->state_state;
      }
    }

    int nResRemoved = 0;
    for (int i = 0; i < NUM_THREADS; ++i) {
      for (PointFrameResidual* r : toRemove[i]) {
        PointHessian* ph = r->point;

        if (ph->lastResiduals[0].first == r) {
          ph->lastResiduals[0].first = 0;
        } else if (ph->lastResiduals[1].first == r) {
          ph->lastResiduals[1].first = 0;
        }

        for (unsigned int k = 0; k < ph->residuals.size(); ++k)
          if (ph->residuals[k] == r) {
            ef->dropResidual(r->efResidual);
            deleteOut<PointFrameResidual>(ph->residuals, k);
            ++nResRemoved;
            break;
          }
      }
    }
  }

  return Vec3(lastEnergyP, lastEnergyR, num);
}

bool FullSystem::doStepFromBackup(float stepfacC, float stepfacT,
                                  float stepfacR, float stepfacA,
                                  float stepfacD) {
  Vec10 pstepfac;
  pstepfac.segment<3>(0).setConstant(stepfacT);
  pstepfac.segment<3>(3).setConstant(stepfacR);
  pstepfac.segment<4>(6).setConstant(stepfacA);

  float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

  float sumNID = 0;

  if (setting_solverMode & SOLVER_MOMENTUM) {
    Hcalib.setValue(Hcalib.value_backup + Hcalib.step);
    for (FrameHessian* fh : frameHessians) {
      Vec10 step = fh->step;
      step.head<6>() += 0.5f * (fh->step_backup.head<6>());

      fh->setState(fh->state_backup + step);
      sumA += step[6] * step[6];
      sumB += step[7] * step[7];
      sumT += step.segment<3>(0).squaredNorm();
      sumR += step.segment<3>(3).squaredNorm();

      for (PointHessian* ph : fh->pointHessians) {
        float step = ph->step + 0.5f * (ph->step_backup);
        ph->setIdepth(ph->idepth_backup + step);
        sumID += step * step;
        sumNID += fabsf(ph->idepth_backup);
        ++numID;

        ph->setIdepthZero(ph->idepth_backup + step);
      }
    }
  } else {
    Hcalib.setValue(Hcalib.value_backup + stepfacC * Hcalib.step);
    for (FrameHessian* fh : frameHessians) {
      fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
      sumA += fh->step[6] * fh->step[6];
      sumB += fh->step[7] * fh->step[7];
      sumT += fh->step.segment<3>(0).squaredNorm();
      sumR += fh->step.segment<3>(3).squaredNorm();

      for (PointHessian* ph : fh->pointHessians) {
        ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
        sumID += ph->step * ph->step;
        sumNID += fabsf(ph->idepth_backup);
        ++numID;

        ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);
      }
    }
  }

  sumA /= frameHessians.size();
  sumB /= frameHessians.size();
  sumR /= frameHessians.size();
  sumT /= frameHessians.size();
  sumID /= numID;
  sumNID /= numID;

  if (!setting_debugout_runquiet) {
    LOG(INFO) << "STEPS: A " << sqrtf(sumA) / (0.0005 * setting_thOptIterations)
              << "; B " << sqrtf(sumB) / (0.00005 * setting_thOptIterations)
              << "; R " << sqrtf(sumR) / (0.00005 * setting_thOptIterations)
              << "; T "
              << sqrtf(sumT) * sumNID / (0.00005 * setting_thOptIterations)
              << ".";
  }

  EFDeltaValid = false;
  setPrecalcValues();

  return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
         sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
         sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
         sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
}

void FullSystem::backupState(const bool backupLastStep) {
  if (setting_solverMode & SOLVER_MOMENTUM) {
    // We never come into this part
    if (backupLastStep) {
      Hcalib.step_backup = Hcalib.step;
      Hcalib.value_backup = Hcalib.value;
      for (FrameHessian* fh : frameHessians) {
        fh->step_backup = fh->step;
        fh->state_backup = fh->get_state();
        for (PointHessian* ph : fh->pointHessians) {
          ph->idepth_backup = ph->idepth;
          ph->step_backup = ph->step;
        }
      }
    } else {
      Hcalib.step_backup.setZero();
      Hcalib.value_backup = Hcalib.value;
      for (FrameHessian* fh : frameHessians) {
        fh->step_backup.setZero();
        fh->state_backup = fh->get_state();
        for (PointHessian* ph : fh->pointHessians) {
          ph->idepth_backup = ph->idepth;
          ph->step_backup = 0;
        }
      }
    }
  } else {
    // We always work here
    Hcalib.value_backup = Hcalib.value;
    for (FrameHessian* fh : frameHessians) {
      fh->state_backup = fh->get_state();
      for (PointHessian* ph : fh->pointHessians) {
        ph->idepth_backup = ph->idepth;
      }
    }
  }
}

void FullSystem::loadSateBackup() {
  Hcalib.setValue(Hcalib.value_backup);
  for (FrameHessian* fh : frameHessians) {
    fh->setState(fh->state_backup);
    for (PointHessian* ph : fh->pointHessians) {
      ph->setIdepth(ph->idepth_backup);
      ph->setIdepthZero(ph->idepth_backup);
    }
  }

  EFDeltaValid = false;
  setPrecalcValues();
}

double FullSystem::calcLEnergy() {
  if (setting_forceAceptStep) {
    return 0;
  }

  double Ef = ef->calcLEnergyF_MT();
  return Ef;
}

double FullSystem::calcMEnergy() {
  if (setting_forceAceptStep) {
    return 0;
  }
  return ef->calcMEnergyF();
}

void FullSystem::printOptRes(const Vec3& res, double resL, double resM,
                             double resPrior, double LExact, float a, float b) {
  LOG(INFO) << "A(" << res[0] << ")=(AV "
            << sqrtf((float)(res[0] / (patternNum * ef->resInA)))
            << "). Num: A(" << ef->resInA << ") + M(" << ef->resInM << "); ab "
            << a << " " << b << "!";
}

float FullSystem::optimize(int mnumOptIts) {
  if (frameHessians.size() < 2) {
    return 0;
  } else if (frameHessians.size() < 3) {
    mnumOptIts = 20;
  } else if (frameHessians.size() < 4) {
    mnumOptIts = 15;
  }

  //----- Get statistics and active residuals -----//
  activeResiduals.clear();
  int numPoints = 0;  // number of pointHessians
  int numLRes = 0;    // number of linearized PointFrameResiduals
  for (FrameHessian* fh : frameHessians) {
    for (PointHessian* ph : fh->pointHessians) {
      for (PointFrameResidual* r : ph->residuals) {
        if (!r->efResidual->isLinearized) {
          activeResiduals.emplace_back(r);
          r->resetOOB();
        } else {
          ++numLRes;
        }
      }
      ++numPoints;
    }
  }

  if (!setting_debugout_runquiet) {
    LOG(INFO) << "OPTIMIZE " << ef->nPoints << " pts, "
              << activeResiduals.size() << " active res, " << numLRes
              << " lin res!";
  }
  // ------------------------------------------------

  Vec3 lastEnergy = linearizeAll(false);

  double lastEnergyL = calcLEnergy();  // always 0
  double lastEnergyM = calcMEnergy();  // always 0

  if (multiThreading) {
    treadReduce.reduce(
        boost::bind(&FullSystem::applyRes_Reductor, this, true, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4),
        0, activeResiduals.size(), 50);
  } else {
    applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);
  }

  if (!setting_debugout_runquiet) {
    LOG(INFO) << "Initial Error";
    printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0,
                frameHessians.back()->aff_g2l().a,
                frameHessians.back()->aff_g2l().b);
  }

  debugPlotTracking();

  double lambda = 1e-1;
  float stepsize = 1;

  // 4 + 8 * N (Note: 8 = pose+ (a, b))
  VecX previousX = VecX::Constant(CPARS + 8 * frameHessians.size(), NAN);
  for (int iteration = 0; iteration < mnumOptIts; ++iteration) {
    // solve!
    backupState(iteration != 0);
    // solveSystemNew(0);
    solveSystem(iteration, lambda);
    double incDirChange = (1e-20 + previousX.dot(ef->lastX)) /
                          (1e-20 + previousX.norm() * ef->lastX.norm());
    previousX = ef->lastX;

    if (std::isfinite(incDirChange) &&
        (setting_solverMode & SOLVER_STEPMOMENTUM)) {
      float newStepsize = exp(incDirChange * 1.4);
      if (incDirChange < 0 && stepsize > 1) {
        stepsize = 1;
      }

      stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
      if (stepsize > 2) {
        stepsize = 2;
      }
      if (stepsize < 0.25) {
        stepsize = 0.25;
      }
    }

    bool canbreak =
        doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

    // eval new energy!
    Vec3 newEnergy = linearizeAll(false);

    double newEnergyL = calcLEnergy();  // always 0
    double newEnergyM = calcMEnergy();  // always 0

    if (!setting_debugout_runquiet) {
      const std::string status =
          (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
           lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)
              ? "ACCEPT"
              : "REJECT";
      LOG(INFO) << status << ", iteration: " << iteration
                << ", log10(lambda): " << log10(lambda)
                << ", incDirChange: " << incDirChange
                << ", stepsize: " << stepsize << "): ";
      printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0,
                  frameHessians.back()->aff_g2l().a,
                  frameHessians.back()->aff_g2l().b);
    }

    if (setting_forceAceptStep ||
        (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
         lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)) {
      if (multiThreading) {
        treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this,
                                       true, boost::placeholders::_1, boost::placeholders::_2, boost::placeholders::_3, boost::placeholders::_4),
                           0, activeResiduals.size(), 50);
      } else {
        applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);
      }

      lastEnergy = newEnergy;
      lastEnergyL = newEnergyL;
      lastEnergyM = newEnergyM;

      lambda *= 0.25;
    } else {
      loadSateBackup();
      lastEnergy = linearizeAll(false);

      lastEnergyL = calcLEnergy();  // always 0
      lastEnergyM = calcMEnergy();  // always 0
      lambda *= 1e2;
    }

    if (canbreak && iteration >= setting_minOptIterations) {
      break;
    }
  }

  Vec10 newStateZero = Vec10::Zero();
  newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);

  frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam,
                                  newStateZero);
  EFDeltaValid = false;
  EFAdjointsValid = false;
  ef->setAdjointsF(&Hcalib);
  setPrecalcValues();

  lastEnergy = linearizeAll(true);

  if (!std::isfinite(lastEnergy[0]) || !std::isfinite(lastEnergy[1]) ||
      !std::isfinite(lastEnergy[2])) {
    LOG(ERROR) << "KF Tracking failed: LOST!";
    isLost = true;
  }

  statistics_lastFineTrackRMSE =
      sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));

  if (calibLog != 0) {
    (*calibLog) << Hcalib.value_scaled.transpose() << " "
                << frameHessians.back()->get_state_scaled().transpose() << " "
                << sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)))
                << " " << ef->resInM << "\n";
    calibLog->flush();
  }

  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    for (FrameHessian* fh : frameHessians) {
      fh->shell->camToWorld = fh->PRE_camToWorld;
      fh->shell->aff_g2l = fh->aff_g2l();
    }
  }

  debugPlotTracking();

  return sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));
}

void FullSystem::solveSystem(int iteration, double lambda) {
  ef->lastNullspaces_forLogging =
      getNullspaces(ef->lastNullspaces_pose, ef->lastNullspaces_scale,
                    ef->lastNullspaces_affA, ef->lastNullspaces_affB);

  ef->solveSystemF(iteration, lambda, &Hcalib);
}

void FullSystem::removeOutliers() {
  int numPointsDropped = 0;
  for (FrameHessian* fh : frameHessians) {
    for (unsigned int i = 0; i < fh->pointHessians.size(); ++i) {
      PointHessian* ph = fh->pointHessians[i];
      if (ph == nullptr) {
        continue;
      }

      if (ph->residuals.empty()) {
        fh->pointHessiansOut.emplace_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        fh->pointHessians[i] = fh->pointHessians.back();
        fh->pointHessians.pop_back();
        --i;
        ++numPointsDropped;
      }
    }
  }
  ef->dropPointsF();
}

std::vector<VecX> FullSystem::getNullspaces(
    std::vector<VecX>& nullspaces_pose, std::vector<VecX>& nullspaces_scale,
    std::vector<VecX>& nullspaces_affA, std::vector<VecX>& nullspaces_affB) {
  nullspaces_pose.clear();
  nullspaces_scale.clear();
  nullspaces_affA.clear();
  nullspaces_affB.clear();

  int n = CPARS + frameHessians.size() * 8;
  std::vector<VecX> nullspaces_x0_pre;
  for (int i = 0; i < 6; ++i) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian* fh : frameHessians) {
      nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(i);
      nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
      nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
    }
    nullspaces_x0_pre.emplace_back(nullspace_x0);
    nullspaces_pose.emplace_back(nullspace_x0);
  }
  for (int i = 0; i < 2; ++i) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian* fh : frameHessians) {
      nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) =
          fh->nullspaces_affine.col(i).head<2>();
      nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE;
      nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
    }
    nullspaces_x0_pre.emplace_back(nullspace_x0);
    if (i == 0) {
      nullspaces_affA.emplace_back(nullspace_x0);
    } else if (i == 1) {
      nullspaces_affB.emplace_back(nullspace_x0);
    }
  }

  VecX nullspace_x0(n);
  nullspace_x0.setZero();
  for (FrameHessian* fh : frameHessians) {
    nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;
    nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
    nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
  }
  nullspaces_x0_pre.emplace_back(nullspace_x0);
  nullspaces_scale.emplace_back(nullspace_x0);

  return nullspaces_x0_pre;
}

}  // dso
