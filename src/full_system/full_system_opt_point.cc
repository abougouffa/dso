#include "full_system/full_system.h"

#include <math.h>
#include <stdio.h>
#include <algorithm>

#include <glog/logging.h>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "full_system/immature_point.h"
#include "io_wrapper/image_display.h"
#include "util/global_calib.h"
#include "util/global_funcs.h"

namespace dso {

PointHessian* FullSystem::optimizeImmaturePoint(
    ImmaturePoint* const point, const int minObs,
    ImmaturePointTemporaryResidual* const residuals) {
  int nres = 0;
  for (FrameHessian* fh : frameHessians) {
    if (fh != point->host) {
      residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
      residuals[nres].state_NewState = ResState::OUTLIER;
      residuals[nres].state_state = ResState::IN;
      residuals[nres].target = fh;
      ++nres;
    }
  }

  CHECK_EQ(nres + 1, frameHessians.size());

  bool print = false;  // rand()%50==0;

  float lastEnergy = 0;
  float lastHdd = 0;
  float lastbd = 0;
  float currentIdepth = (point->idepth_max + point->idepth_min) * 0.5f;

  for (int i = 0; i < nres; ++i) {
    lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals + i,
                                           lastHdd, lastbd, currentIdepth);
    residuals[i].state_state = residuals[i].state_NewState;
    residuals[i].state_energy = residuals[i].state_NewEnergy;
  }

  if (!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) {
    if (print) {
      LOG(WARNING) << "OptPoint: Not well-constrained (" << nres
                   << " res, H=" << lastHdd << "). E=" << lastEnergy
                   << ". SKIP!";
    }
    return 0;
  }

  if (print) {
    LOG(INFO) << "Activate point. " << nres << " residuals. H=" << lastHdd
              << ". Initial Energy: " << lastEnergy
              << ". Initial Id=" << currentIdepth;
  }

  float lambda = 0.1;
  for (int iteration = 0; iteration < setting_GNItsOnPointActivation;
       ++iteration) {
    float H = lastHdd;
    H *= 1 + lambda;
    float step = (1.0 / H) * lastbd;
    float newIdepth = currentIdepth - step;

    float newHdd = 0;
    float newbd = 0;
    float newEnergy = 0;
    for (int i = 0; i < nres; ++i)
      newEnergy += point->linearizeResidual(&Hcalib, 1, residuals + i, newHdd,
                                            newbd, newIdepth);

    if (!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act) {
      if (print) {
        LOG(WARNING) << "OptPoint: Not well-constrained (" << nres
                     << " res, H=" << newHdd << "). E=" << lastEnergy
                     << ". SKIP!";
      }
      return 0;
    }

    if (print) {
      const std::string status =
          (true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT";
      LOG(INFO) << status << ", iteration: " << iteration
                << " (log10(lambda): " << log10(lambda)
                << "), energy last to new: " << lastEnergy << " -> "
                << newEnergy << " (idepth " << newIdepth << ")!";
    }

    if (newEnergy < lastEnergy) {
      currentIdepth = newIdepth;
      lastHdd = newHdd;
      lastbd = newbd;
      lastEnergy = newEnergy;
      for (int i = 0; i < nres; ++i) {
        residuals[i].state_state = residuals[i].state_NewState;
        residuals[i].state_energy = residuals[i].state_NewEnergy;
      }

      lambda *= 0.5;
    } else {
      lambda *= 5;
    }

    if (fabsf(step) < 0.0001 * currentIdepth) {
      break;
    }
  }

  if (!std::isfinite(currentIdepth)) {
    LOG(ERROR) << "MAJOR ERROR! point idepth is nan after initialization "
               << currentIdepth;

    // yeah I'm like 99% sure this is OK on 32bit systems.
    return (PointHessian*)((long)(-1));
  }

  int numGoodRes = 0;
  for (int i = 0; i < nres; ++i) {
    if (residuals[i].state_state == ResState::IN) {
      ++numGoodRes;
    }
  }

  if (numGoodRes < minObs) {
    if (print) {
      LOG(WARNING) << "OptPoint: OUTLIER!";
    }

    // yeah I'm like 99% sure this is OK on 32bit systems.
    return (PointHessian*)((long)(-1));
  }

  PointHessian* p = new PointHessian(point, &Hcalib);
  if (!std::isfinite(p->energyTH)) {
    delete p;
    return (PointHessian*)((long)(-1));
  }

  p->lastResiduals[0].first = 0;
  p->lastResiduals[0].second = ResState::OOB;
  p->lastResiduals[1].first = 0;
  p->lastResiduals[1].second = ResState::OOB;
  p->setIdepthZero(currentIdepth);
  p->setIdepth(currentIdepth);
  p->setPointStatus(PointHessian::ACTIVE);

  for (int i = 0; i < nres; ++i)
    if (residuals[i].state_state == ResState::IN) {
      PointFrameResidual* r =
          new PointFrameResidual(p, p->host, residuals[i].target);
      r->state_NewEnergy = r->state_energy = 0;
      r->state_NewState = ResState::OUTLIER;
      r->setState(ResState::IN);
      p->residuals.emplace_back(r);

      if (r->target == frameHessians.back()) {
        p->lastResiduals[0].first = r;
        p->lastResiduals[0].second = ResState::IN;
      } else if (r->target == (frameHessians.size() < 2
                                   ? 0
                                   : frameHessians[frameHessians.size() - 2])) {
        p->lastResiduals[1].first = r;
        p->lastResiduals[1].second = ResState::IN;
      }
    }

  if (print) {
    LOG(INFO) << "point activated!";
  }

  ++statistics_numActivatedPoints;
  return p;
}
}
