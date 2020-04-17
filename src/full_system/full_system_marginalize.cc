#include "full_system/full_system.h"

#include <stdio.h>
#include <algorithm>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "full_system/immature_point.h"
#include "full_system/residual_projections.h"
#include "full_system/tracker/coarse_tracker.h"
#include "io_wrapper/image_display.h"
#include "io_wrapper/output_3d_wrapper.h"
#include "optimization_backend/energy_functional/energy_functional.h"
#include "optimization_backend/energy_functional/energy_functional_structs.h"
#include "util/global_calib.h"
#include "util/global_funcs.h"

namespace dso {

void FullSystem::flagFramesForMarginalization(FrameHessian* newFH) {
  if (setting_minFrameAge > setting_maxFrames) {
    for (int i = setting_maxFrames; i < (int)frameHessians.size(); ++i) {
      FrameHessian* fh = frameHessians[i - setting_maxFrames];
      fh->flaggedForMarginalization = true;
    }
    return;
  }

  int flagged = 0;
  // Marginalize all frames that have not enough points.
  for (size_t i = 0; i < frameHessians.size(); ++i) {
    FrameHessian* fh = frameHessians[i];
    int in = fh->pointHessians.size() + fh->immaturePoints.size();
    int out =
        fh->pointHessiansMarginalized.size() + fh->pointHessiansOut.size();

    Vec2 refToFh = AffLight::fromToVecExposure(
        frameHessians.back()->ab_exposure, fh->ab_exposure,
        frameHessians.back()->aff_g2l(), fh->aff_g2l());

    if ((in < setting_minPointsRemaining * (in + out) ||
         fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow) &&
        static_cast<int>(frameHessians.size()) - flagged > setting_minFrames) {
      fh->flaggedForMarginalization = true;
      ++flagged;
    }
  }

  // marginalize one.
  if (static_cast<int>(frameHessians.size()) - flagged >= setting_maxFrames) {
    double smallestScore = 1;
    FrameHessian* toMarginalize = 0;
    FrameHessian* latest = frameHessians.back();

    for (FrameHessian* fh : frameHessians) {
      if (fh->frameID > latest->frameID - setting_minFrameAge ||
          fh->frameID == 0) {
        continue;
      }

      double distScore = 0;
      for (FrameFramePrecalc& ffh : fh->targetPrecalc) {
        if (ffh.target->frameID > latest->frameID - setting_minFrameAge + 1 ||
            ffh.target == ffh.host) {
          continue;
        }
        distScore += 1 / (1e-5 + ffh.distanceLL);
      }
      distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);

      if (distScore < smallestScore) {
        smallestScore = distScore;
        toMarginalize = fh;
      }
    }
    toMarginalize->flaggedForMarginalization = true;
    ++flagged;
  }
}

void FullSystem::marginalizeFrame(FrameHessian* frame) {
  // marginalize or remove all this frames points.

  CHECK_EQ(frame->pointHessians.size(), 0);

  ef->marginalizeFrame(frame->efFrame);

  // drop all observations of existing points in that frame.

  for (FrameHessian* fh : frameHessians) {
    if (fh == frame) {
      continue;
    }

    for (PointHessian* ph : fh->pointHessians) {
      for (size_t i = 0; i < ph->residuals.size(); ++i) {
        PointFrameResidual* r = ph->residuals[i];
        if (r->target == frame) {
          if (ph->lastResiduals[0].first == r) {
            ph->lastResiduals[0].first = 0;
          } else if (ph->lastResiduals[1].first == r) {
            ph->lastResiduals[1].first = 0;
          }

          if (r->host->frameID < r->target->frameID) {
            ++statistics_numForceDroppedResFwd;
          } else {
            ++statistics_numForceDroppedResBwd;
          }

          ef->dropResidual(r->efResidual);
          deleteOut<PointFrameResidual>(ph->residuals, i);
          break;
        }
      }
    }
  }

  {
    std::vector<FrameHessian*> v;
    v.emplace_back(frame);
    for (IOWrap::Output3DWrapper* ow : outputWrapper) {
      ow->publishKeyframes(v, true, &Hcalib);
    }
  }

  frame->shell->marginalizedAt = frameHessians.back()->shell->id;
  frame->shell->movedByOpt = frame->w2c_leftEps().norm();

  deleteOutOrder<FrameHessian>(frameHessians, frame);
  for (size_t i = 0; i < frameHessians.size(); ++i) {
    frameHessians[i]->idx = i;
  }

  setPrecalcValues();
  ef->setAdjointsF(&Hcalib);
}

}  // dso
