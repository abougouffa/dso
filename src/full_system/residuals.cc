#include "full_system/residuals.h"

#include <stdio.h>
#include <algorithm>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "full_system/hessian_blocks/hessian_blocks.h"
#include "full_system/residual_projections.h"
#include "io_wrapper/image_display.h"
#include "optimization_backend/energy_functional/energy_functional.h"
#include "optimization_backend/energy_functional/energy_functional_structs.h"
#include "util/global_calib.h"
#include "util/global_funcs.h"

namespace dso {
int PointFrameResidual::instanceCounter = 0;

long runningResID = 0;

PointFrameResidual::PointFrameResidual() { ++instanceCounter; }

PointFrameResidual::~PointFrameResidual() {
  CHECK(efResidual == nullptr);
  --instanceCounter;
  delete J;
}

PointFrameResidual::PointFrameResidual(PointHessian* point_,
                                       FrameHessian* host_,
                                       FrameHessian* target_)
    : point(point_), host(host_), target(target_) {
  efResidual = 0;
  ++instanceCounter;
  resetOOB();
  J = new RawResidualJacobian();
  CHECK(((long)J) % 16 == 0);

  isNew = true;
}

double PointFrameResidual::linearize(CalibHessian* const HCalib) {
  CHECK_NOTNULL(HCalib);

  state_NewEnergyWithOutlier = -1;

  if (state_state == ResState::OOB) {
    state_NewState = ResState::OOB;
    return state_energy;
  }

  FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);
  const Eigen::Vector3f* dIl = target->dI;  // [intensity gx gy]

  // K * R * K^{-1}: from host to target
  const Mat33f& PRE_KRKiTll = precalc->PRE_KRKiTll;
  const Vec3f& PRE_KtTll = precalc->PRE_KtTll;

  // Rth_0: rotation from host to target
  const Mat33f& PRE_RTll_0 = precalc->PRE_RTll_0;

  // tth_0: translation from host to target
  const Vec3f& PRE_tTll_0 = precalc->PRE_tTll_0;

  const float* const color = point->color;
  const float* const weights = point->weights;

  const Vec2f affLL = precalc->PRE_aff_mode;  // ATTENTION: FIRST ESTIMATE
  const float b0 = precalc->PRE_b0_mode;      // ATTENTION: FIRST ESTIMATE

  Vec6f d_xi_x, d_xi_y;
  Vec4f d_C_x, d_C_y;
  float d_d_x, d_d_y;
  {
    float drescale;    // inverse depth target / inverse depth host
    float new_idepth;  // pixel inverse depth wrt. target
    float u, v;        // pixel coordinates in normalized plane target
    float Ku, Kv;      // pixel coordinates in image target
    Vec3f KliP;        // pixel coordinates in normalized plane host

    // All estimates including idepth_zero_scaled are the first estimates!
    if (!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,
                      HCalib, PRE_RTll_0, PRE_tTll_0, &drescale, &u, &v, &Ku,
                      &Kv, &KliP, &new_idepth)) {
      // if point is out of boundary of image 2
      state_NewState = ResState::OOB;
      return state_energy;
    }

    centerProjectedTo = Vec3f(Ku, Kv, new_idepth);  // [Ku Kv inv_d]

    // d(pixel coordinates in image target) / d(inverse depth wrt. host)
    d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u) * SCALE_IDEPTH *
            HCalib->fxl();
    d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v) * SCALE_IDEPTH *
            HCalib->fyl();

    // d(pixel coordinates in image target) / d(fx, fy, cx, cy)
    d_C_x[2] = drescale * (PRE_RTll_0(2, 0) * u - PRE_RTll_0(0, 0));
    d_C_x[3] = HCalib->fxl() * drescale *
               (PRE_RTll_0(2, 1) * u - PRE_RTll_0(0, 1)) * HCalib->fyli();
    d_C_x[0] = KliP[0] * d_C_x[2];
    d_C_x[1] = KliP[1] * d_C_x[3];

    d_C_y[2] = HCalib->fyl() * drescale *
               (PRE_RTll_0(2, 0) * v - PRE_RTll_0(1, 0)) * HCalib->fxli();
    d_C_y[3] = drescale * (PRE_RTll_0(2, 1) * v - PRE_RTll_0(1, 1));
    d_C_y[0] = KliP[0] * d_C_y[2];
    d_C_y[1] = KliP[1] * d_C_y[3];

    d_C_x[0] = (d_C_x[0] + u) * SCALE_F;
    d_C_x[1] *= SCALE_F;
    d_C_x[2] = (d_C_x[2] + 1) * SCALE_C;
    d_C_x[3] *= SCALE_C;

    d_C_y[0] *= SCALE_F;
    d_C_y[1] = (d_C_y[1] + v) * SCALE_F;
    d_C_y[2] *= SCALE_C;
    d_C_y[3] = (d_C_y[3] + 1) * SCALE_C;

    // d(pixel cooridinates in image target) / d(relative pose Tth)
    d_xi_x[0] = new_idepth * HCalib->fxl();
    d_xi_x[1] = 0;
    d_xi_x[2] = -new_idepth * u * HCalib->fxl();
    d_xi_x[3] = -u * v * HCalib->fxl();
    d_xi_x[4] = (1 + u * u) * HCalib->fxl();
    d_xi_x[5] = -v * HCalib->fxl();

    d_xi_y[0] = 0;
    d_xi_y[1] = new_idepth * HCalib->fyl();
    d_xi_y[2] = -new_idepth * v * HCalib->fyl();
    d_xi_y[3] = -(1 + v * v) * HCalib->fyl();
    d_xi_y[4] = u * v * HCalib->fyl();
    d_xi_y[5] = u * HCalib->fyl();
  }

  // ATTENTION: These derivatives are computed using FIRST ESTIMATE
  {
    J->Jpdxi[0] = d_xi_x;
    J->Jpdxi[1] = d_xi_y;

    J->Jpdc[0] = d_C_x;
    J->Jpdc[1] = d_C_y;

    J->Jpdd[0] = d_d_x;
    J->Jpdd[1] = d_d_y;
  }

  float JIdxJIdx_00 = 0, JIdxJIdx_11 = 0, JIdxJIdx_10 = 0;
  float JabJIdx_00 = 0, JabJIdx_01 = 0, JabJIdx_10 = 0, JabJIdx_11 = 0;
  float JabJab_00 = 0, JabJab_01 = 0, JabJab_11 = 0;

  float wJI2_sum = 0;
  float energyLeft = 0;  // total energy of this point (and the whole pattern)
  for (int idx = 0; idx < patternNum; ++idx) {
    float Ku, Kv;
    if (!projectPoint(point->u + patternP[idx][0], point->v + patternP[idx][1],
                      point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, &Ku, &Kv)) {
      state_NewState = ResState::OOB;
      return state_energy;
    }

    projectedTo[idx][0] = Ku;
    projectedTo[idx][1] = Kv;

    // [intensity gx gy]
    Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
    float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

    float drdA = (color[idx] - b0);
    if (!std::isfinite(hitColor[0])) {
      state_NewState = ResState::OOB;
      return state_energy;
    }

    float w = sqrtf(
        setting_outlierTHSumComponent /
        (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
    w = 0.5f * (w + weights[idx]);

    float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH /
                                                           fabsf(residual);
    energyLeft += w * w * hw * residual * residual * (2 - hw);

    {
      if (hw < 1) {
        hw = sqrtf(hw);
      }
      hw = hw * w;

      hitColor[1] *= hw;
      hitColor[2] *= hw;

      J->resF[idx] = residual * hw;

      // ATTENTION: These two derivatives are computed using CURRENT ESTIMATE
      J->JIdx[0][idx] = hitColor[1];
      J->JIdx[1][idx] = hitColor[2];

      // ATTENTION: These two derivatives are computed using FIRST ESTIMATE
      J->JabF[0][idx] = drdA * hw;
      J->JabF[1][idx] = hw;

      JIdxJIdx_00 += hitColor[1] * hitColor[1];
      JIdxJIdx_11 += hitColor[2] * hitColor[2];
      JIdxJIdx_10 += hitColor[1] * hitColor[2];

      JabJIdx_00 += drdA * hw * hitColor[1];
      JabJIdx_01 += drdA * hw * hitColor[2];
      JabJIdx_10 += hw * hitColor[1];
      JabJIdx_11 += hw * hitColor[2];

      JabJab_00 += drdA * drdA * hw * hw;
      JabJab_01 += drdA * hw * hw;
      JabJab_11 += hw * hw;

      wJI2_sum +=
          hw * hw * (hitColor[1] * hitColor[1] + hitColor[2] * hitColor[2]);

      if (setting_affineOptModeA < 0) {
        J->JabF[0][idx] = 0;
      }
      if (setting_affineOptModeB < 0) {
        J->JabF[1][idx] = 0;
      }
    }
  }

  J->JIdx2(0, 0) = JIdxJIdx_00;
  J->JIdx2(0, 1) = JIdxJIdx_10;
  J->JIdx2(1, 0) = JIdxJIdx_10;
  J->JIdx2(1, 1) = JIdxJIdx_11;
  J->JabJIdx(0, 0) = JabJIdx_00;
  J->JabJIdx(0, 1) = JabJIdx_01;
  J->JabJIdx(1, 0) = JabJIdx_10;
  J->JabJIdx(1, 1) = JabJIdx_11;
  J->Jab2(0, 0) = JabJab_00;
  J->Jab2(0, 1) = JabJab_01;
  J->Jab2(1, 0) = JabJab_01;
  J->Jab2(1, 1) = JabJab_11;

  state_NewEnergyWithOutlier = energyLeft;

  if (energyLeft >
          std::max<float>(host->frameEnergyTH, target->frameEnergyTH) ||
      wJI2_sum < 2) {
    // if the residual is too large, set it as outlier and return the energy
    // threshold
    energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
    state_NewState = ResState::OUTLIER;
  } else {
    state_NewState = ResState::IN;
  }

  state_NewEnergy = energyLeft;  // new energy will be set in applyRes()
  return energyLeft;
}

void PointFrameResidual::debugPlot() {
  if (state_state == ResState::OOB) {
    return;
  }
  Vec3b cT = Vec3b(0, 0, 0);

  if (freeDebugParam5 == 0) {
    float rT = 20 * sqrt(state_energy / 9);
    if (rT < 0) {
      rT = 0;
    } else if (rT > 255) {
      rT = 255;
    }
    cT = Vec3b(0, 255 - rT, rT);
  } else {
    switch (state_state) {
      case ResState::IN:
        cT = Vec3b(255, 0, 0);
        break;
      case ResState::OOB:
        cT = Vec3b(255, 255, 0);
        break;
      case ResState::OUTLIER:
        cT = Vec3b(0, 0, 255);
        break;
      default:
        cT = Vec3b(255, 255, 255);
        break;
    }
  }

  for (int i = 0; i < patternNum; ++i) {
    if ((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 &&
         projectedTo[i][0] < wG[0] - 3 && projectedTo[i][1] < hG[0] - 3)) {
      target->debugImage->setPixel1((float)projectedTo[i][0],
                                    (float)projectedTo[i][1], cT);
    }
  }
}

void PointFrameResidual::applyRes(const bool copyJacobians) {
  if (copyJacobians) {
    if (state_state == ResState::OOB) {
      CHECK(!efResidual->isActiveAndIsGoodNEW);
      return;  // can never go back from OOB
    }
    if (state_NewState == ResState::IN) {
      efResidual->isActiveAndIsGoodNEW = true;
      efResidual->takeDataF();
    } else {
      efResidual->isActiveAndIsGoodNEW = false;
    }
  }

  setState(state_NewState);
  state_energy = state_NewEnergy;
}

}  // dso
