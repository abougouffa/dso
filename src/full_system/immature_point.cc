#include "full_system/immature_point.h"

#include <glog/logging.h>

#include "full_system/residual_projections.h"
#include "util/frame_shell.h"

namespace dso {
ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian* host_, float type,
                             CalibHessian* HCalib)
    : u(u_),
      v(v_),
      host(host_),
      my_type(type),
      idepth_min(0),
      idepth_max(NAN),
      lastTraceStatus(IPS_UNINITIALIZED) {
  gradH.setZero();

  for (int idx = 0; idx < patternNum; ++idx) {
    int dx = patternP[idx][0];
    int dy = patternP[idx][1];

    Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy, wG[0]);

    color[idx] = ptc[0];
    if (!std::isfinite(color[idx])) {
      energyTH = NAN;
      return;
    }

    gradH += ptc.tail<2>() * ptc.tail<2>().transpose();

    // 梯度越大, weight越小(因为潜在误差越大)
    weights[idx] =
        sqrtf(setting_outlierTHSumComponent /
              (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
  }

  energyTH = patternNum * setting_outlierTH;
  energyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

  idepth_GT = 0;
  quality = 10000;
}

ImmaturePoint::~ImmaturePoint() {}

/*
 * returns
 * * OOB -> Out of Bound, point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
ImmaturePointStatus ImmaturePoint::traceOn(FrameHessian* frame,
                                           const Mat33f& hostToFrame_KRKi,
                                           const Vec3f& hostToFrame_Kt,
                                           const Vec2f& hostToFrame_affine,
                                           CalibHessian* HCalib,
                                           bool debugPrint) {
  if (lastTraceStatus == ImmaturePointStatus::IPS_OOB) {
    return lastTraceStatus;
  }

  debugPrint = false;  // rand()%100==0;

  // 极线搜索范围的最大值
  float maxPixSearch = (wG[0] + hG[0]) * setting_maxPixSearch;

  if (debugPrint) {
    LOG(INFO) << "trace pt (" << u << " " << v << ") from frame "
              << host->shell->id << " to " << frame->shell->id
              << ". Inv depth range " << idepth_min << " -> " << idepth_max
              << ". t " << hostToFrame_Kt[0] << " " << hostToFrame_Kt[1] << " "
              << hostToFrame_Kt[2];
  }

  // const float stepsize = 1.0;        // stepsize for initial discrete search.
  // const int GNIterations = 3;        // max GN iterations
  // const float GNThreshold = 0.1;     // GN stop after this stepsize.
  // const float extraSlackOnTH = 1.2;  // for energy-based outlier check, be
  //                                    // slightly more relaxed by this factor.
  // const float slackInterval =
  //     0.8;  // if pixel-interval is smaller than this, leave it be.
  // const float minImprovementFactor =
  //     2;  // if pixel-interval is smaller than this, leave it be.

  // =========== project min and max. return if one of them is OOB ===========

  // 计算host中(u, v)点在当前frame的位置
  Vec3f pr = hostToFrame_KRKi * Vec3f(u, v, 1);
  Vec3f ptpMin = pr + hostToFrame_Kt * idepth_min;
  float uMin = ptpMin[0] / ptpMin[2];
  float vMin = ptpMin[1] / ptpMin[2];

  if (!(uMin > 4 && vMin > 4 && uMin < wG[0] - 5 && vMin < hG[0] - 5)) {
    // 在image border之外, out of bound
    if (debugPrint) {
      LOG(INFO) << "OOB uMin " << u << " " << v << " - " << uMin << " " << vMin
                << " " << ptpMin[2] << " (inv depth  " << idepth_min << "-"
                << idepth_max << ")!";
    }
    lastTraceUV = Vec2f(-1, -1);
    lastTracePixelInterval = 0;
    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
  }

  float dist;  // 极线搜索的范围
  float uMax;
  float vMax;
  Vec3f ptpMax;
  if (std::isfinite(idepth_max)) {
    ptpMax = pr + hostToFrame_Kt * idepth_max;
    uMax = ptpMax[0] / ptpMax[2];
    vMax = ptpMax[1] / ptpMax[2];

    if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5)) {
      // 在image border之外, out of bound
      if (debugPrint) {
        LOG(INFO) << "OOB uMax  " << u << " " << v << " - " << uMax << " "
                  << vMax << "!";
      }
      lastTraceUV = Vec2f(-1, -1);
      lastTracePixelInterval = 0;
      return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }

    // Check their distance. everything below 2px is OK (-> skip).
    dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
    dist = sqrtf(dist);
    if (dist < setting_trace_slackInterval) {
      if (debugPrint) {
        LOG(INFO) << "TOO CERTAIN ALREADY (dist " << dist << ")!";
      }

      lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
      lastTracePixelInterval = dist;
      return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
    }
    CHECK_GT(dist, 0.f);
  } else {
    dist = maxPixSearch;

    // project to arbitrary depth to get direction.
    ptpMax = pr + hostToFrame_Kt * 0.01;
    uMax = ptpMax[0] / ptpMax[2];
    vMax = ptpMax[1] / ptpMax[2];

    // direction.
    float dx = uMax - uMin;
    float dy = vMax - vMin;
    float d = 1.0f / sqrtf(dx * dx + dy * dy);

    // set to [setting_maxPixSearch].
    uMax = uMin + dist * dx * d;
    vMax = vMin + dist * dy * d;

    // may still be out!
    if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5)) {
      if (debugPrint) {
        LOG(INFO) << "OOB uMax-coarse " << uMax << " " << vMax << " "
                  << ptpMax[2] << "!";
      }
      lastTraceUV = Vec2f(-1, -1);
      lastTracePixelInterval = 0;
      return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }
    CHECK_GT(dist, 0.f);
  }

  // set OOB if scale change too big.
  if (!(idepth_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5))) {
    if (debugPrint) {
      LOG(INFO) << "OOB SCALE " << uMax << " " << vMax << " " << ptpMin[2];
    }
    lastTraceUV = Vec2f(-1, -1);
    lastTracePixelInterval = 0;
    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
  }

  // ============== Compute error-bounds on result in pixel. ==============
  // If the new interval is not at least 1/2 of the old, SKIP
  float dx = setting_trace_stepsize * (uMax - uMin);
  float dy = setting_trace_stepsize * (vMax - vMin);

  // a = (gx * dx + gy * dy)^2
  // gradient direction * epipolar line direction
  float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));

  // b = (gx * dy - gy * dx)^2
  // gradient direction * direction perpendicular to epipolar line
  float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));

  // If epipolar line parallel to gradient, then, b small, a big,
  // --> error in pixel is small
  // If epipolar line perpendicular to gradient, then, b big, a small,
  // --> error in pixel is big
  float errorInPixel = 0.2f + 0.2f * (a + b) / a;

  if (errorInPixel * setting_trace_minImprovementFactor > dist &&
      std::isfinite(idepth_max)) {
    if (debugPrint) {
      LOG(INFO) << "NO SIGNIFICANT IMPROVMENT (" << errorInPixel << ")!";
    }
    lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
    lastTracePixelInterval = dist;
    return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
  }

  if (errorInPixel > 10) {
    errorInPixel = 10;
  }

  // ============== do the discrete search ===================
  dx /= dist;  // 每次移动x方向的便宜量
  dy /= dist;  // 每次移动y方向的便宜量

  if (debugPrint) {
    LOG(INFO) << "trace pt (" << u << " " << v << ") from frame "
              << host->shell->id << " to " << frame->shell->id
              << ". Inv depth range " << idepth_min << "(" << uMin << " "
              << vMin << ") -> " << idepth_max << "(" << uMax << " " << vMax
              << ")! ErrorInPixel " << errorInPixel;
  }

  if (dist > maxPixSearch) {
    uMax = uMin + maxPixSearch * dx;
    vMax = vMin + maxPixSearch * dy;
    dist = maxPixSearch;
  }

  // 要移动的步数
  int numSteps = 1.9999f + dist / setting_trace_stepsize;
  Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2, 2>();

  float randShift = uMin * 1000 - floorf(uMin * 1000);
  float ptx = uMin - randShift * dx;
  float pty = vMin - randShift * dy;

  Vec2f rotatetPattern[MAX_RES_PER_POINT];
  for (int idx = 0; idx < patternNum; ++idx) {
    rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);
  }

  if (!std::isfinite(dx) || !std::isfinite(dy)) {
    // printf("CAUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);

    lastTracePixelInterval = 0;
    lastTraceUV = Vec2f(-1, -1);
    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
  }

  float errors[100];
  float bestU = 0, bestV = 0, bestEnergy = 1e10;
  int bestIdx = -1;
  if (numSteps >= 100) {
    numSteps = 99;
  }

  for (int i = 0; i < numSteps; ++i) {
    float energy = 0;
    for (int idx = 0; idx < patternNum; ++idx) {
      float hitColor = getInterpolatedElement31(
          frame->dI, (float)(ptx + rotatetPattern[idx][0]),
          (float)(pty + rotatetPattern[idx][1]), wG[0]);

      if (!std::isfinite(hitColor)) {
        energy += 1e5;
        continue;
      }
      float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] +
                                          hostToFrame_affine[1]);
      float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH /
                                                            fabs(residual);
      energy += hw * residual * residual * (2 - hw);
    }

    if (debugPrint) {
      LOG(INFO) << "step " << ptx << " " << pty
                << " (id 0): energy = " << energy << "!";
    }

    errors[i] = energy;
    if (energy < bestEnergy) {
      bestU = ptx;
      bestV = pty;
      bestEnergy = energy;
      bestIdx = i;
    }

    ptx += dx;
    pty += dy;
  }

  // find best score outside a +-2px radius.
  float secondBest = 1e10;
  for (int i = 0; i < numSteps; ++i) {
    if ((i < bestIdx - setting_minTraceTestRadius ||
         i > bestIdx + setting_minTraceTestRadius) &&
        errors[i] < secondBest)
      secondBest = errors[i];
  }
  float newQuality = secondBest / bestEnergy;
  if (newQuality < quality || numSteps > 10) {
    quality = newQuality;
  }

  // ============== do GN optimization ===================
  float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
  if (setting_trace_GNIterations > 0) {
    bestEnergy = 1e5;
  }
  int gnStepsGood = 0, gnStepsBad = 0;
  for (int it = 0; it < setting_trace_GNIterations; ++it) {
    float H = 1, b = 0, energy = 0;
    for (int idx = 0; idx < patternNum; ++idx) {
      Vec3f hitColor = getInterpolatedElement33(
          frame->dI, (float)(bestU + rotatetPattern[idx][0]),
          (float)(bestV + rotatetPattern[idx][1]), wG[0]);

      if (!std::isfinite((float)hitColor[0])) {
        energy += 1e5;
        continue;
      }
      float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] +
                                      hostToFrame_affine[1]);
      float dResdDist = dx * hitColor[1] + dy * hitColor[2];
      float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH /
                                                            fabs(residual);

      H += hw * dResdDist * dResdDist;
      b += hw * residual * dResdDist;
      energy +=
          weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
    }

    if (energy > bestEnergy) {
      ++gnStepsBad;

      // do a smaller step from old point.
      stepBack *= 0.5;
      bestU = uBak + stepBack * dx;
      bestV = vBak + stepBack * dy;
      if (debugPrint) {
        LOG(INFO) << "GN BACK " << it << ": E " << energy << ", H " << H
                  << ", b " << b << ". id-step " << stepBack << ". UV " << uBak
                  << " " << vBak << " -> " << bestU << " " << bestV << ".";
      }
    } else {
      ++gnStepsGood;

      float step = -gnstepsize * b / H;
      if (step < -0.5f) {
        step = -0.5;
      } else if (step > 0.5f) {
        step = 0.5;
      }

      if (!std::isfinite(step)) {
        step = 0.f;
      }

      uBak = bestU;
      vBak = bestV;
      stepBack = step;

      bestU += step * dx;
      bestV += step * dy;
      bestEnergy = energy;

      if (debugPrint) {
        LOG(INFO) << "GN step " << it << ": E " << energy << ", H " << H
                  << ", b " << b << ". id-step " << stepBack << ". UV " << uBak
                  << " " << vBak << " -> " << bestU << " " << bestV << ".";
      }
    }

    if (fabsf(stepBack) < setting_trace_GNThreshold) {
      break;
    }
  }

  // ============== detect energy-based outlier. ===================
  //	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU,
  // bestV, wG[0]);
  //	float absGrad1 =
  // getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25,
  // bestV*0.5-0.25, wG[1]);
  //	float absGrad2 =
  // getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375,
  // bestV*0.25-0.375, wG[2]);
  // if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH))
  //			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
  //		     && absGrad1*areaGradientSlackFactor <
  // host->frameGradTH*0.75f
  //			 && absGrad2*areaGradientSlackFactor <
  // host->frameGradTH*0.50f))

  if (bestEnergy >= energyTH * setting_trace_extraSlackOnTH) {
    if (debugPrint) {
      LOG(WARNING) << "OUTLIER!";
    }

    lastTracePixelInterval = 0;
    lastTraceUV = Vec2f(-1, -1);
    if (lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) {
      return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    } else {
      return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }
  }

  // ============== set new interval ===================
  if (dx * dx > dy * dy) {
    idepth_min =
        (pr[2] * (bestU - errorInPixel * dx) - pr[0]) /
        (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU - errorInPixel * dx));
    idepth_max =
        (pr[2] * (bestU + errorInPixel * dx) - pr[0]) /
        (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU + errorInPixel * dx));
  } else {
    idepth_min =
        (pr[2] * (bestV - errorInPixel * dy) - pr[1]) /
        (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV - errorInPixel * dy));
    idepth_max =
        (pr[2] * (bestV + errorInPixel * dy) - pr[1]) /
        (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV + errorInPixel * dy));
  }
  if (idepth_min > idepth_max) {
    std::swap<float>(idepth_min, idepth_max);
  }

  if (!std::isfinite(idepth_min) || !std::isfinite(idepth_max) ||
      (idepth_max < 0)) {
    // printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min,
    // idepth_max);

    lastTracePixelInterval = 0;
    lastTraceUV = Vec2f(-1, -1);
    return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
  }

  lastTracePixelInterval = 2 * errorInPixel;
  lastTraceUV = Vec2f(bestU, bestV);
  return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
}

float ImmaturePoint::getdPixdd(CalibHessian* HCalib,
                               ImmaturePointTemporaryResidual* tmpRes,
                               float idepth) {
  FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);
  const Vec3f& PRE_tTll = precalc->PRE_tTll;
  float drescale, u = 0, v = 0, new_idepth;
  float Ku, Kv;
  Vec3f KliP;

  projectPoint(this->u, this->v, idepth, 0, 0, HCalib, precalc->PRE_RTll,
               PRE_tTll, &drescale, &u, &v, &Ku, &Kv, &KliP, &new_idepth);

  float dxdd = (PRE_tTll[0] - PRE_tTll[2] * u) * HCalib->fxl();
  float dydd = (PRE_tTll[1] - PRE_tTll[2] * v) * HCalib->fyl();
  return drescale * sqrtf(dxdd * dxdd + dydd * dydd);
}

float ImmaturePoint::calcResidual(CalibHessian* HCalib,
                                  const float outlierTHSlack,
                                  ImmaturePointTemporaryResidual* tmpRes,
                                  float idepth) {
  FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

  float energyLeft = 0;
  const Eigen::Vector3f* dIl = tmpRes->target->dI;
  const Mat33f& PRE_KRKiTll = precalc->PRE_KRKiTll;
  const Vec3f& PRE_KtTll = precalc->PRE_KtTll;
  Vec2f affLL = precalc->PRE_aff_mode;

  for (int idx = 0; idx < patternNum; ++idx) {
    float Ku, Kv;
    if (!projectPoint(this->u + patternP[idx][0], this->v + patternP[idx][1],
                      idepth, PRE_KRKiTll, PRE_KtTll, &Ku, &Kv)) {
      return 1e10;
    }

    Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
    if (!std::isfinite((float)hitColor[0])) {
      return 1e10;
    }
    // if(benchmarkSpecialOption==5) hitColor =
    // (getInterpolatedElement13BiCub(tmpRes->target->I, Ku, Kv, wG[0]));

    float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

    float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH /
                                                           fabsf(residual);
    energyLeft +=
        weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
  }

  if (energyLeft > energyTH * outlierTHSlack) {
    energyLeft = energyTH * outlierTHSlack;
  }
  return energyLeft;
}

double ImmaturePoint::linearizeResidual(CalibHessian* HCalib,
                                        const float outlierTHSlack,
                                        ImmaturePointTemporaryResidual* tmpRes,
                                        float& Hdd, float& bd, float idepth) {
  if (tmpRes->state_state == ResState::OOB) {
    tmpRes->state_NewState = ResState::OOB;
    return tmpRes->state_energy;
  }

  FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

  // check OOB due to scale angle change.

  float energyLeft = 0;
  const Eigen::Vector3f* dIl = tmpRes->target->dI;
  const Mat33f& PRE_RTll = precalc->PRE_RTll;
  const Vec3f& PRE_tTll = precalc->PRE_tTll;
  // const float * const Il = tmpRes->target->I;

  Vec2f affLL = precalc->PRE_aff_mode;

  for (int idx = 0; idx < patternNum; ++idx) {
    int dx = patternP[idx][0];
    int dy = patternP[idx][1];

    float drescale, u, v, new_idepth;
    float Ku, Kv;
    Vec3f KliP;

    if (!projectPoint(this->u, this->v, idepth, dx, dy, HCalib, PRE_RTll,
                      PRE_tTll, &drescale, &u, &v, &Ku, &Kv, &KliP,
                      &new_idepth)) {
      tmpRes->state_NewState = ResState::OOB;
      return tmpRes->state_energy;
    }

    Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

    if (!std::isfinite((float)hitColor[0])) {
      tmpRes->state_NewState = ResState::OOB;
      return tmpRes->state_energy;
    }
    float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

    float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH /
                                                           fabsf(residual);
    energyLeft +=
        weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);

    // depth derivatives.
    float dxInterp = hitColor[1] * HCalib->fxl();
    float dyInterp = hitColor[2] * HCalib->fyl();
    float d_idepth =
        derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

    hw *= weights[idx] * weights[idx];

    Hdd += (hw * d_idepth) * d_idepth;
    bd += (hw * residual) * d_idepth;
  }

  if (energyLeft > energyTH * outlierTHSlack) {
    energyLeft = energyTH * outlierTHSlack;
    tmpRes->state_NewState = ResState::OUTLIER;
  } else {
    tmpRes->state_NewState = ResState::IN;
  }

  tmpRes->state_NewEnergy = energyLeft;
  return energyLeft;
}
}
