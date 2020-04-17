#include "full_system/initializer/coarse_initializer.h"

#include "full_system/full_system.h"
#include "full_system/hessian_blocks/hessian_blocks.h"
#include "full_system/initializer/flann_point_cloud.h"
#include "full_system/initializer/pnt.h"
#include "full_system/pixel_selector.h"
#include "full_system/pixel_selector2.h"
#include "full_system/residuals.h"
#include "util/nanoflann.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

CoarseInitializer::CoarseInitializer(int ww, int hh)
    : thisToNext_aff(0, 0), thisToNext(SE3()) {
  for (int lvl = 0; lvl < PYR_LEVELS_USED; ++lvl) {
    points[lvl] = 0;
    numPoints[lvl] = 0;
  }

  JbBuffer = new Vec10f[ww * hh];
  JbBuffer_new = new Vec10f[ww * hh];

  frameID = -1;
  fixAffine = true;
  printDebug = false;

  wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
  wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
  wM.diagonal()[6] = SCALE_A;
  wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer() {
  for (int lvl = 0; lvl < PYR_LEVELS_USED; ++lvl) {
    if (points[lvl] != 0) {
      delete[] points[lvl];
    }
  }

  delete[] JbBuffer;
  delete[] JbBuffer_new;
}

bool CoarseInitializer::trackFrame(
    FrameHessian* newFrameHessian,
    std::vector<IOWrap::Output3DWrapper*>& wraps) {
  newFrame = newFrameHessian;

  for (IOWrap::Output3DWrapper* ow : wraps) {
    ow->pushLiveFrame(newFrameHessian);
  }

  int maxIterations[] = {5, 5, 10, 30, 50};

  alphaK = 2.5 * 2.5;  //*freeDebugParam1*freeDebugParam1;
  alphaW = 150 * 150;  //*freeDebugParam2*freeDebugParam2;
  regWeight = 0.8;     //*freeDebugParam4;
  couplingWeight = 1;  //*freeDebugParam5;

  if (!snapped) {
    thisToNext.translation().setZero();  // 假设zero motion
    for (int lvl = 0; lvl < PYR_LEVELS_USED; ++lvl) {
      int npts = numPoints[lvl];
      Pnt* ptsl = points[lvl];
      for (int i = 0; i < npts; ++i) {
        ptsl[i].iR = 1;
        ptsl[i].idepth_new = 1;
        ptsl[i].lastHessian = 0;
      }
    }
  }

  SE3 refToNew_current = thisToNext;
  AffLight refToNew_aff_current = thisToNext_aff;

  if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0) {
    // coarse approximation
    // (Tong) Note: why logf() here?
    refToNew_aff_current =
        AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure), 0);
  }

  Vec3f latestRes = Vec3f::Zero();

  // coarse to fine optmization
  for (int lvl = PYR_LEVELS_USED - 1; lvl >= 0; --lvl) {
    if (lvl < PYR_LEVELS_USED - 1) {
      propagateDown(lvl + 1);
    }

    Mat88f H, Hsc;
    Vec8f b, bsc;
    resetPoints(lvl);

    //　Calculate the residual before optimization
    Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current,
                                refToNew_aff_current, false);
    applyStep(lvl);

    // 初始化优化时用到的一些值
    float lambda = 0.1;
    float eps = 1e-4;
    int fails = 0;

    if (printDebug) {
      LOG(INFO) << "lvl " << lvl << ", it " << 0 << " (l=" << lambda
                << ") INITIAL: " << sqrtf((float)(resOld[0] / resOld[2])) << "+"
                << sqrtf((float)(resOld[1] / resOld[2])) << " -> "
                << sqrtf((float)(resOld[0] / resOld[2])) << "+"
                << sqrtf((float)(resOld[1] / resOld[2])) << " ("
                << (resOld[0] + resOld[1]) / resOld[2] << "->"
                << (resOld[0] + resOld[1]) / resOld[2] << " (|inc| = 0)！";
      LOG(INFO) << refToNew_current.log().transpose() << " AFF "
                << refToNew_aff_current.vec().transpose();
    }

    int iteration = 0;
    while (true) {
      Mat88f Hl = H;
      for (int i = 0; i < 8; ++i) {
        // Variant of L-M?
        // Paper: A modified Marquardt subroutine for non-linear least squares
        Hl(i, i) *= (1 + lambda);
      }

      // Use Schur complement to compute H and b wrt. pose and [a b]
      Hl -= Hsc * (1 / (1 + lambda));
      Vec8f bl = b - bsc * (1 / (1 + lambda));

      Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
      bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));

      // Compute the increment of pose and [a b]
      Vec8f inc;
      if (fixAffine) {
        // a, b不更新
        inc.head<6>() =
            -(wM.toDenseMatrix().topLeftCorner<6, 6>() *
              (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
        inc.tail<2>().setZero();
      } else {
        // =-H^-1 * b.
        inc = -(wM * (Hl.ldlt().solve(bl)));
      }

      // Update pose and [a b]
      SE3 refToNew_new =
          SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
      AffLight refToNew_aff_new = refToNew_aff_current;
      refToNew_aff_new.a += inc[6];
      refToNew_aff_new.b += inc[7];

      // Update points' inverse depths
      doStep(lvl, lambda, inc);

      // Compute the new energy after updating variables
      Mat88f H_new, Hsc_new;
      Vec8f b_new, bsc_new;
      Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new,
                                  refToNew_new, refToNew_aff_new, false);
      Vec3f regEnergy = calcEC(lvl);

      float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
      float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

      // Check if the energy has reduced
      bool accept = eTotalOld > eTotalNew;

      if (printDebug) {
        LOG(INFO) << "lvl " << lvl << ", it " << iteration << " (l=" << lambda
                  << ") " << (accept ? "ACCEPT" : "REJECT") << ": "
                  << sqrtf((float)(resOld[0] / resOld[2])) << " + "
                  << sqrtf((float)(regEnergy[0] / regEnergy[2])) << " + "
                  << sqrtf((float)(resOld[1] / resOld[2])) << " -> "
                  << sqrtf((float)(resNew[0] / resNew[2])) << " + "
                  << sqrtf((float)(regEnergy[1] / regEnergy[2])) << " + "
                  << sqrtf((float)(resNew[1] / resNew[2])) << "("
                  << eTotalOld / resNew[2] << "->" << eTotalNew / resNew[2]
                  << ") (|inc| = " << inc.norm() << ")!";

        LOG(INFO) << refToNew_new.log().transpose() << " AFF "
                  << refToNew_aff_new.vec().transpose();
      }

      if (accept) {
        if (resNew[1] == alphaK * numPoints[lvl]) {
          // Optimization already converged
          snapped = true;
        }
        H = H_new;
        b = b_new;
        Hsc = Hsc_new;
        bsc = bsc_new;
        resOld = resNew;
        refToNew_aff_current = refToNew_aff_new;
        refToNew_current = refToNew_new;
        applyStep(lvl);
        optReg(lvl);
        lambda *= 0.5;
        fails = 0;
        if (lambda < 0.0001) {
          lambda = 0.0001;
        }
      } else {
        ++fails;
        lambda *= 4;
        if (lambda > 10000) {
          lambda = 10000;
        }
      }

      if (inc.norm() <= eps || iteration >= maxIterations[lvl] || fails >= 2) {
        Mat88f H, Hsc;
        Vec8f b, bsc;
        break;
      }

      ++iteration;
    }
    latestRes = resOld;
  }

  thisToNext = refToNew_current;
  thisToNext_aff = refToNew_aff_current;

  // Propagate the result from bottom to up again
  for (int i = 0; i < PYR_LEVELS_USED - 1; ++i) {
    propagateUp(i);
  }

  ++frameID;
  if (!snapped) {
    snappedAt = 0;
  }

  if (snapped && snappedAt == 0) {
    snappedAt = frameID;
  }

  debugPlot(0, wraps);

  // After the first convergence, we need to successfully track 5 frames more
  return snapped && frameID > snappedAt + 5;
}

void CoarseInitializer::debugPlot(
    int lvl, std::vector<IOWrap::Output3DWrapper*>& wraps) {
  bool needCall = false;
  for (IOWrap::Output3DWrapper* ow : wraps) {
    needCall = needCall || ow->needPushDepthImage();
  }
  if (!needCall) {
    return;
  }

  int wl = w[lvl], hl = h[lvl];
  Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

  MinimalImageB3 iRImg(wl, hl);

  for (int i = 0; i < wl * hl; ++i) {
    iRImg.at(i) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);
  }

  int npts = numPoints[lvl];

  float nid = 0, sid = 0;
  for (int i = 0; i < npts; ++i) {
    Pnt* point = points[lvl] + i;
    if (point->isGood) {
      ++nid;
      sid += point->iR;
    }
  }
  float fac = nid / sid;

  for (int i = 0; i < npts; ++i) {
    Pnt* point = points[lvl] + i;

    if (!point->isGood) {
      iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0));
    } else {
      iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f,
                      makeRainbow3B(point->iR * fac));
    }
  }

  // IOWrap::displayImage("idepth-R", &iRImg, false);
  for (IOWrap::Output3DWrapper* ow : wraps) {
    ow->pushDepthImage(&iRImg);
  }
}

Vec3f CoarseInitializer::calcResAndGS(int lvl, Mat88f& H_out, Vec8f& b_out,
                                      Mat88f& H_out_sc, Vec8f& b_out_sc,
                                      const SE3& refToNew,
                                      AffLight refToNew_aff, bool plot) {
  int wl = w[lvl], hl = h[lvl];

  // colorXxx[0]: intensity
  // colorXxx[1]: gradient x (gx)
  // colorXxx[2]: gradient y (gy)
  Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
  Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

  // R * K^{-1}
  Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();

  Vec3f t = refToNew.translation().cast<float>();

  // Since we uses logf() before, we need to use exp() here
  // This ensures the coefficients are largen than 0
  Eigen::Vector2f r2new_aff =
      Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

  // intrinsic parameters in this level
  float fxl = fx[lvl];
  float fyl = fy[lvl];
  float cxl = cx[lvl];
  float cyl = cy[lvl];

  Accumulator11 E;
  acc9.initialize();
  E.initialize();

  int npts = numPoints[lvl];
  Pnt* ptsl = points[lvl];
  for (int i = 0; i < npts; ++i) {
    Pnt* point = ptsl + i;

    point->maxstep = 1e10;
    if (!point->isGood) {
      E.updateSingle(point->energy[0]);
      point->energy_new = point->energy;
      point->isGood_new = false;
      continue;
    }

    // dp0 - dp5: Derivative of residual wrt. se(3)
    // dp6 - dp7: Derivative of residual wrt. [a b]
    //        dd: Derivative of residual wrt. inverse depth
    //         r: sum of single residuals
    VecNRf dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7, dd;
    VecNRf r;
    JbBuffer_new[i].setZero();

    // sum over all residuals through residual pattern.
    bool isGood = true;
    float energy = 0;  // total energy (cost)
    for (int idx = 0; idx < patternNum; ++idx) {
      int dx = patternP[idx][0];
      int dy = patternP[idx][1];

      // pt[2] = inv_d_ref / inv_d_new
      Vec3f pt =
          RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;
      float u = pt[0] / pt[2];
      float v = pt[1] / pt[2];
      float Ku = fxl * u + cxl;  // u in image new
      float Kv = fyl * v + cyl;  // v in image new

      // inverse depth wrt image new
      float new_idepth = point->idepth_new / pt[2];

      if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0)) {
        // If pixel not inside image or depth not positive, break the
        // computation for this pixel
        isGood = false;
        break;
      }
      // interpolated [intensity gx gy] in image new
      Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
      // Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

      // float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];

      // interpolated [intensity] in image ref
      float rlR =
          getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

      if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])) {
        isGood = false;
        break;
      }

      // Photometric error
      // Note: Here is slightly different from the one in paper.
      // Here, we merge four coefficients into two.
      // e = I2 - (e^a * I + b)
      float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];

      // Huber kernel for a robust estimation
      float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH /
                                                            fabs(residual);
      energy += hw * residual * residual * (2 - hw);

      // (Tong) Note: I would prefer the following formulation, which makes more
      // sense
      // hw = hw * (2 - hw);
      // energy += hw * residual * residual;
      if (hw < 1) {
        // If smaller than 1, it means residual is too large, and Huber works
        hw = sqrtf(hw);
      }

      float dxdd = (t[0] - t[2] * u) / pt[2];
      float dydd = (t[1] - t[2] * v) / pt[2];
      float dxInterp = hw * hitColor[1] * fxl;
      float dyInterp = hw * hitColor[2] * fyl;
      dp0[idx] = new_idepth * dxInterp;
      dp1[idx] = new_idepth * dyInterp;
      dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp);

      dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;
      dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
      dp5[idx] = -v * dxInterp + u * dyInterp;

      dp6[idx] = -hw * r2new_aff[0] * rlR;
      dp7[idx] = -hw * 1;
      dd[idx] = dxInterp * dxdd + dyInterp * dydd;
      r[idx] = hw * residual;

      // (Tong) Why compute max step this way?
      float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
      if (maxstep < point->maxstep) {
        point->maxstep = maxstep;
      }

      // immediately compute dp*dd' and dd*dd' in JbBuffer1.
      JbBuffer_new[i][0] += dp0[idx] * dd[idx];
      JbBuffer_new[i][1] += dp1[idx] * dd[idx];
      JbBuffer_new[i][2] += dp2[idx] * dd[idx];
      JbBuffer_new[i][3] += dp3[idx] * dd[idx];
      JbBuffer_new[i][4] += dp4[idx] * dd[idx];
      JbBuffer_new[i][5] += dp5[idx] * dd[idx];
      JbBuffer_new[i][6] += dp6[idx] * dd[idx];
      JbBuffer_new[i][7] += dp7[idx] * dd[idx];
      // residual Jacobian
      JbBuffer_new[i][8] += r[idx] * dd[idx];
      JbBuffer_new[i][9] += dd[idx] * dd[idx];
    }

    if (!isGood || energy > point->outlierTH * 20) {
      // 如果该pixel的误差太大，不更新它的residual，继续下一个pixel
      E.updateSingle((float)(point->energy[0]));
      point->isGood_new = false;
      point->energy_new = point->energy;
      continue;
    }

    // Add into energy.
    E.updateSingle(energy);
    point->isGood_new = true;
    point->energy_new[0] = energy;

    // Update Hessian matrix by computing 4 floats once
    for (int i = 0; i + 3 < patternNum; i += 4) {
      // Update Hessian
      acc9.updateSSE(
          _mm_load_ps(((float*)(&dp0)) + i), _mm_load_ps(((float*)(&dp1)) + i),
          _mm_load_ps(((float*)(&dp2)) + i), _mm_load_ps(((float*)(&dp3)) + i),
          _mm_load_ps(((float*)(&dp4)) + i), _mm_load_ps(((float*)(&dp5)) + i),
          _mm_load_ps(((float*)(&dp6)) + i), _mm_load_ps(((float*)(&dp7)) + i),
          _mm_load_ps(((float*)(&r)) + i));
    }

    // If patternNum is not multiple times of 4, then we need to add the
    // remaning points one by one
    for (int i = ((patternNum >> 2) << 2); i < patternNum; ++i) {
      acc9.updateSingle((float)dp0[i], (float)dp1[i], (float)dp2[i],
                        (float)dp3[i], (float)dp4[i], (float)dp5[i],
                        (float)dp6[i], (float)dp7[i], (float)r[i]);
    }
  }

  E.finish();
  acc9.finish();

  // Calculate alpha energy, and decide if we cap it.
  // But We did NOT add anything into it?
  Accumulator11 EAlpha;
  EAlpha.initialize();
  for (int i = 0; i < npts; ++i) {
    Pnt* point = ptsl + i;
    if (!point->isGood_new) {
      E.updateSingle((float)(point->energy[1]));
    } else {
      point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1);
      E.updateSingle((float)(point->energy_new[1]));
    }
  }
  EAlpha.finish();
  // (Tong) EAlpha.A always zero, Bug???
  float alphaEnergy =
      alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts);

  // printf("AE = %f * %f + %f\n", alphaW, EAlpha.A,
  // refToNew.translation().squaredNorm() * npts);

  // compute alpha opt.
  float alphaOpt;
  if (alphaEnergy > alphaK * npts) {
    // If alphaEnergy is big enough (translation large enough), we have the
    // chance to set "snapped" to true.
    alphaOpt = 0;
    alphaEnergy = alphaK * npts;
  } else {
    alphaOpt = alphaW;
  }

  acc9SC.initialize();
  for (int i = 0; i < npts; ++i) {
    Pnt* point = ptsl + i;
    if (!point->isGood_new) {
      continue;
    }

    point->lastHessian_new = JbBuffer_new[i][9];

    if (alphaOpt == 0) {
      JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->iR);
      JbBuffer_new[i][9] += couplingWeight;
    } else {
      JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1);
      JbBuffer_new[i][9] += alphaOpt;
    }

    // TODO 光度部分进行了尺度的缩放?
    JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]);
    acc9SC.updateSingleWeighted(JbBuffer_new[i][0], JbBuffer_new[i][1],
                                JbBuffer_new[i][2], JbBuffer_new[i][3],
                                JbBuffer_new[i][4], JbBuffer_new[i][5],
                                JbBuffer_new[i][6], JbBuffer_new[i][7],
                                JbBuffer_new[i][8], JbBuffer_new[i][9]);
  }
  acc9SC.finish();

  // printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num,
  // (int)E.num, (int)acc9SC.num*9);
  H_out = acc9.H.topLeftCorner<8, 8>();        // / acc9.num;
  b_out = acc9.H.topRightCorner<8, 1>();       // / acc9.num;
  H_out_sc = acc9SC.H.topLeftCorner<8, 8>();   // / acc9.num;
  b_out_sc = acc9SC.H.topRightCorner<8, 1>();  // / acc9.num;

  H_out(0, 0) += alphaOpt * npts;
  H_out(1, 1) += alphaOpt * npts;
  H_out(2, 2) += alphaOpt * npts;

  Vec3f tlog = refToNew.log().head<3>().cast<float>();
  b_out[0] += tlog[0] * alphaOpt * npts;
  b_out[1] += tlog[1] * alphaOpt * npts;
  b_out[2] += tlog[2] * alphaOpt * npts;

  return Vec3f(E.A, alphaEnergy, E.num);
}

float CoarseInitializer::rescale() {
  float factor = 20 * thisToNext.translation().norm();
  return factor;
}

Vec3f CoarseInitializer::calcEC(int lvl) {
  if (!snapped) {
    return Vec3f(0, 0, numPoints[lvl]);
  }
  AccumulatorX<2> E;
  E.initialize();
  int npts = numPoints[lvl];
  for (int i = 0; i < npts; ++i) {
    Pnt* point = points[lvl] + i;
    if (!point->isGood_new) {
      continue;
    }
    float rOld = (point->idepth - point->iR);
    float rNew = (point->idepth_new - point->iR);
    E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));
  }
  E.finish();

  return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
}

void CoarseInitializer::optReg(int lvl) {
  int npts = numPoints[lvl];
  Pnt* ptsl = points[lvl];
  if (!snapped) {
    // 如果优化还没有收敛
    for (int i = 0; i < npts; ++i) {
      ptsl[i].iR = 1;
    }
    return;
  }

  // smooth各个点的逆深度
  for (int i = 0; i < npts; ++i) {
    Pnt* point = ptsl + i;
    if (!point->isGood) {
      continue;
    }

    float idnn[10];
    int nnn = 0;
    for (int j = 0; j < 10; ++j) {
      if (point->neighbours[j] == -1) {
        continue;
      }
      Pnt* other = ptsl + point->neighbours[j];
      if (!other->isGood) {
        continue;
      }
      idnn[nnn] = other->iR;
      ++nnn;
    }

    if (nnn > 2) {
      std::nth_element(idnn, idnn + nnn / 2, idnn + nnn);
      point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
    }
  }
}

void CoarseInitializer::propagateUp(int srcLvl) {
  CHECK_LT(srcLvl + 1, PYR_LEVELS_USED);
  // set idepth of target

  int nptss = numPoints[srcLvl];
  int nptst = numPoints[srcLvl + 1];
  Pnt* ptss = points[srcLvl];
  Pnt* ptst = points[srcLvl + 1];

  // set to zero.
  for (int i = 0; i < nptst; ++i) {
    Pnt* parent = ptst + i;
    parent->iR = 0;
    parent->iRSumNum = 0;
  }

  for (int i = 0; i < nptss; ++i) {
    Pnt* point = ptss + i;
    if (!point->isGood) {
      continue;
    }

    Pnt* parent = ptst + point->parent;
    parent->iR += point->iR * point->lastHessian;
    parent->iRSumNum += point->lastHessian;
  }

  for (int i = 0; i < nptst; ++i) {
    Pnt* parent = ptst + i;
    if (parent->iRSumNum > 0) {
      parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
      parent->isGood = true;
    }
  }

  optReg(srcLvl + 1);
}

void CoarseInitializer::propagateDown(int srcLvl) {
  CHECK_GT(srcLvl, 0);
  // set idepth of target

  int nptst = numPoints[srcLvl - 1];  // number of points in lower level
  Pnt* ptss = points[srcLvl];         // current level
  Pnt* ptst = points[srcLvl - 1];     // points in lower level

  for (int i = 0; i < nptst; ++i) {
    Pnt* point = ptst + i;
    Pnt* parent = ptss + point->parent;

    if (!parent->isGood || parent->lastHessian < 0.1) {
      continue;
    }
    if (!point->isGood) {
      point->iR = point->idepth = point->idepth_new = parent->iR;
      point->isGood = true;
      point->lastHessian = 0;
    } else {
      float newiR = (point->iR * point->lastHessian * 2 +
                     parent->iR * parent->lastHessian) /
                    (point->lastHessian * 2 + parent->lastHessian);
      point->iR = point->idepth = point->idepth_new = newiR;
    }
  }
  optReg(srcLvl - 1);
}

void CoarseInitializer::makeGradients(Eigen::Vector3f** data) {
  for (int lvl = 1; lvl < PYR_LEVELS_USED; ++lvl) {
    int lvlm1 = lvl - 1;
    int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

    Eigen::Vector3f* dINew_l = data[lvl];
    Eigen::Vector3f* dINew_lm = data[lvlm1];

    for (int y = 0; y < hl; ++y)
      for (int x = 0; x < wl; ++x)
        dINew_l[x + y * wl][0] =
            0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] +
                     dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
                     dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                     dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

    for (int idx = wl; idx < wl * (hl - 1); ++idx) {
      dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
      dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
    }
  }
}

void CoarseInitializer::setFirst(CalibHessian* HCalib,
                                 FrameHessian* newFrameHessian) {
  makeK(HCalib);
  firstFrame = newFrameHessian;

  PixelSelector sel(w[0], h[0]);

  float* statusMap = new float[w[0] * h[0]];
  bool* statusMapB = new bool[w[0] * h[0]];

  // Point densities needed in different levels
  float densities[] = {0.03, 0.05, 0.15, 0.5, 1};
  for (int lvl = 0; lvl < PYR_LEVELS_USED; ++lvl) {
    sel.currentPotential = 3;
    int npts;
    if (lvl == 0) {
      npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0],
                          1, false, 2);
    } else {
      npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl],
                             densities[lvl] * w[0] * h[0]);
    }

    if (points[lvl] != nullptr) {
      delete[] points[lvl];
    }
    points[lvl] = new Pnt[npts];

    // set idepth map to initially 1 everywhere.
    int wl = w[lvl], hl = h[lvl];
    Pnt* pl = points[lvl];
    int nl = 0;
    for (int y = patternPadding + 1; y < hl - patternPadding - 2; ++y) {
      for (int x = patternPadding + 1; x < wl - patternPadding - 2; ++x) {
        if ((lvl != 0 && statusMapB[x + y * wl]) ||
            (lvl == 0 && statusMap[x + y * wl] != 0)) {
          // Initialize high-gradient points in every level
          pl[nl].u = x + 0.1;
          pl[nl].v = y + 0.1;
          pl[nl].idepth = 1;
          pl[nl].iR = 1;
          pl[nl].isGood = true;
          pl[nl].energy.setZero();
          pl[nl].lastHessian = 0;
          pl[nl].lastHessian_new = 0;
          pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

          // cpt[0]: intensity
          // cpt[1]: gx
          // cpt[2]: gy
          Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y * w[lvl];
          float sumGrad2 = 0;

          for (int idx = 0; idx < patternNum; ++idx) {
            // Residual pattern, 8 points
            int dx = patternP[idx][0];
            int dy = patternP[idx][1];

            // Sum of squared gradients
            float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
            sumGrad2 += absgrad;
          }

          pl[nl].outlierTH = patternNum * setting_outlierTH;

          ++nl;
          CHECK_LE(nl, npts);
        }
      }
    }

    numPoints[lvl] = nl;
  }
  delete[] statusMap;
  delete[] statusMapB;

  makeNN();

  thisToNext = SE3();
  snapped = false;
  frameID = snappedAt = 0;

  for (int i = 0; i < PYR_LEVELS_USED; ++i) {
    dGrads[i].setZero();
  }
}

void CoarseInitializer::resetPoints(int lvl) {
  Pnt* pts = points[lvl];
  int npts = numPoints[lvl];
  for (int i = 0; i < npts; ++i) {
    pts[i].energy.setZero();
    pts[i].idepth_new = pts[i].idepth;

    if (lvl == PYR_LEVELS_USED - 1 && !pts[i].isGood) {
      float snd = 0, sn = 0;
      for (int n = 0; n < 10; ++n) {
        if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) {
          continue;
        }
        snd += pts[pts[i].neighbours[n]].iR;
        sn += 1;
      }

      // If a point contains some points which are good, then this point will
      // also be set to true and its inverse depth is set to the mean of its
      // neighbors
      if (sn > 0) {
        pts[i].isGood = true;
        pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;
      }
    }
  }
}

void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc) {
  const float maxPixelStep = 0.25;
  const float idMaxStep = 1e10;
  Pnt* pts = points[lvl];
  int npts = numPoints[lvl];
  for (int i = 0; i < npts; ++i) {
    if (!pts[i].isGood) {
      continue;
    }

    float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
    float step = -b * JbBuffer[i][9] / (1 + lambda);

    float maxstep = maxPixelStep * pts[i].maxstep;
    if (maxstep > idMaxStep) {
      maxstep = idMaxStep;
    }

    if (step > maxstep) {
      step = maxstep;
    } else if (step < -maxstep) {
      step = -maxstep;
    }

    float newIdepth = pts[i].idepth + step;

    // (Tong) Why constrain the inverse depths? Why not just remove them?
    if (newIdepth < 1e-3) {
      newIdepth = 1e-3;
    } else if (newIdepth > 50) {
      newIdepth = 50;
    }
    pts[i].idepth_new = newIdepth;
  }
}

void CoarseInitializer::applyStep(int lvl) {
  Pnt* pts = points[lvl];
  int npts = numPoints[lvl];
  for (int i = 0; i < npts; ++i) {
    if (!pts[i].isGood) {
      pts[i].idepth = pts[i].idepth_new = pts[i].iR;
      continue;
    }
    pts[i].energy = pts[i].energy_new;
    pts[i].isGood = pts[i].isGood_new;
    pts[i].idepth = pts[i].idepth_new;
    pts[i].lastHessian = pts[i].lastHessian_new;
  }
  std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

void CoarseInitializer::makeK(CalibHessian* HCalib) {
  w[0] = wG[0];
  h[0] = hG[0];

  fx[0] = HCalib->fxl();
  fy[0] = HCalib->fyl();
  cx[0] = HCalib->cxl();
  cy[0] = HCalib->cyl();

  for (int level = 1; level < PYR_LEVELS_USED; ++level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
  }

  for (int level = 0; level < PYR_LEVELS_USED; ++level) {
    K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0,
        1.0;
    Ki[level] = K[level].inverse();
    fxi[level] = Ki[level](0, 0);
    fyi[level] = Ki[level](1, 1);
    cxi[level] = Ki[level](0, 2);
    cyi[level] = Ki[level](1, 2);
  }
}

void CoarseInitializer::makeNN() {
  const float NNDistFactor = 0.05;

  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>, FLANNPointcloud, 2>
      KDTree;

  // build indices
  FLANNPointcloud pcs[PYR_LEVELS];
  KDTree* indexes[PYR_LEVELS];
  for (int i = 0; i < PYR_LEVELS_USED; ++i) {
    pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
    indexes[i] =
        new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
    indexes[i]->buildIndex();
  }

  const int nn = 10;

  // find NN & parents
  for (int lvl = 0; lvl < PYR_LEVELS_USED; ++lvl) {
    Pnt* pts = points[lvl];
    int npts = numPoints[lvl];

    int ret_index[nn];
    float ret_dist[nn];
    nanoflann::KNNResultSet<float, int, int> resultSet(nn);
    nanoflann::KNNResultSet<float, int, int> resultSet1(1);

    for (int i = 0; i < npts; ++i) {
      // resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
      resultSet.init(ret_index, ret_dist);
      Vec2f pt = Vec2f(pts[i].u, pts[i].v);
      indexes[lvl]->findNeighbors(resultSet, (float*)&pt,
                                  nanoflann::SearchParams());
      int myidx = 0;
      float sumDF = 0;
      for (int k = 0; k < nn; ++k) {
        pts[i].neighbours[myidx] = ret_index[k];
        float df = expf(-ret_dist[k] * NNDistFactor);
        sumDF += df;
        pts[i].neighboursDist[myidx] = df;
        CHECK_GE(ret_index[k], 0);
        CHECK_LT(ret_index[k], npts);
        ++myidx;
      }
      for (int k = 0; k < nn; ++k) {
        pts[i].neighboursDist[k] *= 10 / sumDF;
      }

      if (lvl < PYR_LEVELS_USED - 1) {
        resultSet1.init(ret_index, ret_dist);
        pt = pt * 0.5f - Vec2f(0.25f, 0.25f);
        indexes[lvl + 1]->findNeighbors(resultSet1, (float*)&pt,
                                        nanoflann::SearchParams());

        // Set the parent in the higher level, which will be useful when
        // propagating results of optimization
        pts[i].parent = ret_index[0];
        pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor);
        CHECK_GE(ret_index[0], 0);
        CHECK_LT(ret_index[0], numPoints[lvl + 1]);
      } else {
        pts[i].parent = -1;
        pts[i].parentDist = -1;
      }
    }
  }

  for (int i = 0; i < PYR_LEVELS_USED; ++i) {
    delete indexes[i];
  }
}

}  // dso
