#include "optimization_backend/accumulated_top_hessian_sse.h"

#include <glog/logging.h>

#include "optimization_backend/energy_functional/energy_functional.h"
#include "optimization_backend/energy_functional/energy_functional_structs.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

void AccumulatedTopHessianSSE::setZero(const int nFrames, const int min,
                                       const int max, Vec10 *const stats,
                                       const int tid) {
  if (nFrames != nframes[tid]) {
    // if the number of frames has changed, reset accumulator of Hessian to
    // fit the current shape (N x N)
    if (acc[tid] != nullptr) {
      delete[] acc[tid];
    }
#if USE_XI_MODEL
    acc[tid] = new Accumulator14[nFrames * nFrames];
#else
    acc[tid] = new AccumulatorApprox[nFrames * nFrames];
#endif
  }

  for (int i = 0; i < nFrames * nFrames; ++i) {
    acc[tid][i].initialize();
  }

  nframes[tid] = nFrames;
  nres[tid] = 0;
}

template <int mode>
void AccumulatedTopHessianSSE::addPoint(EFPoint *const p,
                                        const EnergyFunctional *const ef,
                                        const int tid) {
  // 0 = active, 1 = linearized, 2 = marginalize
  CHECK(mode == 0 || mode == 1 || mode == 2);

  VecCf dc = ef->cDeltaF;
  float dd = p->deltaF;

  float bd_acc = 0;
  float Hdd_acc = 0;
  VecCf Hcd_acc = VecCf::Zero();

  for (EFResidual *r : p->residualsAll) {
    if (mode == 0) {
      if (r->isLinearized || !r->isActive()) {
        // In mode active, we only process points which are NOT linearized
        continue;
      }
    } else if (mode == 1) {
      if (!r->isLinearized || !r->isActive()) {
        // In mode linearized, we only process points which are ALREADY
        // linearized
        continue;
      }
    } else if (mode == 2) {
      if (!r->isActive()) {
        continue;
      }
      CHECK(r->isLinearized);
    }

    RawResidualJacobian *rJ = r->J;

    // Compute an id of a part of Hessian according to host id and target id
    const int htIDX = r->hostIDX + r->targetIDX * nframes[tid];
    Mat18f dp = ef->adHTdeltaF[htIDX];

    VecNRf resApprox;
    if (mode == 0) {
      resApprox = rJ->resF;
    } else if (mode == 2) {
      resApprox = r->res_toZeroF;
    } else if (mode == 1) {
      // compute Jp*delta
      __m128 Jp_delta_x = _mm_set1_ps(rJ->Jpdxi[0].dot(dp.head<6>()) +
                                      rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd);
      __m128 Jp_delta_y = _mm_set1_ps(rJ->Jpdxi[1].dot(dp.head<6>()) +
                                      rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd);
      __m128 delta_a = _mm_set1_ps((float)(dp[6]));
      __m128 delta_b = _mm_set1_ps((float)(dp[7]));

      for (int i = 0; i < patternNum; i += 4) {
        // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
        __m128 rtz = _mm_load_ps(((float *)&r->res_toZeroF) + i);
        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx)) + i),
                                         Jp_delta_x));
        rtz = _mm_add_ps(
            rtz,
            _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx + 1)) + i), Jp_delta_y));
        rtz = _mm_add_ps(
            rtz, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF)) + i), delta_a));
        rtz = _mm_add_ps(
            rtz,
            _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF + 1)) + i), delta_b));
        _mm_store_ps(((float *)&resApprox) + i, rtz);
      }
    }

    Vec2f JI_r(0, 0);  // [0]: r * (dr / du) ; [1]: r * (dr / dv)
    Vec2f Jab_r(0, 0); // [0]: r * (dr / da) ; [1]: r * (dr / db)
    float rr = 0;      // squared residual
    for (int i = 0; i < patternNum; ++i) {
      JI_r[0] += resApprox[i] * rJ->JIdx[0][i];
      JI_r[1] += resApprox[i] * rJ->JIdx[1][i];
      Jab_r[0] += resApprox[i] * rJ->JabF[0][i];
      Jab_r[1] += resApprox[i] * rJ->JabF[1][i];
      rr += resApprox[i] * resApprox[i];
    }

    acc[tid][htIDX].update(rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
                           rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
                           rJ->JIdx2(0, 0), rJ->JIdx2(0, 1), rJ->JIdx2(1, 1));

    acc[tid][htIDX].updateBotRight(rJ->Jab2(0, 0), rJ->Jab2(0, 1), Jab_r[0],
                                   rJ->Jab2(1, 1), Jab_r[1], rr);

    acc[tid][htIDX].updateTopRight(
        rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(), rJ->Jpdc[1].data(),
        rJ->Jpdxi[1].data(), rJ->JabJIdx(0, 0), rJ->JabJIdx(0, 1),
        rJ->JabJIdx(1, 0), rJ->JabJIdx(1, 1), JI_r[0], JI_r[1]);

    Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd; // (dr / dp)^T * (dr / dd)

    // r * (dr / dd)
    bd_acc += JI_r[0] * rJ->Jpdd[0] + JI_r[1] * rJ->Jpdd[1];

    // (dr / dd)^T * (dr / dd)
    Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);

    // (dr / dc)^T * (dr / dd)
    Hcd_acc += rJ->Jpdc[0] * Ji2_Jpdd[0] + rJ->Jpdc[1] * Ji2_Jpdd[1];

    ++nres[tid];
  }

  if (mode == 0) {
    p->Hdd_accAF = Hdd_acc;
    p->bd_accAF = bd_acc;
    p->Hcd_accAF = Hcd_acc;
  }
  if (mode == 1 || mode == 2) {
    p->Hdd_accLF = Hdd_acc;
    p->bd_accLF = bd_acc;
    p->Hcd_accLF = Hcd_acc;
  }
  if (mode == 2) {
    p->Hcd_accAF.setZero();
    p->Hdd_accAF = 0;
    p->bd_accAF = 0;
  }
}

template void AccumulatedTopHessianSSE::addPoint<0>(
    EFPoint *const p, const EnergyFunctional *const ef, int tid);
template void AccumulatedTopHessianSSE::addPoint<1>(
    EFPoint *const p, const EnergyFunctional *const ef, int tid);
template void AccumulatedTopHessianSSE::addPoint<2>(
    EFPoint *const p, const EnergyFunctional *const ef, int tid);

void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b,
                                            const EnergyFunctional *const EF,
                                            bool usePrior, bool useDelta,
                                            int tid) {
  H = MatXX::Zero(nframes[tid] * 8 + CPARS, nframes[tid] * 8 + CPARS);
  b = VecX::Zero(nframes[tid] * 8 + CPARS);

  for (int h = 0; h < nframes[tid]; ++h)
    for (int t = 0; t < nframes[tid]; ++t) {
      int hIdx = CPARS + h * 8;
      int tIdx = CPARS + t * 8;
      int aidx = h + nframes[tid] * t;

      acc[tid][aidx].finish();
      if (acc[tid][aidx].num == 0) {
        continue;
      }

      MatPCPC accH = acc[tid][aidx].H.cast<double>();

      H.block<8, 8>(hIdx, hIdx).noalias() += EF->adHost[aidx] *
                                             accH.block<8, 8>(CPARS, CPARS) *
                                             EF->adHost[aidx].transpose();

      H.block<8, 8>(tIdx, tIdx).noalias() += EF->adTarget[aidx] *
                                             accH.block<8, 8>(CPARS, CPARS) *
                                             EF->adTarget[aidx].transpose();

      H.block<8, 8>(hIdx, tIdx).noalias() += EF->adHost[aidx] *
                                             accH.block<8, 8>(CPARS, CPARS) *
                                             EF->adTarget[aidx].transpose();

      H.block<8, CPARS>(hIdx, 0).noalias() +=
          EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);

      H.block<8, CPARS>(tIdx, 0).noalias() +=
          EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);

      H.topLeftCorner<CPARS, CPARS>().noalias() +=
          accH.block<CPARS, CPARS>(0, 0);

      b.segment<8>(hIdx).noalias() +=
          EF->adHost[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);

      b.segment<8>(tIdx).noalias() +=
          EF->adTarget[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);

      b.head<CPARS>().noalias() += accH.block<CPARS, 1>(0, 8 + CPARS);
    }

  // ----- new: copy transposed parts.
  for (int h = 0; h < nframes[tid]; ++h) {
    int hIdx = CPARS + h * 8;
    H.block<CPARS, 8>(0, hIdx).noalias() =
        H.block<8, CPARS>(hIdx, 0).transpose();

    for (int t = h + 1; t < nframes[tid]; ++t) {
      int tIdx = CPARS + t * 8;
      H.block<8, 8>(hIdx, tIdx).noalias() +=
          H.block<8, 8>(tIdx, hIdx).transpose();
      H.block<8, 8>(tIdx, hIdx).noalias() =
          H.block<8, 8>(hIdx, tIdx).transpose();
    }
  }

  if (usePrior) {
    CHECK(useDelta);
    H.diagonal().head<CPARS>() += EF->cPrior;
    b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
    for (int h = 0; h < nframes[tid]; ++h) {
      H.diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
      b.segment<8>(CPARS + h * 8) +=
          EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
    }
  }
}

void AccumulatedTopHessianSSE::stitchDoubleInternal(
    MatXX *H, VecX *b, const EnergyFunctional *const EF, bool usePrior, int min,
    int max, Vec10 *stats, int tid) {
  int toAggregate = NUM_THREADS;
  if (tid == -1) {
    toAggregate = 1;
    tid = 0;
  } // special case: if we dont do multithreading, dont aggregate.

  if (min == max) {
    return;
  }

  for (int k = min; k < max; ++k) {
    int h = k % nframes[0];
    int t = k / nframes[0];

    int hIdx = CPARS + h * 8;
    int tIdx = CPARS + t * 8;
    int aidx = h + nframes[0] * t;

    CHECK_EQ(aidx, k);

    MatPCPC accH = MatPCPC::Zero();

    for (int tid2 = 0; tid2 < toAggregate; ++tid2) {
      acc[tid2][aidx].finish();
      if (acc[tid2][aidx].num == 0) {
        continue;
      }
      accH += acc[tid2][aidx].H.cast<double>();
    }

    H[tid].block<8, 8>(hIdx, hIdx).noalias() += EF->adHost[aidx] *
                                                accH.block<8, 8>(CPARS, CPARS) *
                                                EF->adHost[aidx].transpose();

    H[tid].block<8, 8>(tIdx, tIdx).noalias() += EF->adTarget[aidx] *
                                                accH.block<8, 8>(CPARS, CPARS) *
                                                EF->adTarget[aidx].transpose();

    H[tid].block<8, 8>(hIdx, tIdx).noalias() += EF->adHost[aidx] *
                                                accH.block<8, 8>(CPARS, CPARS) *
                                                EF->adTarget[aidx].transpose();

    H[tid].block<8, CPARS>(hIdx, 0).noalias() +=
        EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);

    H[tid].block<8, CPARS>(tIdx, 0).noalias() +=
        EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);

    H[tid].topLeftCorner<CPARS, CPARS>().noalias() +=
        accH.block<CPARS, CPARS>(0, 0);

    b[tid].segment<8>(hIdx).noalias() +=
        EF->adHost[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);

    b[tid].segment<8>(tIdx).noalias() +=
        EF->adTarget[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);

    b[tid].head<CPARS>().noalias() += accH.block<CPARS, 1>(0, CPARS + 8);
  }

  // only do this on one thread.
  if (min == 0 && usePrior) {
    H[tid].diagonal().head<CPARS>() += EF->cPrior;
    b[tid].head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
    for (int h = 0; h < nframes[tid]; ++h) {
      H[tid].diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
      b[tid].segment<8>(CPARS + h * 8) +=
          EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
    }
  }
}

void AccumulatedTopHessianSSE::stitchDoubleMT(IndexThreadReduce<Vec10> *red,
                                              MatXX &H, VecX &b,
                                              const EnergyFunctional *const EF,
                                              const bool usePrior,
                                              const bool MT) {
  // sum up, splitting by bock in square.
  if (MT) {
    MatXX Hs[NUM_THREADS];
    VecX bs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i) {
      assert(nframes[0] == nframes[i]);
      Hs[i] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
      bs[i] = VecX::Zero(nframes[0] * 8 + CPARS);
    }

    red->reduce(boost::bind(&AccumulatedTopHessianSSE::stitchDoubleInternal,
                            this, Hs, bs, EF, usePrior, boost::placeholders::_1,
                            boost::placeholders::_2, boost::placeholders::_3,
                            boost::placeholders::_4),
                0, nframes[0] * nframes[0], 0);

    // sum up results
    H = Hs[0];
    b = bs[0];

    for (int i = 1; i < NUM_THREADS; ++i) {
      H.noalias() += Hs[i];
      b.noalias() += bs[i];
      nres[0] += nres[i];
    }
  } else {
    H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
    b = VecX::Zero(nframes[0] * 8 + CPARS);
    stitchDoubleInternal(&H, &b, EF, usePrior, 0, nframes[0] * nframes[0],
                         nullptr, -1);
  }

  // make diagonal by copying over parts.
  for (int h = 0; h < nframes[0]; ++h) {
    int hIdx = CPARS + h * 8;
    H.block<CPARS, 8>(0, hIdx).noalias() =
        H.block<8, CPARS>(hIdx, 0).transpose();

    for (int t = h + 1; t < nframes[0]; ++t) {
      int tIdx = CPARS + t * 8;
      H.block<8, 8>(hIdx, tIdx).noalias() +=
          H.block<8, 8>(tIdx, hIdx).transpose();
      H.block<8, 8>(tIdx, hIdx).noalias() =
          H.block<8, 8>(hIdx, tIdx).transpose();
    }
  }
}

} // namespace dso
