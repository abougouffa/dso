#include "optimization_backend/energy_functional/ef_residual.h"

#include "full_system/residuals.h"
#include "optimization_backend/energy_functional/ef_point.h"
#include "optimization_backend/energy_functional/energy_functional.h"

namespace dso {

void EFResidual::takeDataF() {
  std::swap<RawResidualJacobian*>(J, data->J);

  // (\frac{\partial r_{i}}{\partial \mathbf{p}_{j}})^{T} \frac{\partial
  // r_{ji}}{\partial \rho_{i}}
  Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;

  for (int i = 0; i < 6; ++i) {
    JpJdF[i] = J->Jpdxi[0][i] * JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];
  }

  JpJdF.segment<2>(6) = J->JabJIdx * J->Jpdd;
}

void EFResidual::fixLinearizationF(EnergyFunctional* ef) {
  Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];

  // compute Jp*delta
  __m128 Jp_delta_x =
      _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>()) + J->Jpdc[0].dot(ef->cDeltaF) +
                  J->Jpdd[0] * point->deltaF);
  __m128 Jp_delta_y =
      _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>()) + J->Jpdc[1].dot(ef->cDeltaF) +
                  J->Jpdd[1] * point->deltaF);
  __m128 delta_a = _mm_set1_ps((float)(dp[6]));
  __m128 delta_b = _mm_set1_ps((float)(dp[7]));

  for (int i = 0; i < patternNum; i += 4) {
    // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
    __m128 rtz = _mm_load_ps(((float*)&J->resF) + i);
    rtz = _mm_sub_ps(
        rtz, _mm_mul_ps(_mm_load_ps(((float*)(J->JIdx)) + i), Jp_delta_x));
    rtz = _mm_sub_ps(
        rtz, _mm_mul_ps(_mm_load_ps(((float*)(J->JIdx + 1)) + i), Jp_delta_y));
    rtz = _mm_sub_ps(rtz,
                     _mm_mul_ps(_mm_load_ps(((float*)(J->JabF)) + i), delta_a));
    rtz = _mm_sub_ps(
        rtz, _mm_mul_ps(_mm_load_ps(((float*)(J->JabF + 1)) + i), delta_b));
    _mm_store_ps(((float*)&res_toZeroF) + i, rtz);
  }

  isLinearized = true;
}

}  // dso