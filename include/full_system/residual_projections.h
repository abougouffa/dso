#pragma once

#include <glog/logging.h>

#include "full_system/full_system.h"
#include "full_system/hessian_blocks/hessian_blocks.h"
#include "util/num_type.h"
#include "util/settings.h"

namespace dso {

/** TODO
 *  @param[in]  t
 *  @param[in]  u
 *  @param[in]  v
 *  @param[in]  dx
 *  @param[in]  dy
 *  @param[in] dxInterp
 *  @param[in] dyInterp
 *  @param[in] drescale
 *  @return
 */
EIGEN_STRONG_INLINE float derive_idepth(const Vec3f& t, const float u,
                                        const float v, const int dx,
                                        const int dy, const float dxInterp,
                                        const float dyInterp,
                                        const float drescale) {
  return (dxInterp * drescale * (t[0] - t[2] * u) +
          dyInterp * drescale * (t[1] - t[2] * v)) *
         SCALE_IDEPTH;
}

/** project a pixel from image 1 to image 2
 *  @param[in]  u_pt   pixel u in image 1
 *  @param[in]  v_pt   pixel v in image 1
 *  @param[in]  idepth inverse depth wrt. image 1
 *  @param[in]  KRKi   \f$K R_{21} K^{-1}\f$
 *  @param[in]  Kt     \f$K t_{21}\f$
 *  @param[out] Ku     pixel u in image 2
 *  @param[out] Kv     pixel v in image 2
 *  @return true if inside image, false otherwise
 */
EIGEN_STRONG_INLINE bool projectPoint(const float u_pt, const float v_pt,
                                      const float idepth, const Mat33f& KRKi,
                                      const Vec3f& Kt, float* const Ku,
                                      float* const Kv) {
  CHECK_NOTNULL(Ku);
  CHECK_NOTNULL(Kv);
  Vec3f ptp = KRKi * Vec3f(u_pt, v_pt, 1) + Kt * idepth;
  *Ku = ptp[0] / ptp[2];
  *Kv = ptp[1] / ptp[2];
  return *Ku > 1.1f && *Kv > 1.1f && *Ku < wM3G && *Kv < hM3G;
}

/** project a pixel from image 1 to image 2
 *  @param[in]  u_pt       pixel u in image 1
 *  @param[in]  v_pt       pixel v in image 1
 *  @param[in]  idepth     inverse depth wrt. image 1
 *  @param[in]  dx         x-offset of pixel coordinates (for residual pattern)
 *  @param[in]  dy         y-offset of pixel coordinates (for residual pattern)
 *  @param[in]  HCalib     intrinsic parameters
 *  @param[in]  R          relative rotation R21 from 1 to 2
 *  @param[in]  t          relative translation t21 from 1 to 2
 *  @param[out] drescale   (inverse depth 2) / (inverse depth 1)
 *  @param[out] u          pixel u in normalized plane 2
 *  @param[out] v          pixel v in normalized plane 1
 *  @param[out] Ku         pixel u in image 2
 *  @param[out] Kv         pixel v in image 2
 *  @param[out] KliP       pixel coordinates in normalized plane 1
 *  @param[out] new_idepth inverse depth wrt. image 2
 *  @return true if inside image, false otherwise
 */
EIGEN_STRONG_INLINE bool projectPoint(
    const float u_pt, const float v_pt, const float idepth, const int dx,
    const int dy, CalibHessian* const HCalib, const Mat33f& R, const Vec3f& t,
    float* const drescale, float* const u, float* const v, float* const Ku,
    float* const Kv, Vec3f* const KliP, float* const new_idepth) {
  CHECK_NOTNULL(drescale);
  CHECK_NOTNULL(u);
  CHECK_NOTNULL(v);
  CHECK_NOTNULL(Ku);
  CHECK_NOTNULL(Kv);
  CHECK_NOTNULL(KliP);
  CHECK_NOTNULL(new_idepth);

  *KliP = Vec3f((u_pt + dx - HCalib->cxl()) * HCalib->fxli(),
                (v_pt + dy - HCalib->cyl()) * HCalib->fyli(), 1);

  Vec3f ptp = R * (*KliP) + t * idepth;
  *drescale = 1.0f / ptp[2];
  *new_idepth = idepth * (*drescale);

  if (*drescale <= 0.f) {
    // point cannot be behind image 2
    return false;
  }

  *u = ptp[0] * (*drescale);
  *v = ptp[1] * (*drescale);
  *Ku = (*u) * HCalib->fxl() + HCalib->cxl();
  *Kv = (*v) * HCalib->fyl() + HCalib->cyl();

  return *Ku > 1.1f && *Kv > 1.1f && *Ku < wM3G && *Kv < hM3G;
}
}
