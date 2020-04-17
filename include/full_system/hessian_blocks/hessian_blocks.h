#pragma once

#include "full_system/hessian_blocks/calib_hessian.h"
#include "full_system/hessian_blocks/frame_frame_pre_calc.h"
#include "full_system/hessian_blocks/frame_hessian.h"
#include "full_system/hessian_blocks/point_hessian.h"
#include "util/num_type.h"

namespace dso {

inline Vec2 affFromTo(const Vec2& from, const Vec2& to) {
  // contains affine parameters as XtoWorld.
  return Vec2(from[0] / to[0], (from[1] - to[1]) / to[0]);
}

}  // dso