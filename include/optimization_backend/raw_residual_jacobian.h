#pragma once

#include "util/num_type.h"

namespace dso {
struct RawResidualJacobian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  //! [8 x 1] Individual residual of every point in a pattern
  VecNRf resF;

  //! [2 x 6] Derivative of pixel j wrt. relative pose from i to j
  Vec6f Jpdxi[2];

  //! [2 x 4] Derivative of pixel j wrt. intrinsic parameters fx, fy, cx, cy
  VecCf Jpdc[2];

  //! [2 x 1] Derivative of pixel j wrt. inverse depth i
  Vec2f Jpdd;  // 2x1

  //! [8 x 2] Derivative of residual wrt. pixel j (whole pattern 8 points)
  VecNRf JIdx[2];

  //! [8 x 2] Derivative of residual wrt. photometric parameters (whole pattern)
  VecNRf JabF[2];

  //! [2 x 2] Intermediate variable for computing Hesssian
  Mat22f JIdx2;

  //! [2 x 2] Intermediate variable for computing Hesssian
  Mat22f JabJIdx;

  //! [2 x 2] Intermediate variable for computing Hesssian
  Mat22f Jab2;
};
}
