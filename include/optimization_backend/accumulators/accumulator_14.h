#pragma once

#include <glog/logging.h>

#include "util/num_type.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

class Accumulator14 {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  inline void initialize() {
    H.setZero();
    b.setZero();
    memset(SSEData, 0, sizeof(float) * 4 * 105);
    memset(SSEData1k, 0, sizeof(float) * 4 * 105);
    memset(SSEData1m, 0, sizeof(float) * 4 * 105);
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish() {
    H.setZero();
    shiftUp(true);
    CHECK_EQ(numIn1, 0.f);
    CHECK_EQ(numIn1k, 0.f);
    int idx = 0;
    for (int r = 0; r < 14; ++r)
      for (int c = r; c < 14; ++c) {
        float d = SSEData1m[idx + 0] + SSEData1m[idx + 1] + SSEData1m[idx + 2] +
                  SSEData1m[idx + 3];
        H(r, c) = H(c, r) = d;
        idx += 4;
      }
    CHECK_EQ(idx, 4 * 105);
    num = numIn1 + numIn1k + numIn1m;
  }

  void updateSSE(const __m128 J0, const __m128 J1, const __m128 J2,
                 const __m128 J3, const __m128 J4, const __m128 J5,
                 const __m128 J6, const __m128 J7, const __m128 J8,
                 const __m128 J9, const __m128 J10, const __m128 J11,
                 const __m128 J12, const __m128 J13);

  void updateSingle(const float J0, const float J1, const float J2,
                    const float J3, const float J4, const float J5,
                    const float J6, const float J7, const float J8,
                    const float J9, const float J10, const float J11,
                    const float J12, const float J13, int off = 0);

 public:
  Mat1414f H;
  Vec14f b;
  size_t num;

 private:
  void shiftUp(bool force);

 private:
  EIGEN_ALIGN16 float SSEData[4 * 105];
  EIGEN_ALIGN16 float SSEData1k[4 * 105];
  EIGEN_ALIGN16 float SSEData1m[4 * 105];
  float numIn1, numIn1k, numIn1m;
};

}  // dso