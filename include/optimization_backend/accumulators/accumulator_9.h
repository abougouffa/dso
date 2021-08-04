#pragma once

#include <glog/logging.h>

#include "util/num_type.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

class Accumulator9 {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline void initialize() {
    H.setZero();
    b.setZero();
    // 由于Hessian为对角阵, 因此只存储一半的元素45, 而不是全部元素 9*9 = 81
    // 所谓9是姿态扰动6维, 光度系数2维, 逆深度1维
    // 1+2+3+...+9 = 45, 仅保存一半
    memset(SSEData, 0, sizeof(float) * 4 * 45);
    memset(SSEData1k, 0, sizeof(float) * 4 * 45);
    memset(SSEData1m, 0, sizeof(float) * 4 * 45);
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish() {
    H.setZero();
    shiftUp(true);
    CHECK_EQ(numIn1, 0.f);
    CHECK_EQ(numIn1k, 0.f);
    int idx = 0;
    for (int r = 0; r < 9; ++r) {
      for (int c = r; c < 9; ++c) {
        float d = SSEData1m[idx + 0] + SSEData1m[idx + 1] + SSEData1m[idx + 2] +
                  SSEData1m[idx + 3];
        H(r, c) = H(c, r) = d;
        idx += 4;
      }
    }
    CHECK_EQ(idx, 4 * 45);
  }

  // 计算H11右上方的值, 一次累加进去4个数
  void updateSSE(const __m128 J0, const __m128 J1, const __m128 J2,
                 const __m128 J3, const __m128 J4, const __m128 J5,
                 const __m128 J6, const __m128 J7, const __m128 J8);

  void updateSSE_eighted(const __m128 J0, const __m128 J1, const __m128 J2,
                         const __m128 J3, const __m128 J4, const __m128 J5,
                         const __m128 J6, const __m128 J7, const __m128 J8,
                         const __m128 w);

  // 计算H11右上方的值, 一次只加进去1个数
  void updateSingle(const float J0, const float J1, const float J2,
                    const float J3, const float J4, const float J5,
                    const float J6, const float J7, const float J8,
                    int off = 0);

  void updateSingleWeighted(float J0, float J1, float J2, float J3, float J4,
                            float J5, float J6, float J7, float J8, float w,
                            int off = 0);

 public:
  Mat99f H;
  Vec9f b;
  size_t num;

 private:
  void shiftUp(bool force);

 private:
  EIGEN_ALIGN16 float SSEData[4 * 45];
  EIGEN_ALIGN16 float SSEData1k[4 * 45];
  EIGEN_ALIGN16 float SSEData1m[4 * 45];
  float numIn1, numIn1k, numIn1m;
};

}  // dso