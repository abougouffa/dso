#pragma once

#include "util/num_type.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

// 模板匹配？
class Accumulator11 {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline void initialize() {
    A = 0;
    memset(SSEData, 0, sizeof(float) * 4 * 1);
    memset(SSEData1k, 0, sizeof(float) * 4 * 1);
    memset(SSEData1m, 0, sizeof(float) * 4 * 1);
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish() {
    shiftUp(true);
    A = SSEData1m[0 + 0] + SSEData1m[0 + 1] + SSEData1m[0 + 2] +
        SSEData1m[0 + 3];
  }

  inline void updateSingle(const float val) {
    SSEData[0] += val;
    ++num;
    ++numIn1;
    shiftUp(false);
  }

  inline void updateSSE(const __m128 val) {
    _mm_store_ps(SSEData, _mm_add_ps(_mm_load_ps(SSEData), val));
    num += 4;
    ++numIn1;
    shiftUp(false);
  }

  inline void updateSingleNoShift(const float val) {
    SSEData[0] += val;
    ++num;
    ++numIn1;
  }

  inline void updateSSENoShift(const __m128 val) {
    _mm_store_ps(SSEData, _mm_add_ps(_mm_load_ps(SSEData), val));
    num += 4;
    ++numIn1;
  }

 public:
  float A;     // Total energe
  size_t num;  // Total number of points

 private:
  void shiftUp(bool force) {
    if (numIn1 > 1000 || force) {
      _mm_store_ps(SSEData1k,
                   _mm_add_ps(_mm_load_ps(SSEData), _mm_load_ps(SSEData1k)));
      numIn1k += numIn1;
      numIn1 = 0;
      memset(SSEData, 0, sizeof(float) * 4 * 1);
    }
    if (numIn1k > 1000 || force) {
      _mm_store_ps(SSEData1m,
                   _mm_add_ps(_mm_load_ps(SSEData1k), _mm_load_ps(SSEData1m)));
      numIn1m += numIn1k;
      numIn1k = 0;
      memset(SSEData1k, 0, sizeof(float) * 4 * 1);
    }
  }

 private:
  EIGEN_ALIGN16 float SSEData[4 * 1];
  EIGEN_ALIGN16 float SSEData1k[4 * 1];
  EIGEN_ALIGN16 float SSEData1m[4 * 1];
  float numIn1, numIn1k, numIn1m;
};

}  // dso