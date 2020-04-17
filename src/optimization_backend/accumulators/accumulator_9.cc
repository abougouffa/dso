#include "optimization_backend/accumulators/accumulator_9.h"

namespace dso {

void Accumulator9::updateSSE(const __m128 J0, const __m128 J1, const __m128 J2,
                             const __m128 J3, const __m128 J4, const __m128 J5,
                             const __m128 J6, const __m128 J7,
                             const __m128 J8) {
  float* pt = SSEData;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J0)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J1)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J2)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J3)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J4)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J8)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J1)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J2)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J3)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J4)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J8)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J2)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J3)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J4)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J8)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J3)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J4)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J8)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J4)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J8)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J8)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J8)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J8)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J8)));
  pt += 4;

  num += 4;
  ++numIn1;
  shiftUp(false);
}

void Accumulator9::updateSSE_eighted(const __m128 J0, const __m128 J1,
                                     const __m128 J2, const __m128 J3,
                                     const __m128 J4, const __m128 J5,
                                     const __m128 J6, const __m128 J7,
                                     const __m128 J8, const __m128 w) {
  float* pt = SSEData;

  __m128 J0w = _mm_mul_ps(J0, w);
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J0)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J1)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J2)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J3)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J4)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J8)));
  pt += 4;

  __m128 J1w = _mm_mul_ps(J1, w);
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J1)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J2)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J3)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J4)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J8)));
  pt += 4;

  __m128 J2w = _mm_mul_ps(J2, w);
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J2)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J3)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J4)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J8)));
  pt += 4;

  __m128 J3w = _mm_mul_ps(J3, w);
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J3)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J4)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J8)));
  pt += 4;

  __m128 J4w = _mm_mul_ps(J4, w);
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J4)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J8)));
  pt += 4;

  __m128 J5w = _mm_mul_ps(J5, w);
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J8)));
  pt += 4;

  __m128 J6w = _mm_mul_ps(J6, w);
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J8)));
  pt += 4;

  __m128 J7w = _mm_mul_ps(J7, w);
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7w, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7w, J8)));
  pt += 4;

  __m128 J8w = _mm_mul_ps(J8, w);
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8w, J8)));
  pt += 4;

  num += 4;
  ++numIn1;
  shiftUp(false);
}

void Accumulator9::updateSingle(const float J0, const float J1, const float J2,
                                const float J3, const float J4, const float J5,
                                const float J6, const float J7, const float J8,
                                int off) {
  float* pt = SSEData + off;
  *pt += J0 * J0;
  pt += 4;
  *pt += J1 * J0;
  pt += 4;
  *pt += J2 * J0;
  pt += 4;
  *pt += J3 * J0;
  pt += 4;
  *pt += J4 * J0;
  pt += 4;
  *pt += J5 * J0;
  pt += 4;
  *pt += J6 * J0;
  pt += 4;
  *pt += J7 * J0;
  pt += 4;
  *pt += J8 * J0;
  pt += 4;

  *pt += J1 * J1;
  pt += 4;
  *pt += J2 * J1;
  pt += 4;
  *pt += J3 * J1;
  pt += 4;
  *pt += J4 * J1;
  pt += 4;
  *pt += J5 * J1;
  pt += 4;
  *pt += J6 * J1;
  pt += 4;
  *pt += J7 * J1;
  pt += 4;
  *pt += J8 * J1;
  pt += 4;

  *pt += J2 * J2;
  pt += 4;
  *pt += J3 * J2;
  pt += 4;
  *pt += J4 * J2;
  pt += 4;
  *pt += J5 * J2;
  pt += 4;
  *pt += J6 * J2;
  pt += 4;
  *pt += J7 * J2;
  pt += 4;
  *pt += J8 * J2;
  pt += 4;

  *pt += J3 * J3;
  pt += 4;
  *pt += J4 * J3;
  pt += 4;
  *pt += J5 * J3;
  pt += 4;
  *pt += J6 * J3;
  pt += 4;
  *pt += J7 * J3;
  pt += 4;
  *pt += J8 * J3;
  pt += 4;

  *pt += J4 * J4;
  pt += 4;
  *pt += J5 * J4;
  pt += 4;
  *pt += J6 * J4;
  pt += 4;
  *pt += J7 * J4;
  pt += 4;
  *pt += J8 * J4;
  pt += 4;

  *pt += J5 * J5;
  pt += 4;
  *pt += J6 * J5;
  pt += 4;
  *pt += J7 * J5;
  pt += 4;
  *pt += J8 * J5;
  pt += 4;

  *pt += J6 * J6;
  pt += 4;
  *pt += J7 * J6;
  pt += 4;
  *pt += J8 * J6;
  pt += 4;

  *pt += J7 * J7;
  pt += 4;
  *pt += J8 * J7;
  pt += 4;

  *pt += J8 * J8;
  pt += 4;

  ++num;
  ++numIn1;
  shiftUp(false);
}

void Accumulator9::updateSingleWeighted(float J0, float J1, float J2, float J3,
                                        float J4, float J5, float J6, float J7,
                                        float J8, float w, int off) {
  float* pt = SSEData + off;
  *pt += J0 * J0 * w;
  pt += 4;
  J0 *= w;
  *pt += J1 * J0;
  pt += 4;
  *pt += J2 * J0;
  pt += 4;
  *pt += J3 * J0;
  pt += 4;
  *pt += J4 * J0;
  pt += 4;
  *pt += J5 * J0;
  pt += 4;
  *pt += J6 * J0;
  pt += 4;
  *pt += J7 * J0;
  pt += 4;
  *pt += J8 * J0;
  pt += 4;

  *pt += J1 * J1 * w;
  pt += 4;
  J1 *= w;
  *pt += J2 * J1;
  pt += 4;
  *pt += J3 * J1;
  pt += 4;
  *pt += J4 * J1;
  pt += 4;
  *pt += J5 * J1;
  pt += 4;
  *pt += J6 * J1;
  pt += 4;
  *pt += J7 * J1;
  pt += 4;
  *pt += J8 * J1;
  pt += 4;

  *pt += J2 * J2 * w;
  pt += 4;
  J2 *= w;
  *pt += J3 * J2;
  pt += 4;
  *pt += J4 * J2;
  pt += 4;
  *pt += J5 * J2;
  pt += 4;
  *pt += J6 * J2;
  pt += 4;
  *pt += J7 * J2;
  pt += 4;
  *pt += J8 * J2;
  pt += 4;

  *pt += J3 * J3 * w;
  pt += 4;
  J3 *= w;
  *pt += J4 * J3;
  pt += 4;
  *pt += J5 * J3;
  pt += 4;
  *pt += J6 * J3;
  pt += 4;
  *pt += J7 * J3;
  pt += 4;
  *pt += J8 * J3;
  pt += 4;

  *pt += J4 * J4 * w;
  pt += 4;
  J4 *= w;
  *pt += J5 * J4;
  pt += 4;
  *pt += J6 * J4;
  pt += 4;
  *pt += J7 * J4;
  pt += 4;
  *pt += J8 * J4;
  pt += 4;

  *pt += J5 * J5 * w;
  pt += 4;
  J5 *= w;
  *pt += J6 * J5;
  pt += 4;
  *pt += J7 * J5;
  pt += 4;
  *pt += J8 * J5;
  pt += 4;

  *pt += J6 * J6 * w;
  pt += 4;
  J6 *= w;
  *pt += J7 * J6;
  pt += 4;
  *pt += J8 * J6;
  pt += 4;

  *pt += J7 * J7 * w;
  pt += 4;
  J7 *= w;
  *pt += J8 * J7;
  pt += 4;

  *pt += J8 * J8 * w;
  pt += 4;

  ++num;
  ++numIn1;
  shiftUp(false);
}

void Accumulator9::shiftUp(bool force) {
  if (numIn1 > 1000 || force) {
    for (int i = 0; i < 45; ++i) {
      _mm_store_ps(SSEData1k + 4 * i,
                   _mm_add_ps(_mm_load_ps(SSEData + 4 * i),
                              _mm_load_ps(SSEData1k + 4 * i)));
    }
    numIn1k += numIn1;
    numIn1 = 0;
    memset(SSEData, 0, sizeof(float) * 4 * 45);
  }

  if (numIn1k > 1000 || force) {
    for (int i = 0; i < 45; ++i) {
      _mm_store_ps(SSEData1m + 4 * i,
                   _mm_add_ps(_mm_load_ps(SSEData1k + 4 * i),
                              _mm_load_ps(SSEData1m + 4 * i)));
    }
    numIn1m += numIn1k;
    numIn1k = 0;
    memset(SSEData1k, 0, sizeof(float) * 4 * 45);
  }
}

}  // dso