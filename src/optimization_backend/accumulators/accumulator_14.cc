#include "optimization_backend/accumulators/accumulator_14.h"

namespace dso {

void Accumulator14::updateSSE(const __m128 J0, const __m128 J1, const __m128 J2,
                              const __m128 J3, const __m128 J4, const __m128 J5,
                              const __m128 J6, const __m128 J7, const __m128 J8,
                              const __m128 J9, const __m128 J10,
                              const __m128 J11, const __m128 J12,
                              const __m128 J13) {
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
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J9)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J13)));
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
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J9)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J13)));
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
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J9)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J13)));
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
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J9)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J13)));
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
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J9)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J13)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J5)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J8)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J9)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J13)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J6)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J8)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J9)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J13)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J7)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J8)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J9)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J13)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J8)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J9)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J13)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J9)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J13)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J10)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J13)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J11, J11)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J11, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J11, J13)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J12, J12)));
  pt += 4;
  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J12, J13)));
  pt += 4;

  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J13, J13)));
  pt += 4;

  num += 4;
  ++numIn1;
  shiftUp(false);
}

void Accumulator14::updateSingle(const float J0, const float J1, const float J2,
                                 const float J3, const float J4, const float J5,
                                 const float J6, const float J7, const float J8,
                                 const float J9, const float J10,
                                 const float J11, const float J12,
                                 const float J13, int off) {
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
  *pt += J9 * J0;
  pt += 4;
  *pt += J10 * J0;
  pt += 4;
  *pt += J11 * J0;
  pt += 4;
  *pt += J12 * J0;
  pt += 4;
  *pt += J13 * J0;
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
  *pt += J9 * J1;
  pt += 4;
  *pt += J10 * J1;
  pt += 4;
  *pt += J11 * J1;
  pt += 4;
  *pt += J12 * J1;
  pt += 4;
  *pt += J13 * J1;
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
  *pt += J9 * J2;
  pt += 4;
  *pt += J10 * J2;
  pt += 4;
  *pt += J11 * J2;
  pt += 4;
  *pt += J12 * J2;
  pt += 4;
  *pt += J13 * J2;
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
  *pt += J9 * J3;
  pt += 4;
  *pt += J10 * J3;
  pt += 4;
  *pt += J11 * J3;
  pt += 4;
  *pt += J12 * J3;
  pt += 4;
  *pt += J13 * J3;
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
  *pt += J9 * J4;
  pt += 4;
  *pt += J10 * J4;
  pt += 4;
  *pt += J11 * J4;
  pt += 4;
  *pt += J12 * J4;
  pt += 4;
  *pt += J13 * J4;
  pt += 4;

  *pt += J5 * J5;
  pt += 4;
  *pt += J6 * J5;
  pt += 4;
  *pt += J7 * J5;
  pt += 4;
  *pt += J8 * J5;
  pt += 4;
  *pt += J9 * J5;
  pt += 4;
  *pt += J10 * J5;
  pt += 4;
  *pt += J11 * J5;
  pt += 4;
  *pt += J12 * J5;
  pt += 4;
  *pt += J13 * J5;
  pt += 4;

  *pt += J6 * J6;
  pt += 4;
  *pt += J7 * J6;
  pt += 4;
  *pt += J8 * J6;
  pt += 4;
  *pt += J9 * J6;
  pt += 4;
  *pt += J10 * J6;
  pt += 4;
  *pt += J11 * J6;
  pt += 4;
  *pt += J12 * J6;
  pt += 4;
  *pt += J13 * J6;
  pt += 4;

  *pt += J7 * J7;
  pt += 4;
  *pt += J8 * J7;
  pt += 4;
  *pt += J9 * J7;
  pt += 4;
  *pt += J10 * J7;
  pt += 4;
  *pt += J11 * J7;
  pt += 4;
  *pt += J12 * J7;
  pt += 4;
  *pt += J13 * J7;
  pt += 4;

  *pt += J8 * J8;
  pt += 4;
  *pt += J9 * J8;
  pt += 4;
  *pt += J10 * J8;
  pt += 4;
  *pt += J11 * J8;
  pt += 4;
  *pt += J12 * J8;
  pt += 4;
  *pt += J13 * J8;
  pt += 4;

  *pt += J9 * J9;
  pt += 4;
  *pt += J10 * J9;
  pt += 4;
  *pt += J11 * J9;
  pt += 4;
  *pt += J12 * J9;
  pt += 4;
  *pt += J13 * J9;
  pt += 4;

  *pt += J10 * J10;
  pt += 4;
  *pt += J11 * J10;
  pt += 4;
  *pt += J12 * J10;
  pt += 4;
  *pt += J13 * J10;
  pt += 4;

  *pt += J11 * J11;
  pt += 4;
  *pt += J12 * J11;
  pt += 4;
  *pt += J13 * J11;
  pt += 4;

  *pt += J12 * J12;
  pt += 4;
  *pt += J13 * J12;
  pt += 4;

  *pt += J13 * J13;
  pt += 4;

  ++num;
  ++numIn1;
  shiftUp(false);
}

void Accumulator14::shiftUp(bool force) {
  if (numIn1 > 1000 || force) {
    for (int i = 0; i < 105; ++i) {
      _mm_store_ps(SSEData1k + 4 * i,
                   _mm_add_ps(_mm_load_ps(SSEData + 4 * i),
                              _mm_load_ps(SSEData1k + 4 * i)));
    }
    numIn1k += numIn1;
    numIn1 = 0;
    memset(SSEData, 0, sizeof(float) * 4 * 105);
  }

  if (numIn1k > 1000 || force) {
    for (int i = 0; i < 105; ++i) {
      _mm_store_ps(SSEData1m + 4 * i,
                   _mm_add_ps(_mm_load_ps(SSEData1k + 4 * i),
                              _mm_load_ps(SSEData1m + 4 * i)));
    }
    numIn1m += numIn1k;
    numIn1k = 0;
    memset(SSEData1k, 0, sizeof(float) * 4 * 105);
  }
}

}  // dso