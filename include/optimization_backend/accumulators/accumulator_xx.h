#pragma once

#include "util/num_type.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

template <int i, int j>
class AccumulatorXX {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline void initialize() {
    A.setZero();
    A1k.setZero();
    A1m.setZero();
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish() {
    shiftUp(true);
    num = numIn1 + numIn1k + numIn1m;
  }

  inline void update(const Eigen::Matrix<float, i, 1>& L,
                     const Eigen::Matrix<float, j, 1>& R, float w) {
    A += w * L * R.transpose();
    ++numIn1;
    shiftUp(false);
  }

 public:
  Eigen::Matrix<float, i, j> A;
  Eigen::Matrix<float, i, j> A1k;
  Eigen::Matrix<float, i, j> A1m;
  size_t num;

 private:
  float numIn1, numIn1k, numIn1m;

  void shiftUp(bool force) {
    if (numIn1 > 1000 || force) {
      A1k += A;
      A.setZero();
      numIn1k += numIn1;
      numIn1 = 0;
    }
    if (numIn1k > 1000 || force) {
      A1m += A1k;
      A1k.setZero();
      numIn1m += numIn1k;
      numIn1k = 0;
    }
  }
};

}  // dso