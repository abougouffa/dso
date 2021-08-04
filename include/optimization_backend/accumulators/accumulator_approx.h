#pragma once

#include "util/num_type.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

/*
 * computes the outer sum of 10x2 matrices, weighted with a 2x2 matrix:
 * 			H = [x y] * [a b; b c] * [x y]^T
 * (assuming x,y are column-vectors).
 * numerically robust to large sums.
 */
class AccumulatorApprox {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline void initialize() {
    memset(Data, 0, sizeof(float) * 60);
    memset(Data1k, 0, sizeof(float) * 60);
    memset(Data1m, 0, sizeof(float) * 60);

    memset(TopRight_Data, 0, sizeof(float) * 32);
    memset(TopRight_Data1k, 0, sizeof(float) * 32);
    memset(TopRight_Data1m, 0, sizeof(float) * 32);

    memset(BotRight_Data, 0, sizeof(float) * 8);
    memset(BotRight_Data1k, 0, sizeof(float) * 8);
    memset(BotRight_Data1m, 0, sizeof(float) * 8);
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  void finish();

  void updateSSE(const float* const x, const float* const y, const float a,
                 const float b, const float c);

  //! Compute the top left part of Hessian
  /*!
    Compute the Hessian ONLY related to intrinsic parameters and relative pose

    @param[in] x4 - \f$\frac{\partial u_j}{\partial\begin{bmatrix} f_{x}
    \\ f_{y} \\ c_{x} \\ c_{y}\end{bmatrix}}\f$
    @param[in] x6 - \f$\frac{\partial u_j}{\partial\boldsymbol{\xi_{ji}}}\f$
    @param[in] y4 - \f$\frac{\partial v_j}{\partial\begin{bmatrix} f_{x}
    \\ f_{y} \\ c_{x} \\ c_{y}\end{bmatrix}}\f$
    @param[in] y6 - \f$\frac{\partial v_j}{\partial\boldsymbol{\xi_{ji}}}\f$
    @param[in] a  - \f$\frac{\partial r_{ji}}{\partial u_j}\frac{\partial
    r_{ji}}{\partial u_j}\f$
    @param[in] b  - \f$\frac{\partial r_{ji}}{\partial u_j}\frac{\partial
    r_{ji}}{\partial v_j}\f$
    @param[in] c  - \f$\frac{\partial r_{ji}}{\partial v_j}\frac{\partial
    r_{ji}}{\partial v_j}\f$
  */
  void update(const float* const x4, const float* const x6,
              const float* const y4, const float* const y6, const float a,
              const float b, const float c);

  //! Compute the top right part of Hessian
  /*!
    Compute the Hessian and b related to intrinsic parameters, relative pose and
    photometric affine parameters

    @param[in] x4   - \f$\frac{\partial u_j}{\partial\begin{bmatrix} f_{x}
    \\ f_{y} \\ c_{x} \\ c_{y}\end{bmatrix}}\f$
    @param[in] x6   - \f$\frac{\partial u_j}{\partial\boldsymbol{\xi_{ji}}}\f$
    @param[in] y4   - \f$\frac{\partial v_j}{\partial\begin{bmatrix} f_{x}
    \\ f_{y} \\ c_{x} \\ c_{y}\end{bmatrix}}\f$
    @param[in] y6   - \f$\frac{\partial v_j}{\partial\boldsymbol{\xi_{ji}}}\f$
    @param[in] TR00 - \f$\frac{\partial r_{ji}}{\partial a_{ji}}\frac{\partial
    r_{ji}}{\partial u_{j}}\f$
    @param[in] TR10 - \f$\frac{\partial r_{ji}}{\partial a_{ji}}\frac{\partial
    r_{ji}}{\partial v_{j}}\f$
    @param[in] TR01 - \f$\frac{\partial r_{ji}}{\partial b_{ji}}\frac{\partial
    r_{ji}}{\partial u_{j}}\f$
    @param[in] TR11 - \f$\frac{\partial r_{ji}}{\partial b_{ji}}\frac{\partial
    r_{ji}}{\partial v_{ji}}\f$
    @param[in] TR02 - \f$r_{ji}\frac{\partial r_{ji}}{\partial u_{j}}\f$
    @param[in] TR12 - \f$r_{ji}\frac{\partial r_{ji}}{\partial v_{j}}\f$
  */

  void updateTopRight(const float* const x4, const float* const x6,
                      const float* const y4, const float* const y6,
                      const float TR00, const float TR10, const float TR01,
                      const float TR11, const float TR02, const float TR12);

  //! Compute the bottom right part of Hessian
  /*!
    Compute the Hessian and b ONLY related to photometric affine paramters

    @param[in] a00 - \f$\frac{\partial r_{ji}}{\partial a_{ji}}\frac{\partial
    r_{ji}}{\partial a_{ji}}\f$
    @param[in] a01 - \f$\frac{\partial r_{ji}}{\partial a_{ji}}\frac{\partial
    r_{ji}}{\partial b_{ji}}\f$
    @param[in] a02 - \f$r_{ji}\frac{\partial r_{ji}}{\partial a_{ji}}\f$
    @param[in] a11 - \f$\frac{\partial r_{ji}}{\partial b_{ji}}\frac{\partial
    r_{ji}}{\partial b_{ji}}\f$
    @param[in] a12 - \f$r_{ji}\frac{\partial r_{ji}}{\partial b_{ji}}\f$
    @param[in] a22 - \f$r_{ji}^{2}\f$
  */
  inline void updateBotRight(const float a00, const float a01, const float a02,
                             const float a11, const float a12,
                             const float a22) {
    BotRight_Data[0] += a00;
    BotRight_Data[1] += a01;
    BotRight_Data[2] += a02;
    BotRight_Data[3] += a11;
    BotRight_Data[4] += a12;
    BotRight_Data[5] += a22;
  }

 public:
  //! Matrix comprised of Hessian, b and residual
  /*!
    Compute the Hessian and b wrt. intrinsic parameters, relative pose, and
    photometric parameters. Note that in the real implementation, the Hessian is
    divided slightly different from the following structure.

    \f$ \mathbf{H} = \begin{bmatrix}\mathbf{J}^{T} \mathbf{J} & \mathbf{b} \\
    \mathbf{b}^{T} & r^{2} \end{bmatrix}\f$
  */
  Mat1313f H;

  //! Number of residuals
  size_t num;

 private:
  void shiftUp(bool force);

 private:
  EIGEN_ALIGN16 float Data[60];
  EIGEN_ALIGN16 float Data1k[60];
  EIGEN_ALIGN16 float Data1m[60];

  EIGEN_ALIGN16 float TopRight_Data[32];
  EIGEN_ALIGN16 float TopRight_Data1k[32];
  EIGEN_ALIGN16 float TopRight_Data1m[32];

  EIGEN_ALIGN16 float BotRight_Data[8];
  EIGEN_ALIGN16 float BotRight_Data1k[8];
  EIGEN_ALIGN16 float BotRight_Data1m[8];

  float numIn1, numIn1k, numIn1m;
};

}  // dso