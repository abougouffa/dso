#pragma once

#include <math.h>
#include <vector>

#include "full_system/initializer/pnt.h"
#include "io_wrapper/output_3d_wrapper.h"
#include "optimization_backend/accumulators/matrix_accumulators.h"
#include "util/num_type.h"
#include "util/settings.h"

namespace dso {
class CalibHessian;
class FrameHessian;
struct Pnt;

class CoarseInitializer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CoarseInitializer(int w, int h);
  ~CoarseInitializer();

  /** \brief Configure the first frame.
   *
   *  Selection and initialization of pixels in the first frame
   */
  void setFirst(CalibHessian* HCalib, FrameHessian* newFrameHessian);

  /** \brief Update the first frame using a new frame */
  bool trackFrame(FrameHessian* newFrameHessian,
                  std::vector<IOWrap::Output3DWrapper*>& wraps);

  /** \brief NOT USED */
  void calcTGrads(FrameHessian* newFrameHessian);

 public:
  int frameID;

  /** \brief Flag to decide whether fix affine paramters.
   *
   *  TRUE: NOT udate a, b
   *  FALSE: Update a, b
  */
  bool fixAffine;
  bool printDebug;

  //! High gradient points in the first frame
  Pnt* points[PYR_LEVELS];

  //! Number of points in every pyramid level
  int numPoints[PYR_LEVELS];

  //! Photometric paramters: [\f$a_{ji}\f$ \f$b_{ji}\f$]
  AffLight thisToNext_aff;

  SE3 thisToNext;

  FrameHessian* firstFrame;
  FrameHessian* newFrame;

 private:
  // Set intrinsic paramters, image width, image height for every pyramid level
  void makeK(CalibHessian* HCalib);

  /** Calculate residual, Hessians
   *
   * Calculate some Hessians blocks for Schur complement
   *
   * @return [total energy; regularizer energy; number of used points]
  */
  Vec3f calcResAndGS(int lvl, Mat88f& H_out, Vec8f& b_out, Mat88f& H_out_sc,
                     Vec8f& b_out_sc, const SE3& refToNew,
                     AffLight refToNew_aff, bool plot);

  // returns OLD NERGY, NEW ENERGY, NUM TERMS.
  Vec3f calcEC(int lvl);

  /** Smooth points' inverse depths
   *
   * Smoothing points' inverse depths according to the median of their
   * neighbors' inverse depths
   *
   * @param[in] lvl - pyramid level
  */
  void optReg(int lvl);

  /** Propagate the current level to a higher level
   *
   * Use results in the current level in pyramid to udpate the inverse depths in
   * a higher level
   *
   * @param[in] srcLvl - current level
  */
  void propagateUp(int srcLvl);

  /** Propagate the current level to a lower level
   *
   * Use results in the current level in pyramid to udpate the inverse depths in
   * a lower level
   *
   * @param[in] srcLvl - current level
  */
  void propagateDown(int srcLvl);

  /** \brief NOT USED */
  float rescale();

  /** \brief Reset the coarest level of pyramid
   *
   *  1. Set energy to Vec2f(0, 0)
   *  2. Set idepth_new to idepth
   *  3. Set isGood of a point to true if its neighbors contain good points, and
   *  set its inverse depth to the mean of its neighbors
  */
  void resetPoints(int lvl);

  /** \brief Compute increment of inverse depths and update them
   *
   *  1. Use the increment of pose and a, b to compute the increment of inverse
   *  depths
   *  2. Update the inverse depths
  */
  void doStep(int lvl, float lambda, Vec8f inc);

  /** \brief Store some variables for computation of Hessian before udpate. */
  void applyStep(int lvl);

  /** \brief NOT USED */
  void makeGradients(Eigen::Vector3f** data);

  void debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*>& wraps);

  /** \brief Compute 10 nearest neighbors and parent of a point. */
  void makeNN();

 private:
  Mat33 K[PYR_LEVELS];
  // K.inv
  Mat33 Ki[PYR_LEVELS];
  double fx[PYR_LEVELS];
  double fy[PYR_LEVELS];
  double fxi[PYR_LEVELS];
  double fyi[PYR_LEVELS];
  double cx[PYR_LEVELS];
  double cy[PYR_LEVELS];
  double cxi[PYR_LEVELS];
  double cyi[PYR_LEVELS];
  int w[PYR_LEVELS];
  int h[PYR_LEVELS];

  /** \brief true if optimization has converged */
  bool snapped;

  int snappedAt;

  /** \brief Pyramid images & levels on all levels */
  Eigen::Vector3f* dINew[PYR_LEVELS];
  Eigen::Vector3f* dIFist[PYR_LEVELS];

  /** \brief Weight matrix */
  Eigen::DiagonalMatrix<float, 8> wM;

  /** \brief Temporary buffers for H and b.
   *
   *  0-7: sum(dd * dp).
   *    8: sum(res*dd).
   *    9: 1/(1+sum(dd*dd))=inverse hessian entry.
  */
  Vec10f* JbBuffer;

  /** \brief Temporary buffers for H and b.
   *
   *  0-7: sum(dd * dp).
   *    8: sum(res*dd).
   *    9: 1/(1+sum(dd*dd))=inverse hessian entry.
  */
  Vec10f* JbBuffer_new;

  /** \brief Compute the top left of Hessian H11 */
  Accumulator9 acc9;

  /** \brief Schur complement of inverse depths (H12 * H22 * H12^T)
  */
  Accumulator9 acc9SC;

  /** \brief NOT USED */
  Vec3f dGrads[PYR_LEVELS];

  float alphaK;
  float alphaW;
  float regWeight;
  float couplingWeight;
};

}  // dso
