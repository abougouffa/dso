#pragma once

#include <math.h>
#include <vector>

#include "optimization_backend/accumulators/matrix_accumulators.h"
#include "util/index_thread_reduce.h"
#include "util/num_type.h"

namespace dso {

class EFPoint;
class EnergyFunctional;

class AccumulatedTopHessianSSE {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline AccumulatedTopHessianSSE() {
    for (int tid = 0; tid < NUM_THREADS; ++tid) {
      nres[tid] = 0;
      acc[tid] = 0;
      nframes[tid] = 0;
    }
  };

  inline ~AccumulatedTopHessianSSE() {
    for (int tid = 0; tid < NUM_THREADS; ++tid) {
      if (acc[tid] != 0) {
        delete[] acc[tid];
      }
    }
  };

  //! Reset the accumulator and related variables to zero
  /*!
    Construct an accumulator to compute Hessian wrt. frames, and set initial
    residual to zero

    @param[in] nFrames - number of frames currently in energy function
    @param[in] tid     - id of some containers to store our results
    @param[in] min     - not used
    @param[in] max     - not used
    @param[in] stats   - not used
  */
  void setZero(const int nFrames, const int min = 0, const int max = 1,
               Vec10 *const stats = nullptr, const int tid = 0);

  void stitchDouble(MatXX &H, VecX &b, const EnergyFunctional *const EF,
                    bool usePrior, bool useDelta, int tid = 0);

  //! TODO
  /*!
    @param[in] mode - [0]: active, [1]: linearized, [2]: marginalize
    @param[in] p    -
    @param[in] ef   -
    @param[in] tid  -
  */
  template <int mode>
  void addPoint(EFPoint *const p, const EnergyFunctional *const ef,
                const int tid = 0);

  //! TODO
  /*!
    @param[in] red       - sth for multi threading
    @param[in] EF        -
    @param[in] usePrior  -
    @param[in] MT        - whether use multi threads
    @param[out] H        - Hessian matrix (size = (n * (6 + 2) + 4)^2)
    @param[out] b        - b              (size = n * (6 + 2) + 4)
  */
  void stitchDoubleMT(IndexThreadReduce<Vec10> *red, MatXX &H, VecX &b,
                      const EnergyFunctional *const EF, const bool usePrior,
                      const bool MT);

  int nframes[NUM_THREADS];

  EIGEN_ALIGN16 AccumulatorApprox *acc[NUM_THREADS];

  int nres[NUM_THREADS];

  template <int mode>
  void addPointsInternal(std::vector<EFPoint *> *points,
                         const EnergyFunctional *const ef, int min = 0,
                         int max = 1, Vec10 *stats = 0, int tid = 0) {
    for (int i = min; i < max; ++i) {
      addPoint<mode>((*points)[i], ef, tid);
    }
  }

private:
  //! TODO
  /*!
    @param[in] EF        -
    @param[in] usePrior  -
    @param[in] min       -
    @param[in] max       -
    @param[in] stats     -
    @param[in] tid       -
    @param[out] H        - Hessian matrix (size = (n * (6 + 2) + 4)^2)
    @param[out] b        - b              (size = n * (6 + 2) + 4)
  */
  void stitchDoubleInternal(MatXX *H, VecX *b, const EnergyFunctional *const EF,
                            bool usePrior, int min, int max, Vec10 *stats,
                            int tid);
};
} // namespace dso
