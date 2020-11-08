#pragma once

#include <glog/logging.h>

#include "util/minimal_image.h"
#include "util/num_type.h"
#include "util/settings.h"

#define SCALE_A 10.0f
#define SCALE_A_INVERSE (1.0f / SCALE_A)

#define SCALE_B 1000.0f
#define SCALE_B_INVERSE (1.0f / SCALE_B)

#define SCALE_XI_ROT 1.0f
#define SCALE_XI_ROT_INVERSE (1.0f / SCALE_XI_ROT)

#define SCALE_XI_TRANS 0.5f
#define SCALE_XI_TRANS_INVERSE (1.0f / SCALE_XI_TRANS)

namespace dso {

class FrameFramePrecalc;
class FrameHessian;
class PointHessian;
class CalibHessian;

class EFFrame;
class FrameShell;
class ImmaturePoint;

class FrameHessian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  const SE3& get_worldToCam_evalPT() const { return worldToCam_evalPT; }

  const Vec10& get_state_zero() const { return state_zero; }

  const Vec10& get_state() const { return state; }

  // return increment of state
  const Vec10& get_state_scaled() const { return state_scaled; }

  const Vec10 get_state_minus_stateZero() const {
    return get_state() - get_state_zero();
  }

  // return increment of pose
  Vec6 w2c_leftEps() const { return get_state_scaled().head<6>(); }

  AffLight aff_g2l() const {
    return AffLight(get_state_scaled()[6], get_state_scaled()[7]);
  }

  AffLight aff_g2l_0() const {
    return AffLight(get_state_zero()[6] * SCALE_A,
                    get_state_zero()[7] * SCALE_B);
  }

  void setState(const Vec10& state) {
    this->state = state;
    state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
    state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
    state_scaled[6] = SCALE_A * state[6];
    state_scaled[7] = SCALE_B * state[7];
    state_scaled[8] = SCALE_A * state[8];
    state_scaled[9] = SCALE_B * state[9];

    PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
    PRE_camToWorld = PRE_worldToCam.inverse();
  }

  void setStateScaled(const Vec10& state_scaled) {
    this->state_scaled = state_scaled;
    state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
    state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
    state[6] = SCALE_A_INVERSE * state_scaled[6];
    state[7] = SCALE_B_INVERSE * state_scaled[7];
    state[8] = SCALE_A_INVERSE * state_scaled[8];
    state[9] = SCALE_B_INVERSE * state_scaled[9];

    PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
    PRE_camToWorld = PRE_worldToCam.inverse();
  }

  void setEvalPT(const SE3& worldToCam_evalPT, const Vec10& state) {
    this->worldToCam_evalPT = worldToCam_evalPT;
    setState(state);
    setStateZero(state);
  }

  void setEvalPT_scaled(const SE3& worldToCam_evalPT, const AffLight& aff_g2l) {
    Vec10 initial_state = Vec10::Zero();
    initial_state[6] = aff_g2l.a;
    initial_state[7] = aff_g2l.b;
    this->worldToCam_evalPT = worldToCam_evalPT;
    setStateScaled(initial_state);
    setStateZero(this->get_state());
  }

  ~FrameHessian() {
    CHECK(efFrame == nullptr);
    release();
    --instanceCounter;
    for (int i = 0; i < PYR_LEVELS_USED; ++i) {
      delete[] dIp[i];
      delete[] absSquaredGrad[i];
    }

    if (debugImage != nullptr) {
      delete debugImage;
    }
  }

  FrameHessian() {
    ++instanceCounter;
    flaggedForMarginalization = false;
    frameID = -1;
    efFrame = nullptr;
    frameEnergyTH = 8 * 8 * patternNum;

    debugImage = nullptr;
  }

  Vec10 getPriorZero() { return Vec10::Zero(); }

  void setStateZero(const Vec10& state_zero);

  /** \brief Process images
   *
   *  Set intensity, gradients, sum of square gradients in every pyramid level
   */
  void makeImages(float* color, CalibHessian* HCalib);

  void release();
  Vec10 getPrior();

 public:
  EFFrame* efFrame;

  /** \brief Constant info & pre-calculated values */
  FrameShell* shell;

  /** \brief Image info of first level in pyramid
   *
   *  dI = dIp[0], the first level of pyramid
   *  dI[0]: intensity
   *  dI[1]: gradient x (gx)
   *  dI[2]: gradient y (gy)
   */
  Eigen::Vector3f* dI;

  /** \brief Image info.
   *
   *  i: pyramid level
   *  dIp[i][0]: intensity
   *  dIp[i][1]: gradient x (gx)
   *  dIp[i][2]: gradient y (gy)
   */
  Eigen::Vector3f* dIp[PYR_LEVELS];

  /** \brief Sum of squared gradients in every pyramid level
   *
   *  gx * gx + gy * gy. Only used for pixel select (histograms etc.). No NAN.
   */
  float* absSquaredGrad[PYR_LEVELS];

  /** \brief Incremental ID for keyframes only */
  int frameID;

  static int instanceCounter;
  int idx;

  // Photometric Calibration Stuff
  float frameEnergyTH;  // set dynamically depending on tracking residual

  /** \brief Image exposure time */
  float ab_exposure;

  bool flaggedForMarginalization;

  /** \brief Container of all active points. */
  std::vector<PointHessian*> pointHessians;

  /** \brief Container of all marginalized points
   *
   *  Fully marginalized, usually because point went OOB
   */
  std::vector<PointHessian*> pointHessiansMarginalized;

  /** \brief Container of all outliers */
  std::vector<PointHessian*> pointHessiansOut;

  std::vector<ImmaturePoint*> immaturePoints;

  Mat66 nullspaces_pose;
  Mat42 nullspaces_affine;
  Vec6 nullspaces_scale;

  // variable info.
  SE3 worldToCam_evalPT;

  /** \brief Initial increment (for FEJ) */
  Vec10 state_zero;

  /** \brief Current scaled increment */
  Vec10 state_scaled;

  /** \brief Current increment
   *
   *  0-5: increment for pose (translation, rotation)
   *  6-7: a, b
   *  8-9: 0
   */
  Vec10 state;

  /** \brief Increment in one optimization iteration */
  Vec10 step;
  Vec10 step_backup;
  Vec10 state_backup;

  /** \brief Precalculated Tcw (will be updated later) */
  SE3 PRE_worldToCam;

  /** \brief Precalculated Twc (will be updated later) */
  SE3 PRE_camToWorld;

  std::vector<FrameFramePrecalc, Eigen::aligned_allocator<FrameFramePrecalc>>
      targetPrecalc;
  MinimalImageB3* debugImage;
};

}  // namespace dso