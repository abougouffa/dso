#pragma once

#include "util/global_calib.h"
#include "util/num_type.h"
#include "util/settings.h"

#define SCALE_F 50.0f  // scale for fx, fy
#define SCALE_C 50.0f  // scale for cx, cy
#define SCALE_W 1.0f

#define SCALE_F_INVERSE (1.0f / SCALE_F)
#define SCALE_C_INVERSE (1.0f / SCALE_C)
#define SCALE_W_INVERSE (1.0f / SCALE_W)

namespace dso {

class CalibHessian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CalibHessian() {
    VecC initial_value = VecC::Zero();
    initial_value[0] = fxG[0];
    initial_value[1] = fyG[0];
    initial_value[2] = cxG[0];
    initial_value[3] = cyG[0];

    setValueScaled(initial_value);
    value_zero = value;
    value_minus_value_zero.setZero();

    ++instanceCounter;
    for (int i = 0; i < 256; ++i) {
      // set gamma function to identity
      Binv[i] = B[i] = i;
    }
  };

  ~CalibHessian() { --instanceCounter; }

  // normal mode: use the optimized parameters everywhere!
  float& fxl() { return value_scaledf[0]; }
  float& fyl() { return value_scaledf[1]; }
  float& cxl() { return value_scaledf[2]; }
  float& cyl() { return value_scaledf[3]; }
  float& fxli() { return value_scaledi[0]; }
  float& fyli() { return value_scaledi[1]; }
  float& cxli() { return value_scaledi[2]; }
  float& cyli() { return value_scaledi[3]; }

  void setValue(const VecC& value) {
    // [0-3: Kl, 4-7: Kr, 8-12: l2r]
    this->value = value;
    value_scaled[0] = SCALE_F * value[0];
    value_scaled[1] = SCALE_F * value[1];
    value_scaled[2] = SCALE_C * value[2];
    value_scaled[3] = SCALE_C * value[3];

    this->value_scaledf = this->value_scaled.cast<float>();
    this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
    this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
    this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
    this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];
    this->value_minus_value_zero = this->value - this->value_zero;
  };

  void setValueScaled(const VecC& value_scaled) {
    this->value_scaled = value_scaled;
    this->value_scaledf = this->value_scaled.cast<float>();
    value[0] = SCALE_F_INVERSE * value_scaled[0];
    value[1] = SCALE_F_INVERSE * value_scaled[1];
    value[2] = SCALE_C_INVERSE * value_scaled[2];
    value[3] = SCALE_C_INVERSE * value_scaled[3];

    this->value_minus_value_zero = this->value - this->value_zero;
    this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
    this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
    this->value_scaledi[2] = -this->value_scaledf[2] / this->value_scaledf[0];
    this->value_scaledi[3] = -this->value_scaledf[3] / this->value_scaledf[1];
  };

  float getBGradOnly(float color) {
    int c = color + 0.5f;
    if (c < 5) {
      c = 5;
    } else if (c > 250) {
      c = 250;
    }
    return B[c + 1] - B[c];
  }

  float getBInvGradOnly(float color) {
    int c = color + 0.5f;
    if (c < 5) {
      c = 5;
    } else if (c > 250) {
      c = 250;
    }
    return Binv[c + 1] - Binv[c];
  }

 public:
  static int instanceCounter;

  VecC value_zero;
  VecC value_scaled;
  VecCf value_scaledf;
  VecCf value_scaledi;
  VecC value;
  VecC step;
  VecC step_backup;
  VecC value_backup;
  VecC value_minus_value_zero;

  float Binv[256];
  float B[256];
};

}  // namespace dso