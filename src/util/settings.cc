#include "util/settings.h"

#include <glog/logging.h>
#include <boost/bind.hpp>

namespace dso {
int PYR_LEVELS_USED = PYR_LEVELS;

/* Parameters controlling when KF's are taken */
// if !=0, takes a fixed number of KF per second.
float setting_keyframesPerSecond = 0.f;

// if true, takes as many KF's as possible (will break the system if the camera
// stays stationary)
bool setting_realTimeMaxKF = false;

double setting_maxShiftWeightT = 0.04 * (640 + 480);
double setting_maxShiftWeightR = 0. * (640 + 480);
double setting_maxShiftWeightRT = 0.02 * (640 + 480);

// general weight on threshold, the larger the more KF's are taken (e.g., 2 =
// double the amount of KF's).
double setting_kfGlobalWeight = 1.;

double setting_maxAffineWeight = 2.;

/* initial hessian values to fix unobservable dimensions / priors on affine
 * lighting parameters. */
float setting_idepthFixPrior = 50.f * 50.f;
float setting_idepthFixPriorMargFac = 600.f * 600.f;
float setting_initialRotPrior = 1e11f;
float setting_initialTransPrior = 1e10f;
float setting_initialAffBPrior = 1e14f;
float setting_initialAffAPrior = 1e14f;
float setting_initialCalibHessian = 5e9f;

/* some modes for solving the resulting linear system (e.g. orthogonalize wrt.
 * unobservable dimensions) */

//! Default: SOLVER_FIX_LAMBDA | SOLVER_ORTHOGONALIZE_X_LATER
/*! 1000 1000 0000 */
int setting_solverMode = SOLVER_FIX_LAMBDA | SOLVER_ORTHOGONALIZE_X_LATER;

double setting_solverModeDelta = 0.00001;
bool setting_forceAceptStep = true;

/* some thresholds on when to activate / marginalize points */
float setting_minIdepthH_act = 100.f;
float setting_minIdepthH_marg = 50.f;

float setting_desiredImmatureDensity = 1500.f;  // immature points per frame

// aimed total points in the active window.
float setting_desiredPointDensity = 2000.f;

// marg a frame if less than X% points remain.
float setting_minPointsRemaining = 0.05f;

// marg a frame if factor between intensities to current frame is larger than
// 1/X or X.
float setting_maxLogAffFacInWindow = 0.7f;

int setting_minFrames = 5;  // min frames in window.
int setting_maxFrames = 7;  // max frames in window.
int setting_minFrameAge = 1;
int setting_maxOptIterations = 6;  // max GN iterations.
int setting_minOptIterations = 1;  // min GN iterations.

// factor on break threshold for GN iteration (larger = break earlier)
float setting_thOptIterations = 1.2f;

/* Outlier Threshold on photometric energy */
float setting_outlierTH = 12.f * 12.f;  // higher -> less strict

// higher -> less strong gradient-based reweighting .
float setting_outlierTHSumComponent = 50.f * 50.f;

int setting_pattern = 8;  // point pattern used. DISABLED.

// factor on hessian when marginalizing, to account for inaccurate linearization
// points.
float setting_margWeightFac = 0.5f * 0.5f;

/* when to re-track a frame */
float setting_reTrackThreshold = 1.5f;  // (larger = re-track more often)

/* require some minimum number of residuals for a point to become valid */
int setting_minGoodActiveResForMarg = 3;
int setting_minGoodResForMarg = 4;

// 0 = nothing.
// 1 = apply inv. response.
// 2 = apply inv. response & remove V.
int setting_photometricCalibration = 2;
bool setting_useExposure = true;

//-1: fix. >=0: optimize (with prior, if > 0).
float setting_affineOptModeA = 1e12f;

//-1: fix. >=0: optimize (with prior, if > 0).
float setting_affineOptModeB = 1e8f;

// 1 = use original intensity for pixel selection;
// 0 = use gamma-corrected intensity.
int setting_gammaWeightsPixelSelect = 1;

float setting_huberTH = 9.f;  // Huber Threshold

// parameters controlling adaptive energy threshold computation.
float setting_frameEnergyTHConstWeight = 0.5;
float setting_frameEnergyTHN = 0.7f;
float setting_frameEnergyTHFacMedian = 1.5f;
float setting_overallEnergyTHWeight = 1.f;
float setting_coarseCutoffTH = 20.f;

// parameters controlling pixel selection
float setting_minGradHistCut = 0.5f;
float setting_minGradHistAdd = 7.f;
float setting_gradDownweightPerLevel = 0.75f;  // 梯度阈值变化的系数
bool setting_selectDirectionDistribution = true;

/* settings controling initial immature point tracking */

// max length of the ep. line segment searched during immature point tracking.
// relative to image resolution.
float setting_maxPixSearch = 0.027f;

float setting_minTraceQuality = 3.f;
int setting_minTraceTestRadius = 2;
int setting_GNItsOnPointActivation = 3;
float setting_trace_stepsize = 1.f;  // stepsize for initial discrete search.
int setting_trace_GNIterations = 3;  // max # GN iterations
float setting_trace_GNThreshold = 0.1f;  // GN stop after this stepsize.

// for energy-based outlier check, be slightly more relaxed by this factor.
float setting_trace_extraSlackOnTH = 1.2f;

// if pixel-interval is smaller than this, leave it be.
float setting_trace_slackInterval = 1.5f;

// if pixel-interval is smaller than this, leave it be.
float setting_trace_minImprovementFactor = 2.f;

// for benchmarking different undistortion settings
float benchmarkSetting_fxfyfac = 0.f;
int benchmarkSetting_width = 0;
int benchmarkSetting_height = 0;
float benchmark_varNoise = 0.f;
float benchmark_varBlurNoise = 0.f;
float benchmark_initializerSlackFactor = 1.f;
int benchmark_noiseGridsize = 3;

float freeDebugParam1 = 1.f;
float freeDebugParam2 = 1.f;
float freeDebugParam3 = 1.f;
float freeDebugParam4 = 1.f;
float freeDebugParam5 = 1.f;

bool disableReconfigure = false;
bool debugSaveImages = false;
bool multiThreading = true;
bool disableAllDisplay = false;
bool setting_onlyLogKFPoses = true;
bool setting_logStuff = true;

bool goStepByStep = false;

bool setting_render_displayCoarseTrackingFull = false;
bool setting_render_renderWindowFrames = true;
bool setting_render_plotTrackingFull = false;
bool setting_render_display3D = true;
bool setting_render_displayResidual = true;
bool setting_render_displayVideo = true;
bool setting_render_displayDepth = true;

bool setting_fullResetRequested = false;

bool setting_debugout_runquiet = false;

// not actually a setting, only some legacy stuff for coarse initializer.
// Inital: 5
int sparsityFactor = 5;

void handleKey(char k) {
  char kkk = k;
  switch (kkk) {
    case 'd':
    case 'D':
      freeDebugParam5 =
          static_cast<float>((static_cast<int>(freeDebugParam5) + 1) % 10);
      LOG(WARNING) << "new freeDebugParam5: " << freeDebugParam5 << "!";
      break;
    case 's':
    case 'S':
      freeDebugParam5 =
          static_cast<float>((static_cast<int>(freeDebugParam5) - 1 + 10) % 10);
      LOG(WARNING) << "new freeDebugParam5: " << freeDebugParam5 << "!";
      break;
  }
}

int staticPattern[10][40][2] = {
    {{0, 0},       {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},  // .
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}},

    {{0, -1},      {-1, 0},      {0, 0},       {1, 0},       {0, 1},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},  // +
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}},

    {{-1, -1},     {1, 1},       {0, 0},       {-1, 1},      {1, -1},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},  // x
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100}},

    {{-1, -1},     {-1, 0},      {-1, 1},      {-1, 0},
     {0, 0},       {0, 1},       {1, -1},      {1, 0},
     {1, 1},       {-100, -100},  // full-tight
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{0, -2},      {-1, -1},     {1, -1},      {-2, 0},
     {0, 0},       {2, 0},       {-1, 1},      {1, 1},
     {0, 2},       {-100, -100},  // full-spread-9
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{0, -2},      {-1, -1},     {1, -1},      {-2, 0},
     {0, 0},       {2, 0},       {-1, 1},      {1, 1},
     {0, 2},       {-2, -2},  // full-spread-13
     {-2, 2},      {2, -2},      {2, 2},       {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{-2, -2},     {-2, -1},     {-2, -0},     {-2, 1},
     {-2, 2},      {-1, -2},     {-1, -1},     {-1, -0},
     {-1, 1},      {-1, 2},  // full-25
     {-0, -2},     {-0, -1},     {-0, -0},     {-0, 1},
     {-0, 2},      {+1, -2},     {+1, -1},     {+1, -0},
     {+1, 1},      {+1, 2},      {+2, -2},     {+2, -1},
     {+2, -0},     {+2, 1},      {+2, 2},      {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{0, -2},      {-1, -1},     {1, -1},      {-2, 0},
     {0, 0},       {2, 0},       {-1, 1},      {1, 1},
     {0, 2},       {-2, -2},  // full-spread-21
     {-2, 2},      {2, -2},      {2, 2},       {-3, -1},
     {-3, 1},      {3, -1},      {3, 1},       {1, -3},
     {-1, -3},     {1, 3},       {-1, 3},      {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{0, -2},      {-1, -1},     {1, -1},      {-2, 0},
     {0, 0},       {2, 0},       {-1, 1},      {0, 2},
     {-100, -100}, {-100, -100},  // 8 for SSE efficiency, Residual pattern
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}, {-100, -100}, {-100, -100},
     {-100, -100}, {-100, -100}},

    {{-4, -4},     {-4, -2},     {-4, -0},     {-4, 2},
     {-4, 4},      {-2, -4},     {-2, -2},     {-2, -0},
     {-2, 2},      {-2, 4},  // full-45-SPREAD
     {-0, -4},     {-0, -2},     {-0, -0},     {-0, 2},
     {-0, 4},      {+2, -4},     {+2, -2},     {+2, -0},
     {+2, 2},      {+2, 4},      {+4, -4},     {+4, -2},
     {+4, -0},     {+4, 2},      {+4, 4},      {-200, -200},
     {-200, -200}, {-200, -200}, {-200, -200}, {-200, -200},
     {-200, -200}, {-200, -200}, {-200, -200}, {-200, -200},
     {-200, -200}, {-200, -200}, {-200, -200}, {-200, -200},
     {-200, -200}, {-200, -200}},
};

int staticPatternNum[10] = {1, 5, 5, 9, 9, 13, 25, 21, 8, 25};

int staticPatternPadding[10] = {1, 1, 1, 1, 2, 2, 2, 3, 2, 4};
}
