#include "full_system/tracker/coarse_distance_map.h"

#include <glog/logging.h>

#include "full_system/hessian_blocks/hessian_blocks.h"
#include "full_system/residuals.h"

namespace dso {

CoarseDistanceMap::CoarseDistanceMap(int ww, int hh) {
  fwdWarpedIDDistFinal = new float[ww * hh / 4];

  bfsList1 = new Eigen::Vector2i[ww * hh / 4];
  bfsList2 = new Eigen::Vector2i[ww * hh / 4];

  int fac = 1 << (PYR_LEVELS_USED - 1);

  coarseProjectionGrid =
      new PointFrameResidual*[2048 * (ww * hh / (fac * fac))];
  coarseProjectionGridNum = new int[ww * hh / (fac * fac)];

  w[0] = h[0] = 0;
}

CoarseDistanceMap::~CoarseDistanceMap() {
  delete[] fwdWarpedIDDistFinal;
  delete[] bfsList1;
  delete[] bfsList2;
  delete[] coarseProjectionGrid;
  delete[] coarseProjectionGridNum;
}

void CoarseDistanceMap::makeDistanceMap(
    std::vector<FrameHessian*> frameHessians, FrameHessian* frame) {
  int w1 = w[1];
  int h1 = h[1];
  int wh1 = w1 * h1;
  for (int i = 0; i < wh1; ++i) {
    fwdWarpedIDDistFinal[i] = 1000;
  }

  // make coarse tracking templates for latstRef.
  int numItems = 0;

  for (FrameHessian* fh : frameHessians) {
    if (frame == fh) {
      continue;
    }

    SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
    Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
    Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

    for (PointHessian* ph : fh->pointHessians) {
      CHECK_EQ(ph->status, PointHessian::ACTIVE);
      Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * ph->idepth_scaled;
      int u = ptp[0] / ptp[2] + 0.5f;
      int v = ptp[1] / ptp[2] + 0.5f;
      if (!(u > 0 && v > 0 && u < w[1] && v < h[1])) {
        continue;
      }
      fwdWarpedIDDistFinal[u + w1 * v] = 0;
      bfsList1[numItems] = Eigen::Vector2i(u, v);
      ++numItems;
    }
  }

  growDistBFS(numItems);
}

void CoarseDistanceMap::growDistBFS(int bfsNum) {
  CHECK_NE(w[0], 0);
  int w1 = w[1], h1 = h[1];
  for (int k = 1; k < 40; ++k) {
    int bfsNum2 = bfsNum;
    std::swap<Eigen::Vector2i*>(bfsList1, bfsList2);
    bfsNum = 0;

    if (k % 2 == 0) {
      for (int i = 0; i < bfsNum2; ++i) {
        int x = bfsList2[i][0];
        int y = bfsList2[i][1];
        if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) {
          continue;
        }
        int idx = x + y * w1;

        if (fwdWarpedIDDistFinal[idx + 1] > k) {
          fwdWarpedIDDistFinal[idx + 1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
          ++bfsNum;
        }
        if (fwdWarpedIDDistFinal[idx - 1] > k) {
          fwdWarpedIDDistFinal[idx - 1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
          ++bfsNum;
        }
        if (fwdWarpedIDDistFinal[idx + w1] > k) {
          fwdWarpedIDDistFinal[idx + w1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
          ++bfsNum;
        }
        if (fwdWarpedIDDistFinal[idx - w1] > k) {
          fwdWarpedIDDistFinal[idx - w1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
          ++bfsNum;
        }
      }
    } else {
      for (int i = 0; i < bfsNum2; ++i) {
        int x = bfsList2[i][0];
        int y = bfsList2[i][1];
        if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) {
          continue;
        }
        int idx = x + y * w1;

        if (fwdWarpedIDDistFinal[idx + 1] > k) {
          fwdWarpedIDDistFinal[idx + 1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
          ++bfsNum;
        }
        if (fwdWarpedIDDistFinal[idx - 1] > k) {
          fwdWarpedIDDistFinal[idx - 1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
          ++bfsNum;
        }
        if (fwdWarpedIDDistFinal[idx + w1] > k) {
          fwdWarpedIDDistFinal[idx + w1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
          ++bfsNum;
        }
        if (fwdWarpedIDDistFinal[idx - w1] > k) {
          fwdWarpedIDDistFinal[idx - w1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
          ++bfsNum;
        }

        if (fwdWarpedIDDistFinal[idx + 1 + w1] > k) {
          fwdWarpedIDDistFinal[idx + 1 + w1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y + 1);
          ++bfsNum;
        }
        if (fwdWarpedIDDistFinal[idx - 1 + w1] > k) {
          fwdWarpedIDDistFinal[idx - 1 + w1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y + 1);
          ++bfsNum;
        }
        if (fwdWarpedIDDistFinal[idx - 1 - w1] > k) {
          fwdWarpedIDDistFinal[idx - 1 - w1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y - 1);
          ++bfsNum;
        }
        if (fwdWarpedIDDistFinal[idx + 1 - w1] > k) {
          fwdWarpedIDDistFinal[idx + 1 - w1] = k;
          bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y - 1);
          ++bfsNum;
        }
      }
    }
  }
}

void CoarseDistanceMap::addIntoDistFinal(int u, int v) {
  if (w[0] == 0) {
    return;
  }
  bfsList1[0] = Eigen::Vector2i(u, v);
  fwdWarpedIDDistFinal[u + w[1] * v] = 0;
  growDistBFS(1);
}

void CoarseDistanceMap::makeK(CalibHessian* HCalib) {
  w[0] = wG[0];
  h[0] = hG[0];

  fx[0] = HCalib->fxl();
  fy[0] = HCalib->fyl();
  cx[0] = HCalib->cxl();
  cy[0] = HCalib->cyl();

  for (int level = 1; level < PYR_LEVELS_USED; ++level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level - 1] * 0.5;
    fy[level] = fy[level - 1] * 0.5;
    cx[level] = (cx[0] + 0.5) / (1 << level) - 0.5;
    cy[level] = (cy[0] + 0.5) / (1 << level) - 0.5;
  }

  for (int level = 0; level < PYR_LEVELS_USED; ++level) {
    K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0,
        1.0;
    Ki[level] = K[level].inverse();
    fxi[level] = Ki[level](0, 0);
    fyi[level] = Ki[level](1, 1);
    cxi[level] = Ki[level](0, 2);
    cyi[level] = Ki[level](1, 2);
  }
}

}  // dso