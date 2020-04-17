#pragma once

#include "full_system/initializer/pnt.h"

namespace dso {

struct FLANNPointcloud {
  inline FLANNPointcloud() {
    num = 0;
    points = 0;
  }

  inline FLANNPointcloud(int n, Pnt* p) : num(n), points(p) {}

  inline size_t kdtree_get_point_count() const { return num; }

  inline float kdtree_distance(const float* p1, const size_t idx_p2,
                               size_t /*size*/) const {
    const float d0 = p1[0] - points[idx_p2].u;
    const float d1 = p1[1] - points[idx_p2].v;
    return d0 * d0 + d1 * d1;
  }

  /**
   *
   *  dim == 0: return u
   *  dim == 1: return v
  */
  inline float kdtree_get_pt(const size_t idx, int dim) const {
    if (dim == 0) {
      return points[idx].u;
    } else {
      return points[idx].v;
    }
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const {
    return false;
  }

  int num;
  Pnt* points;
};

}  // dso