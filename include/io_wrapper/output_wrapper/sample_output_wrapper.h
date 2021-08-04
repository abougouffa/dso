#pragma once

#include <glog/logging.h>
#include <boost/thread.hpp>

#include "full_system/hessian_blocks/hessian_blocks.h"
#include "io_wrapper/output_3d_wrapper.h"
#include "util/frame_shell.h"
#include "util/minimal_image.h"

namespace dso {

class FrameHessian;
class CalibHessian;
class FrameShell;

namespace IOWrap {

class SampleOutputWrapper : public Output3DWrapper {
 public:
  inline SampleOutputWrapper() {
    LOG(INFO) << "OUT: Created SampleOutputWrapper";
  }

  virtual ~SampleOutputWrapper() {
    LOG(INFO) << "OUT: Destroyed SampleOutputWrapper";
  }

  virtual void publishGraph(
      const std::map<
          uint64_t, Eigen::Vector2i, std::less<uint64_t>,
          Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>>&
          connectivity) override {
    LOG(INFO) << "OUT: Got graph with " << connectivity.size() << " edges.";

    int maxWrite = 5;

    for (const std::pair<uint64_t, Eigen::Vector2i>& p : connectivity) {
      int idHost = p.first >> 32;
      int idTarget = p.first & ((uint64_t)0xFFFFFFFF);
      LOG(INFO) << "OUT: Example Edge " << idHost << " -> " << idTarget
                << " has " << p.second[0] << " active and " << p.second[1]
                << " marg residuals.";
      --maxWrite;
      if (maxWrite == 0) {
        break;
      }
    }
  }

  virtual void publishKeyframes(std::vector<FrameHessian*>& frames, bool final,
                                CalibHessian* HCalib) override {
    for (FrameHessian* f : frames) {
      LOG(INFO) << "OUT: KF " << f->frameID << "("
                << (final ? "final" : "non-final") << ") (id "
                << f->shell->incoming_id << ", time " << f->shell->timestamp
                << "): " << f->pointHessians.size() << " active, "
                << f->immaturePoints.size()
                << " immature points. CameraToWorld: "
                << f->shell->camToWorld.matrix3x4();

      int maxWrite = 5;
      for (PointHessian* p : f->pointHessians) {
        LOG(INFO) << "OUT: Example Point x=" << p->u << ", y=" << p->v
                  << ", idepth=" << p->idepth_scaled << ", idepth std.dev. "
                  << sqrt(1.f / p->idepth_hessian) << ", "
                  << p->numGoodResiduals << " inlier-residuals";
        --maxWrite;
        if (maxWrite == 0) {
          break;
        }
      }
    }
  }

  virtual void publishCamPose(FrameShell* frame,
                              CalibHessian* HCalib) override {
    LOG(INFO) << "OUT:  Current Frame " << frame->incoming_id << ", time "
              << frame->timestamp << ", internal frame-ID " << frame->id
              << "). CameraToWorld: " << frame->camToWorld.matrix3x4();
  }

  virtual void pushLiveFrame(FrameHessian* image) override {
    // can be used to get the raw image / intensity pyramid.
  }

  virtual void pushDepthImage(MinimalImageB3* image) override {
    // can be used to get the raw image with depth overlay.
  }
  virtual bool needPushDepthImage() override { return false; }

  virtual void pushDepthImageFloat(MinimalImageF* image,
                                   FrameHessian* KF) override {
    LOG(INFO) << "OUT: Predicted depth for KF " << KF->frameID << "(id "
              << KF->shell->incoming_id << ", time " << KF->shell->timestamp
              << ", internal frame-ID " << KF->shell->id
              << "). CameraToWorld: " << KF->shell->camToWorld.matrix3x4();

    int maxWrite = 5;
    for (int y = 0; y < image->h; ++y) {
      for (int x = 0; x < image->w; ++x) {
        if (image->at(x, y) <= 0) {
          continue;
        }
        LOG(INFO) << "OUT: Example Idepth at pixel (" << x << ", " << y
                  << "): " << image->at(x, y);
        --maxWrite;
        if (maxWrite == 0) {
          break;
        }
      }
      if (maxWrite == 0) {
        break;
      }
    }
  }
};
}
}
