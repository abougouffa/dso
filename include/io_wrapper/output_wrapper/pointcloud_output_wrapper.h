/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include "io_wrapper/output_3d_wrapper.h"
#include "util/minimal_image.h"

#include "full_system/hessian_blocks/hessian_blocks.h"
#include "util/frame_shell.h"

namespace dso {

class FrameHessian;
class CalibHessian;
class FrameShell;

namespace IOWrap {

class PointCloudOutputWrapper : public Output3DWrapper {
  std::ofstream xyzFile;

public:
  inline PointCloudOutputWrapper() {
    const char XYZ_FILENAME[] = "result_map.xyz";
    xyzFile.open(XYZ_FILENAME);
    std::cout << "OUT: Created PointCloudOutputWrapper" << std::endl;
    std::cout << "OUT: Saving Point Cloud to: " << XYZ_FILENAME << std::endl;
  }

  virtual ~PointCloudOutputWrapper() {
    if (xyzFile.is_open()) {
      xyzFile.flush();
      xyzFile.close();
    }

    std::cout << "OUT: Destroyed PointCloudOutputWrapper" << std::endl;
  }

  virtual void publishKeyframes(std::vector<FrameHessian *> &frames,
                                bool is_final, CalibHessian *HCalib) override {
    float fxl = HCalib->fxl(), fyl = HCalib->fyl(), cxl = HCalib->cxl(),
          cyl = HCalib->cyl();
    float fxi = 1. / fxl, fyi = 1. / fyl, cxi = -cxl / fxl, cyi = -cyl / fyl;

    if (is_final) {
      for (FrameHessian *frame : frames) {
        if (frame->shell->poseValid) {
          auto const &c2w_mat = frame->shell->camToWorld.matrix3x4();

          // Use only marginalized points.
          for (auto const *point : frame->pointHessiansMarginalized) {
            float depth = 1. / point->idepth;
            auto const x = (point->u * fxi + cxi) * depth;
            auto const y = (point->v * fyi + cyi) * depth;
            auto const z = (1. + 2. * fxi) * depth;

            Eigen::Vector4d pt_cam(x, y, z, 1.);
            Eigen::Vector3d pt_world = c2w_mat * pt_cam;

            if (xyzFile.is_open()) {
              xyzFile << pt_world.x() << " " << pt_world.y() << " "
                      << pt_world.z() << std::endl;
            } else {
              std::cout << "OUT: ERROR: Point Cloud file is closed!"
                        << std::endl;
            }
          }
        }
      }
    }
  }
};

} // namespace IOWrap

} // namespace dso
