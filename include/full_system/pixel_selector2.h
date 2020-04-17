#pragma once

#include "util/num_type.h"

namespace dso {

enum PixelSelectorStatus { PIXSEL_VOID = 0, PIXSEL_1, PIXSEL_2, PIXSEL_3 };

class FrameHessian;

class PixelSelector {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  // 寻找高梯度点，输出是map_out
  // map_out[i] == 0: invalid pixel
  // map_out[i] == 1: 第0层梯度满足条件
  // map_out[i] == 2: 第1层梯度满足条件
  // map_out[i] == 4: 第2层梯度满足条件
  // density: 希望找到的点的数量
  // recursionsLeft: 0表示不能再搜索一次, 1表示还能通过调整patch大小来搜索一次
  // plot: 是否显示找到的点的位置
  // thFactor: 比较梯度大小时用的系数
  int makeMaps(const FrameHessian* const fh, float* map_out, float density,
               int recursionsLeft = 1, bool plot = false, float thFactor = 1);

  PixelSelector(int w, int h);
  ~PixelSelector();

  // 将原图片分成多个32x32的patch, 通过直方图统计每一个patch中的梯度,
  // 找出所需要的阈值
  void makeHists(const FrameHessian* const fh);

 public:
  // 决定了select()中的patch的size
  // 数值越小->patch的数量越多->找到的点更多
  // Initial: 3
  int currentPotential;

  // Initial: false
  // 暂时没使用
  bool allowFast;

 private:
  //　遍历一个个的patch(边长为4 * pot, 2 * pot, pot, 1),
  //　找出高梯度点(同时在某些射线方向投影的模较大)
  Eigen::Vector3i select(const FrameHessian* const fh, float* map_out, int pot,
                         float thFactor = 1);

 private:
  // 一组随机数 (size: w * h)
  // 1) 用来删除多余的pixel (如果找到的点太多)
  // 2) 用来选择投影方向 (选取点的时候用)
  unsigned char* randomPattern;

  int* gradHist;  // 梯度直方图(50 bins)
  float* ths;  //　每一个patch的梯度阈值 (阈值以下的pixel不会被考虑)
  float* thsSmoothed;  // 每一个patch的smooth后的梯度阈值
  int thsStep;         // 横向patch的数量, 一个patch是32x32
  const FrameHessian* gradHistFrame;
};
}
