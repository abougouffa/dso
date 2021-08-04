#include "full_system/pixel_selector2.h"

#include "full_system/hessian_blocks/hessian_blocks.h"
#include "io_wrapper/image_display.h"
#include "util/global_calib.h"
#include "util/global_funcs.h"
#include "util/num_type.h"

namespace dso {

PixelSelector::PixelSelector(int w, int h) {
  randomPattern = new unsigned char[w * h];
  std::srand(3141592);  // want to be deterministic.
  for (int i = 0; i < w * h; ++i) {
    randomPattern[i] = rand() & 0xFF;
  }

  currentPotential = 3;

  gradHist = new int[100 * (1 + w / 32) * (1 + h / 32)];
  ths = new float[(w / 32) * (h / 32) + 100];
  thsSmoothed = new float[(w / 32) * (h / 32) + 100];

  allowFast = false;
  gradHistFrame = 0;
}

PixelSelector::~PixelSelector() {
  delete[] randomPattern;
  delete[] gradHist;
  delete[] ths;
  delete[] thsSmoothed;
}

// 找出梯度的阈值
int computeHistQuantil(int* hist, float below) {
  int th = hist[0] * below + 0.5f;  // 舍弃的pixel的数量
  for (int i = 0; i < 90; ++i) {
    th -= hist[i + 1];
    if (th < 0) {
      // i为所要找的梯度阈值
      return i;
    }
  }
  return 90;
}

void PixelSelector::makeHists(const FrameHessian* const fh) {
  gradHistFrame = fh;

  // 取出第0层的梯度平方和
  float* mapmax0 = fh->absSquaredGrad[0];

  int w = wG[0];
  int h = hG[0];

  // 将图片分割成(w32 x h32)个32x32的patch
  int w32 = w / 32;  // 横向patch的数量
  int h32 = h / 32;  // 纵向patch的数量
  thsStep = w32;

  // 遍历每一个patch
  for (int y = 0; y < h32; ++y)
    for (int x = 0; x < w32; ++x) {
      // 所指向patch的梯度平方和
      float* map0 = mapmax0 + 32 * x + 32 * y * w;

      // hist0[0]: number of valid pixels(非边缘点的数量)
      // hist0[1] - hist0[49]: 不同梯度pixel的数量
      int* hist0 = gradHist;  // + 50*(x+y*w32);
      memset(hist0, 0, sizeof(int) * 50);

      // 遍历patch中的每一个pixel
      for (int j = 0; j < 32; ++j) {
        for (int i = 0; i < 32; ++i) {
          int it = i + 32 * x;
          int jt = j + 32 * y;
          if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1) {
            // 忽略图片边缘点
            continue;
          }
          // pixel处的梯度
          int g = sqrtf(map0[i + j * w]);
          if (g > 48) {
            g = 48;
          }

          // 统计到梯度直方图
          ++hist0[g + 1];
          ++hist0[0];
        }
      }

      // 将此patch的梯度阈值保存起来
      ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) +
                         setting_minGradHistAdd;
    }

  // 通过考虑相邻patch, 来对每个patch的梯度进行smoothing
  for (int y = 0; y < h32; ++y) {
    for (int x = 0; x < w32; ++x) {
      float sum = 0, num = 0;
      if (x > 0) {
        if (y > 0) {
          ++num;
          sum += ths[x - 1 + (y - 1) * w32];
        }
        if (y < h32 - 1) {
          ++num;
          sum += ths[x - 1 + (y + 1) * w32];
        }
        ++num;
        sum += ths[x - 1 + (y)*w32];
      }

      if (x < w32 - 1) {
        if (y > 0) {
          ++num;
          sum += ths[x + 1 + (y - 1) * w32];
        }
        if (y < h32 - 1) {
          ++num;
          sum += ths[x + 1 + (y + 1) * w32];
        }
        ++num;
        sum += ths[x + 1 + (y)*w32];
      }

      if (y > 0) {
        ++num;
        sum += ths[x + (y - 1) * w32];
      }
      if (y < h32 - 1) {
        ++num;
        sum += ths[x + (y + 1) * w32];
      }
      ++num;
      sum += ths[x + y * w32];

      thsSmoothed[x + y * w32] = (sum / num) * (sum / num);
    }
  }
}

int PixelSelector::makeMaps(const FrameHessian* const fh, float* map_out,
                            float density, int recursionsLeft, bool plot,
                            float thFactor) {
  float numHave = 0;        // 所找出的高梯度点的数量
  float numWant = density;  // 所需的高梯度点的数量
  float quotia;             // 比例: want / have
  int idealPotential = currentPotential;

  //	if(setting_pixelSelectionUseFast>0 && allowFast)
  //	{
  //		memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
  //		std::vector<cv::KeyPoint> pts;
  //		cv::Mat img8u(hG[0],wG[0],CV_8U);
  //		for(int i=0;i<wG[0]*hG[0];++i)
  //		{
  //			float v = fh->dI[i][0]*0.8;
  //			img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255
  //:
  // v;
  //		}
  //		cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
  //		for(unsigned int i=0;i<pts.size();++i)
  //		{
  //			int x = pts[i].pt.x+0.5;
  //			int y = pts[i].pt.y+0.5;
  //			map_out[x+y*wG[0]]=1;
  //			numHave++;
  //		}
  //
  //		printf("FAST selection: got %f / %f!\n", numHave, numWant);
  //		quotia = numWant / numHave;
  //	}
  //	else
  {
    // the number of selected pixels behaves approximately as
    // K / (pot+1)^2, where K is a scene-dependent constant.
    // we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise
    // we will re-select.

    if (fh != gradHistFrame) {
      makeHists(fh);
    }

    // select!
    Eigen::Vector3i n = this->select(fh, map_out, currentPotential, thFactor);

    // sub-select!
    numHave = n[0] + n[1] + n[2];  // 总共找出的高梯度点的数量
    quotia = numWant / numHave;  // 想要的点的数量/拥有的点的数量

    // by default we want to over-sample by 40% just to be sure.
    // 根据本帧找到的点的结果，调整idealPotential,
    // 之后会赋给currentPotential供下一帧使用.
    // 刻意调大了K, 是为了保证下次找到充足的点
    float K = numHave * (currentPotential + 1) * (currentPotential + 1);
    idealPotential = sqrtf(K / numWant) - 1;  // round down.
    if (idealPotential < 1) {
      idealPotential = 1;
    }

    // 如果recursionsLeft > 0, 我们有机会再调整currentPotential大小,
    // 从而调整找到的pixel的数量
    if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1) {
      // re-sample to get more points!
      // potential needs to be smaller
      if (idealPotential >= currentPotential) {
        idealPotential = currentPotential - 1;
      }

      //		printf("PixelSelector: have %.2f%%, need %.2f%%.
      // RESAMPLE with pot %d -> %d.\n",
      //				100*numHave/(float)(wG[0]*hG[0]),
      //				100*numWant/(float)(wG[0]*hG[0]),
      //				currentPotential,
      //				idealPotential);
      currentPotential = idealPotential;

      // 减小currentPotential, 再次进行搜索高梯度点
      return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
    } else if (recursionsLeft > 0 && quotia < 0.25) {
      // re-sample to get less points!
      // 点太多了
      if (idealPotential <= currentPotential) {
        // 增加寻找的patch的大小, 从而降低找到点的数量
        idealPotential = currentPotential + 1;
      }

      //		printf("PixelSelector: have %.2f%%, need %.2f%%.
      // RESAMPLE with pot %d -> %d.\n",
      //				100*numHave/(float)(wG[0]*hG[0]),
      //				100*numWant/(float)(wG[0]*hG[0]),
      //				currentPotential,
      //				idealPotential);
      currentPotential = idealPotential;
      // 增加currentPotential, 再次进行搜索高梯度点
      return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
    }
  }

  int numHaveSub = numHave;
  if (quotia < 0.95) {
    // 如果拥有的点仍然太多, 随机删除一些点
    int wh = wG[0] * hG[0];
    int rn = 0;
    unsigned char charTH = 255 * quotia;
    for (int i = 0; i < wh; ++i) {
      if (map_out[i] != 0) {
        if (randomPattern[rn] > charTH) {
          map_out[i] = 0;
          --numHaveSub;
        }
        ++rn;
      }
    }
  }

  //	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d ->
  //%d. Subsampled to %.2f%%\n",
  //			100*numHave/(float)(wG[0]*hG[0]),
  //			100*numWant/(float)(wG[0]*hG[0]),
  //			currentPotential,
  //			idealPotential,
  //			100*numHaveSub/(float)(wG[0]*hG[0]));
  currentPotential = idealPotential;

  if (plot) {
    int w = wG[0];
    int h = hG[0];

    MinimalImageB3 img(w, h);

    for (int i = 0; i < w * h; ++i) {
      float c = fh->dI[i][0] * 0.7;
      if (c > 255) {
        c = 255;
      }
      img.at(i) = Vec3b(c, c, c);
    }
    IOWrap::displayImage("Selector Image", &img);

    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        int i = x + y * w;
        if (map_out[i] == 1) {
          img.setPixelCirc(x, y, Vec3b(0, 255, 0));
        } else if (map_out[i] == 2) {
          img.setPixelCirc(x, y, Vec3b(255, 0, 0));
        } else if (map_out[i] == 4) {
          img.setPixelCirc(x, y, Vec3b(0, 0, 255));
        }
      }
    IOWrap::displayImage("Selector Pixels", &img);
  }

  return numHaveSub;
}

Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
                                      float* map_out, int pot, float thFactor) {
  // map0 = dIp[0], the first level of pyramid
  // map0[0]: intensity
  // map0[1]: gradient x (gx)
  // map0[2]: gradient y (gy)
  Eigen::Vector3f const* const map0 = fh->dI;

  float* mapmax0 = fh->absSquaredGrad[0];  // 第0层梯度平方和
  float* mapmax1 = fh->absSquaredGrad[1];  // 第1层梯度平方和
  float* mapmax2 = fh->absSquaredGrad[2];  // 第2层梯度平方和

  int w = wG[0];   //第0层图片宽度
  int w1 = wG[1];  //第1层图片宽度
  int w2 = wG[2];  // 第2层图片宽度
  int h = hG[0];   // 第0层图片高度

  // 单位投影方向
  const Vec2f directions[16] = {
      Vec2f(0, 1.0000),      Vec2f(0.3827, 0.9239),  Vec2f(0.1951, 0.9808),
      Vec2f(0.9239, 0.3827), Vec2f(0.7071, 0.7071),  Vec2f(0.3827, -0.9239),
      Vec2f(0.8315, 0.5556), Vec2f(0.8315, -0.5556), Vec2f(0.5556, -0.8315),
      Vec2f(0.9808, 0.1951), Vec2f(0.9239, -0.3827), Vec2f(0.7071, -0.7071),
      Vec2f(0.5556, 0.8315), Vec2f(0.9808, -0.1951), Vec2f(1.0000, 0.0000),
      Vec2f(0.1951, -0.9808)};

  memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));

  float dw1 = setting_gradDownweightPerLevel;  // 梯度阈值变化的系数
  float dw2 = dw1 * dw1;                       // dw1 * dw1

  int n3 = 0, n2 = 0, n4 = 0;

  // 对原图片的每一个patch4进行遍历, patch4的边长为(4 * pot)
  for (int y4 = 0; y4 < h; y4 += (4 * pot)) {
    for (int x4 = 0; x4 < w; x4 += (4 * pot)) {
      int my3 = std::min((4 * pot), h - y4);
      int mx3 = std::min((4 * pot), w - x4);
      int bestIdx4 = -1;
      float bestVal4 = 0;
      Vec2f dir4 = directions[randomPattern[n2] & 0xF];

      // 将一个patch4分割为更小的patch2进行遍历,
      // patch2的边长为(2 * pot)
      for (int y3 = 0; y3 < my3; y3 += (2 * pot)) {
        for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
          int x34 = x3 + x4;
          int y34 = y3 + y4;
          int my2 = std::min((2 * pot), h - y34);
          int mx2 = std::min((2 * pot), w - x34);
          int bestIdx3 = -1;
          float bestVal3 = 0;
          Vec2f dir3 = directions[randomPattern[n2] & 0xF];

          // 将一个patch2分割为更小的patch1进行遍历,
          // patch1的边长为(pot), 每一个patch1中, 我们只要bestIdx2
          for (int y2 = 0; y2 < my2; y2 += pot) {
            for (int x2 = 0; x2 < mx2; x2 += pot) {
              int x234 = x2 + x34;
              int y234 = y2 + y34;
              int my1 = std::min(pot, h - y234);
              int mx1 = std::min(pot, w - x234);
              int bestIdx2 = -1;
              float bestVal2 = 0;
              Vec2f dir2 = directions[randomPattern[n2] & 0xF];

              // 对每一个patch1中的每一个pixel进行遍历
              for (int y1 = 0; y1 < my1; y1 += 1) {
                for (int x1 = 0; x1 < mx1; x1 += 1) {
                  CHECK_LT(x1 + x234, w);
                  CHECK_LT(y1 + y234, h);
                  int idx = x1 + x234 + w * (y1 + y234);
                  int xf = x1 + x234;
                  int yf = y1 + y234;

                  if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4) {
                    // 不考虑边缘点
                    continue;
                  }

                  // >> 5: 即除以32 (每个patch的大小)
                  // 找到该patch的第0层的梯度阈值
                  float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep];

                  float pixelTH1 = pixelTH0 * dw1;  // 第1层的梯度阈值
                  float pixelTH2 = pixelTH1 * dw2;  // 第2层的梯度阈值

                  float ag0 = mapmax0[idx];  // idx处的第０层梯度平方和

                  // (Tong) TODO: ag0是梯度平方和, pixelTH0是梯度阈值,
                  // 为啥这两个东西互相比较?
                  if (ag0 > pixelTH0 * thFactor) {
                    // 只考虑梯度足够大的pixel
                    Vec2f ag0d = map0[idx].tail<2>();  // 第0层的[gx; gy]

                    // 梯度在一个随机方向上的投影的模
                    float dirNorm = fabsf((float)(ag0d.dot(dir2)));
                    if (!setting_selectDirectionDistribution) {
                      dirNorm = ag0;
                    }

                    if (dirNorm > bestVal2) {
                      // 找到在投影方向上的能得到最大模的pixel,
                      // 一旦找到, 我们就不会再用第1, 2层的梯度寻找了
                      bestVal2 = dirNorm;
                      bestIdx2 = idx;
                      bestIdx3 = -2;
                      bestIdx4 = -2;
                    }
                  }
                  if (bestIdx3 == -2) {
                    // 如果这个pixel是best, 继续下一个pixel
                    continue;
                  }

                  // 如果第0层梯度不满足条件,
                  // 我们再看这个pixel在第1层的梯度是否满足条件
                  float ag1 = mapmax1[static_cast<int>(xf * 0.5f + 0.25f) +
                                      static_cast<int>(yf * 0.5f + 0.25f) * w1];
                  if (ag1 > pixelTH1 * thFactor) {
                    // 注意: 下面计算的仍是第0层的梯度
                    Vec2f ag0d = map0[idx].tail<2>();  // 第0层的[gx; gy]
                    float dirNorm = fabsf((float)(ag0d.dot(dir3)));
                    if (!setting_selectDirectionDistribution) {
                      dirNorm = ag1;
                    }

                    if (dirNorm > bestVal3) {
                      // 找到在投影方向上的能得到最大模的pixel,
                      // 一旦找到, 我们就不会再用第2层的梯度寻找了
                      bestVal3 = dirNorm;
                      bestIdx3 = idx;
                      bestIdx4 = -2;
                    }
                  }
                  if (bestIdx4 == -2) {
                    continue;
                  }

                  // 如果第1层梯度不满足条件,
                  // 我们再看这个pixel在第2层的梯度是否满足条件
                  float ag2 =
                      mapmax2[static_cast<int>(xf * 0.25f + 0.125f) +
                              static_cast<int>(yf * 0.25f + 0.125f) * w2];
                  if (ag2 > pixelTH2 * thFactor) {
                    // 注意: 下面计算的仍是第0层的梯度
                    Vec2f ag0d = map0[idx].tail<2>();  // 第0层的[gx; gy]
                    float dirNorm = fabsf((float)(ag0d.dot(dir4)));
                    if (!setting_selectDirectionDistribution) {
                      dirNorm = ag2;
                    }

                    if (dirNorm > bestVal4) {
                      bestVal4 = dirNorm;
                      bestIdx4 = idx;
                    }
                  }
                }
              }

              if (bestIdx2 > 0) {
                // 如果我们使用第0层梯度找到了满足的点,
                // 增加bestVal3(停止更新bestIdx3)
                map_out[bestIdx2] = 1;
                bestVal3 = 1e10;
                ++n2;
              }
            }
          }

          if (bestIdx3 > 0) {
            // 如果我们使用第1层梯度找到了满足的点,
            // 增加bestVal4(停止更新bestIdx4)
            map_out[bestIdx3] = 2;
            bestVal4 = 1e10;
            ++n3;
          }
        }
      }

      if (bestIdx4 > 0) {
        // 如果我们使用第2层梯度找到了满足的点
        map_out[bestIdx4] = 4;
        ++n4;
      }
    }
  }

  // 返回使用第0,1,2层梯度找出的高梯度pixel的数量
  return Eigen::Vector3i(n2, n3, n4);
}
}
