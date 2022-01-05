//
// Created by slovak on 12/23/21.
//

#pragma once

#include <vector>
#include <memory>

#include <opencv2/core.hpp>

namespace ORB_CUDA {

class ORBextractorImpl;

class ORBextractor {
 public:
  ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

  ~ORBextractor();
  // Compute the ORB features and descriptors on an image.
  // ORB are dispersed on the image using an octree.
  // Mask is ignored in the current implementation.
  void extract( cv::InputArray image, cv::InputArray mask,
                std::vector<cv::KeyPoint>& keypoints,
                cv::OutputArray descriptors);

  int GetLevels();

  float GetScaleFactor();

  std::vector<float> GetScaleFactors();

  std::vector<float> GetInverseScaleFactors();

  std::vector<float> GetScaleSigmaSquares();

  std::vector<float> GetInverseScaleSigmaSquares();

 private:
  std::unique_ptr<ORBextractorImpl> m_impl;

};
}
