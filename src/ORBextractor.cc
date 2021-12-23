//
// Created by slovak on 12/23/21.
//

#include <orb_cuda/ORBextractor.h>
#include <orb_cuda/ORBextractorImpl.h>

using namespace ORB_CUDA;

ORBextractor::ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST)
    : m_impl(std::make_unique<ORBextractorImpl>(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST)) {}

ORBextractor::~ORBextractor() = default;

// Compute the ORB features and descriptors on an image.
// ORB are dispersed on the image using an octree.
// Mask is ignored in the current implementation.
void ORBextractor::extract(cv::InputArray image, cv::InputArray mask,
                           std::vector<cv::KeyPoint> &keypoints,
                           cv::OutputArray descriptors) {
  m_impl->extract(image, mask, keypoints, descriptors);
}

int inline ORBextractor::GetLevels() {
  return m_impl->GetLevels();
}

float inline ORBextractor::GetScaleFactor() {
  return m_impl->GetScaleFactor();
}

std::vector<float> inline ORBextractor::GetScaleFactors() {
  return m_impl->GetScaleFactors();
}

std::vector<float> inline ORBextractor::GetInverseScaleFactors() {
  return m_impl->GetInverseScaleFactors();
}

std::vector<float> inline ORBextractor::GetScaleSigmaSquares() {
  return m_impl->GetScaleSigmaSquares();
}

std::vector<float> inline ORBextractor::GetInverseScaleSigmaSquares() {
  return m_impl->GetInverseScaleSigmaSquares();
}
