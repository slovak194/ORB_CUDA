//
// Created by slovak on 12/21/21.
//

#include <vector>
#include <fstream>
#include <iostream>

#include <ORBextractor.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

int main(int argc, char *argv[]) {

  cv::Mat image = cv::imread("../test/vlcsnap-2021-12-22-14h05m41s129.png", cv::IMREAD_GRAYSCALE);

  assert(image.rows > 0);
  assert(image.cols > 0);

  cv::Mat noise(image.size(), image.type());
  float m = 10;
  float sigma = 10;


  int nFeatures = 5000;
  float fScaleFactor = 1.2;
  int nLevels = 8;
  int fIniThFAST = 20;
  int fMinThFAST = 7;

  auto extractor = ORB_CUDA::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  int n_frames = 10;

  while (n_frames >= 0) {

    cv::randn(noise, m, sigma);
    image += noise;

    extractor.extract(image, cv::Mat(), keypoints, descriptors);

    n_frames = n_frames - 1;

  }

  return 0;
}

