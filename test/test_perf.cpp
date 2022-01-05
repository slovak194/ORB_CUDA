//
// Created by slovak on 12/21/21.
//

#include <vector>
#include <fstream>
#include <iostream>

#include <orb_cuda/ORBextractor.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char *argv[]) {

  std::string image_file_path = PROJECT_SOURCE_DIR"/test/vlcsnap-2021-12-22-14h05m41s129.png";

  cv::Mat image = cv::imread(image_file_path, cv::IMREAD_GRAYSCALE);

  cv::resize(image, image, cv::Size(image.cols/2, image.rows/2), cv::INTER_LINEAR);

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

