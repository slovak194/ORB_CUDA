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
  cv::Mat out;

  cv::namedWindow("out", cv::WINDOW_NORMAL);

  std::vector<double> track_times;

  int n_frames = 10;

  while (true) {

    cv::randn(noise, m, sigma);
    image += noise;

    const auto tp_1 = std::chrono::steady_clock::now();

    extractor.extract(image, cv::Mat(), keypoints, descriptors);

    const auto tp_2 = std::chrono::steady_clock::now();

    const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();

    track_times.push_back(track_time);

    n_frames = n_frames - 1;

    std::cout << n_frames << " " << 1/track_time << " " << image.rows << " " << image.cols << "\n";

    cv::drawKeypoints(image, keypoints, out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::imshow("out", out);
    cv::resizeWindow("out", 1280, 720);
    int k = cv::waitKey(-1);
    if (k == 'q') {
      break;
    }

    image -= noise;

    if (k == ' ') {
      continue;
    }

  }

  std::ofstream ofs("track_times.txt", std::ios::out);
  if (ofs.is_open()) {
    for (const auto track_time : track_times) {
      ofs << track_time << std::endl;
    }
    ofs.close();
  }

  return 0;
}

