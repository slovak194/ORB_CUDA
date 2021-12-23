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

  cv::Mat image = cv::imread("../test/vlcsnap-2021-12-22-14h05m41s129.png", cv::IMREAD_GRAYSCALE);

  cv::resize(image, image, cv::Size(image.cols/2, image.rows/2), cv::INTER_LINEAR);

  assert(image.rows > 0);
  assert(image.cols > 0);

  int nFeatures = 100;
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

  while (true) {

    const auto tp_1 = std::chrono::steady_clock::now();

    extractor.extract(image, cv::Mat(), keypoints, descriptors);

    const auto tp_2 = std::chrono::steady_clock::now();

    const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();

    track_times.push_back(track_time);

    std::cout << 1/track_time << "\n";

    cv::drawKeypoints(image, keypoints, out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("out", out);
    cv::resizeWindow("out", 1280, 720);
    int k = cv::waitKey(-1);
    if (k == 'q') {
      break;
    }

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

