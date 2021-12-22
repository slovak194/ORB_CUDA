//
// Created by slovak on 12/21/21.
//

#include <tuple>
#include <numeric>
#include <vector>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include <ORBextractor.h>

#include "openvslam/feature/orb_extractor.h"

int main(int argc, char *argv[]) {


  int nFeatures = 5000;
  float fScaleFactor = 1.2;
  int nLevels = 8;
  int fIniThFAST = 20;
  int fMinThFAST = 7;

  cv::Mat image = cv::imread("../test/vlcsnap-2021-12-22-14h05m41s129.png", cv::IMREAD_GRAYSCALE);

  assert(image.rows > 0);
  assert(image.cols > 0);

  cv::Mat noise(image.size(), image.type());
  float m = 10;
  float sigma = 10;


  {
    std::cout << "orb_cuda" << std::endl;
    auto extractor = ORB_CUDA::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<double> track_times;
    int n_frames = 100;

    while (n_frames >= 0) {

      cv::randn(noise, m, sigma);
      image += noise;

      const auto tp_1 = std::chrono::steady_clock::now();
      extractor.extract(image, cv::Mat(), keypoints, descriptors);
      const auto tp_2 = std::chrono::steady_clock::now();
      const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
      track_times.push_back(track_time);

      image -= noise;

      n_frames = n_frames - 1;

    }

    std::ofstream ofs("track_times_0.txt", std::ios::out);
    if (ofs.is_open()) {
      for (const auto track_time : track_times) {
        ofs << track_time << std::endl;
      }
      ofs.close();
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;

  }

  {
    std::cout << "orb_openvslam" << std::endl;
    auto extractor = openvslam::feature::orb_extractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<double> track_times;
    int n_frames = 100;

    while (n_frames >= 0) {

      cv::randn(noise, m, sigma);
      image += noise;

      const auto tp_1 = std::chrono::steady_clock::now();
      extractor.extract(image, cv::Mat(), keypoints, descriptors);
      const auto tp_2 = std::chrono::steady_clock::now();
      const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
      track_times.push_back(track_time);

      image -= noise;

      n_frames = n_frames - 1;

    }

    std::ofstream ofs("track_times_1.txt", std::ios::out);
    if (ofs.is_open()) {
      for (const auto track_time : track_times) {
        ofs << track_time << std::endl;
      }
      ofs.close();
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;

  }

  return 0;
}

