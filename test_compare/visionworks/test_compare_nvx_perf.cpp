//
// Created by slovak on 12/21/21.
//

#include <tuple>
#include <numeric>
#include <vector>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <NVX/nvxcu.h>
#include <NVX/nvx.h>
#include <NVX/Utility.hpp>
#include <NVX/nvx_opencv_interop.hpp>

#include <VX/vxu.h>

#include <OVX/UtilityOVX.hpp>
//
//static nvxcu_pitch_linear_image_t createImageRGBX(uint32_t width, uint32_t height)
//{
//  void * dev_ptr = NULL;
//  size_t pitch = 0;
//  NVXIO_CUDA_SAFE_CALL( cudaMallocPitch(&dev_ptr, &pitch, width * sizeof(uint8_t) * 4, height) );
//
//  nvxcu_pitch_linear_image_t image;
//  image.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;
//  image.base.format = NVXCU_DF_IMAGE_RGBX;
//  image.base.width = width;
//  image.base.height = height;
//  image.planes[0].dev_ptr = dev_ptr;
//  image.planes[0].pitch_in_bytes = pitch;
//
//  return image;
//}
//
//nvxcu_plain_array_t createArrayPoint2F(uint32_t capacity)
//{
//  void * dev_ptr = NULL;
//  CUDA_SAFE_CALL( cudaMalloc(&dev_ptr, capacity * sizeof(nvxcu_point2f_t)) );
//
//  uint32_t * num_items_dev_ptr = NULL;
//  CUDA_SAFE_CALL( cudaMalloc((void **)&num_items_dev_ptr, sizeof(uint32_t)) );
//  CUDA_SAFE_CALL( cudaMemset(num_items_dev_ptr, 0, sizeof(uint32_t)) );
//
//  nvxcu_plain_array_t arr;
//  arr.base.array_type = NVXCU_PLAIN_ARRAY;
//  arr.base.item_type = NVXCU_TYPE_POINT2F;
//  arr.base.capacity = capacity;
//  arr.dev_ptr = dev_ptr;
//  arr.num_items_dev_ptr = num_items_dev_ptr;
//
//  return arr;
//}



int main(int argc, char *argv[]) {


  int fIniThFAST = 20;
  int fMinThFAST = 7;

  std::string image_file_path = PROJECT_SOURCE_DIR"/test/vlcsnap-2021-12-22-14h05m41s129.png";
  std::cout << image_file_path << "\n";
  cv::Mat image = cv::imread(image_file_path, cv::IMREAD_GRAYSCALE);
  assert(!image.empty());

  cv::resize(image, image, cv::Size(image.cols/4, image.rows/4), cv::INTER_LINEAR);
//  cv::resize(image, image, cv::Size(image.cols, image.rows), cv::INTER_LINEAR);

  assert(image.rows > 0);
  assert(image.cols > 0);

  cv::Mat noise(image.size(), image.type());
  float m = 10;
  float sigma = 10;

//  fast_opencv
  {
    std::cout << "\nfast_opencv" << std::endl;

    auto fast_type = cv::FastFeatureDetector::TYPE_9_16;
//  auto fast_type = FastFeatureDetector::TYPE_7_12;
//  auto fast_type = FastFeatureDetector::TYPE_5_8;

//    auto extractor = ORB_CUDA::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    std::vector<cv::KeyPoint> keypoints;
//    cv::Mat descriptors;
    std::vector<double> track_times;
    int n_frames = 100;

    while (n_frames >= 0) {

      cv::randn(noise, m, sigma);
      image += noise;

      const auto tp_1 = std::chrono::steady_clock::now();
      cv::FAST(image, keypoints, fIniThFAST, false, fast_type);

      const auto tp_2 = std::chrono::steady_clock::now();
      const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
      track_times.push_back(track_time);

      image -= noise;

      n_frames = n_frames - 1;

    }

    std::ofstream ofs("ocv.txt", std::ios::out);
    if (ofs.is_open()) {
      for (const auto track_time : track_times) {
        ofs << track_time << std::endl;
      }
      ofs.close();
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "N_kp: " << keypoints.size() << std::endl;
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;

  }

//  fast_openvx
  {
    std::cout << "\nfast_openvx" << std::endl;

    ovxio::ContextGuard context_;

    auto main_graph_ = vxCreateGraph(context_);
    NVXIO_CHECK_REFERENCE(main_graph_);

    vx_image input = nvx_cv::createVXImageFromCVMat(context_, image);
    NVXIO_CHECK_REFERENCE(input);

    vx_float32 strength_thresh = 20.0f;
    vx_scalar s_strength_thresh = vxCreateScalar(context_, VX_TYPE_FLOAT32, &strength_thresh);
    NVXIO_CHECK_REFERENCE(s_strength_thresh);

    vx_array keypoints;
    vx_scalar corners;

    // allocate some space to store the keypoints
    keypoints = vxCreateVirtualArray(main_graph_, VX_TYPE_KEYPOINT, 10000);
    NVXIO_CHECK_REFERENCE( keypoints );

    vx_size sz_corners = 0;
    corners = vxCreateScalar(context_, VX_TYPE_SIZE, &sz_corners);
    NVXIO_CHECK_REFERENCE( corners );

    auto corner_detection_node = vxFastCornersNode(
        main_graph_, input, s_strength_thresh, vx_true_e, keypoints, corners);

    auto status = vxGetStatus((vx_reference)corner_detection_node);

    if (status != VX_SUCCESS) {
      std::cout << status << "\n";
    }

    NVXIO_CHECK_REFERENCE(corner_detection_node);

    // Ensure highest graph optimization level
    const char* option = "-O3";
    NVXIO_SAFE_CALL( vxSetGraphAttribute(main_graph_, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)) );

    NVXIO_SAFE_CALL( vxVerifyGraph(main_graph_) );

    vxReleaseScalar(&s_strength_thresh);
    vxReleaseImage(&input);

//    std::vector<cv::KeyPoint> keypoints;
    std::vector<double> track_times;
    int n_frames = 100;

    while (n_frames >= 0) {

      cv::randn(noise, m, sigma);
      image += noise;

      const auto tp_1 = std::chrono::steady_clock::now();


      // Process graph
      NVXIO_SAFE_CALL( vxProcessGraph(main_graph_) );

//      vxuFastCorners(
//          context_,
//          input, s_strength_thresh, vx_bool(true), corners, num_corners);

      const auto tp_2 = std::chrono::steady_clock::now();
      const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
      track_times.push_back(track_time);

      image -= noise;

      n_frames = n_frames - 1;

    }

    vxReleaseImage(&input);

    std::ofstream ofs("fast_openvx.txt", std::ios::out);
    if (ofs.is_open()) {
      for (const auto track_time : track_times) {
        ofs << track_time << std::endl;
      }
      ofs.close();
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);

    vx_int32 num_corners_out = 0;

    NVXIO_SAFE_CALL(vxReadScalarValue(corners, &num_corners_out));

    std::cout << "N_kp: " << num_corners_out << std::endl;
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;

  }

//  fast_openvx
//  {
//    std::cout << "\nfast_openvx_cuda" << std::endl;
//
//    ovxio::ContextGuard context_;
//
//    nvxcu_pitch_linear_image_t frame = createImageRGBX(image.cols, image.rows);
//
//    vx_image input = nvx_cv::createVXImageFromCVMat(context_, image);
//    NVXIO_CHECK_REFERENCE(input);
//
//    float strength_thresh = static_cast<float>(fIniThFAST);
//    int32_t nonmax_suppression = 1;
//
//    nvxcu_plain_array_t corners = createArrayPoint2F(10000);
//
//    uint32_t *num_corners_dev_ptr = nullptr;
//
////                                                      uint32_t *num_corners_dev_ptr,
////                                                      const nvxcu_tmp_buf_t* tmp_buf,
////                                                      const nvxcu_border_t* border,
////                                                      const nvxcu_exec_target_t* exec_target
//
//
//
//    std::vector<double> track_times;
//    int n_frames = 100;
//
//    while (n_frames >= 0) {
//
//      cv::randn(noise, m, sigma);
//      image += noise;
//
//      const auto tp_1 = std::chrono::steady_clock::now();
//
//      auto res = nvxcuFastCorners(&frame.base, strength_thresh, 1, &corners.base, num_corners_dev_ptr, tmp_buf, border, exec_target);
//
////
////      auto res = nvxcuFastCorners(
////      const nvxcu_image_t* input,
////                                                      float strength_thresh,
////                                                      int32_t nonmax_suppression,
////                                                      const nvxcu_array_t* corners,
////                                                      uint32_t *num_corners_dev_ptr,
////                                                      const nvxcu_tmp_buf_t* tmp_buf,
////                                                      const nvxcu_border_t* border,
////                                                      const nvxcu_exec_target_t* exec_target
////                                                      );
//
//      const auto tp_2 = std::chrono::steady_clock::now();
//      const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
//      track_times.push_back(track_time);
//
//      image -= noise;
//
//      n_frames = n_frames - 1;
//
//    }
//
//    vxReleaseImage(&input);
//
//    std::ofstream ofs("fast_openvx.txt", std::ios::out);
//    if (ofs.is_open()) {
//      for (const auto track_time : track_times) {
//        ofs << track_time << std::endl;
//      }
//      ofs.close();
//    }
//
//    std::sort(track_times.begin(), track_times.end());
//    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
//
//    vx_int32 num_corners_out = 0;
//
//    NVXIO_SAFE_CALL(vxReadScalarValue(corners, &num_corners_out));
//
//    std::cout << "N_kp: " << num_corners_out << std::endl;
//    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
//    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
//
//  }


  return 0;
}

