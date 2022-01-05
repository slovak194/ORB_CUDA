#include <tuple>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <NVX/nvx.h>
#include <NVX/nvx_opencv_interop.hpp>

#include <OVX/UtilityOVX.hpp>

vx_status VX2MatC1(vx_context m_vxCtx, vx_image& vxiSrc, cv::Mat& matDst)
{
  vx_imagepatch_addressing_t dst_addr;
  dst_addr.dim_x = matDst.cols;
  dst_addr.dim_y = matDst.rows;
  dst_addr.stride_x = sizeof(vx_uint8);
  dst_addr.stride_y = matDst.step;
  void* dst_ptrs[] = {matDst.data};
  vx_image vxiDst = vxCreateImageFromHandle(m_vxCtx, VX_DF_IMAGE_U8, &dst_addr, dst_ptrs, VX_IMPORT_TYPE_HOST);

  vx_status status = nvxuCopyImage(m_vxCtx, vxiSrc, vxiDst);

  vxReleaseImage(&vxiDst);

  return status;
}

void Mat2VXC1 (vx_context m_vxCtx, cv::Mat& matSrc, vx_image& vxiSrc) {
  vx_imagepatch_addressing_t src_addr;
  src_addr.dim_x = matSrc.cols;
  src_addr.dim_y = matSrc.rows;
  src_addr.stride_x = sizeof(vx_uint8);
  src_addr.stride_y = matSrc.step;
  void* src_ptrs[] = {matSrc.data};
  vxiSrc = vxCreateImageFromHandle(m_vxCtx, VX_DF_IMAGE_U8, &src_addr, src_ptrs, VX_IMPORT_TYPE_HOST);
}


int main(int argc, char *argv[]) {

  std::string image_file_path = PROJECT_SOURCE_DIR"/test/vlcsnap-2021-12-22-14h05m41s129.png";
  std::cout << image_file_path << "\n";
  cv::Mat image = cv::imread(image_file_path, cv::IMREAD_GRAYSCALE);

  assert(image.isContinuous());
  assert(!image.empty());
  assert(image.rows > 0);
  assert(image.cols > 0);

  cv::resize(image, image, cv::Size(image.cols / 4, image.rows / 4), cv::INTER_LINEAR);

  vx_context context = vxCreateContext();
  NVXIO_CHECK_REFERENCE(context);

  vx_image input = nvx_cv::createVXImageFromCVMat(context, image);
  NVXIO_CHECK_REFERENCE(input);

  vx_float32 strength_thresh_init_value = 20.0f;
  vx_scalar strength_thresh = vxCreateScalar(context, VX_TYPE_FLOAT32, &strength_thresh_init_value);
  NVXIO_CHECK_REFERENCE(strength_thresh);

  vx_array keypoints = vxCreateArray(context, VX_TYPE_KEYPOINT, 10000);
  NVXIO_CHECK_REFERENCE(keypoints);

  vx_size n_corners_init_value = 0;
  vx_scalar n_corners = vxCreateScalar(context, VX_TYPE_SIZE, &n_corners_init_value);
  NVXIO_CHECK_REFERENCE(n_corners);

  NVXIO_SAFE_CALL(vxuFastCorners(context, input, strength_thresh, vx_true_e, keypoints, n_corners));

//  vxReleaseArray(&keypoints);
//  vxReleaseScalar(&strength_thresh);
//  vxReleaseImage(&input);

  vx_int32 n_corners_value = 0;
  NVXIO_SAFE_CALL(vxReadScalarValue(n_corners, &n_corners_value));

  vx_size num_items = 0ul;
  NVXIO_SAFE_CALL(vxQueryArray(keypoints, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items, sizeof(num_items)));

  vx_size i, stride = 0ul;
  void *base = NULL;

  NVXIO_SAFE_CALL(vxAccessArrayRange(keypoints, 0, num_items, &stride, &base, VX_READ_ONLY));

  for (i = 0; i < 10; i++) {

    std::cout
        << "x: " << vxArrayItem(nvx_keypointf_t, base, i, stride).x << " "
        << "y: " << vxArrayItem(nvx_keypointf_t, base, i, stride).y << " "
        << "scale: " << vxArrayItem(nvx_keypointf_t, base, i, stride).scale << " "
        << "strength: " << vxArrayItem(nvx_keypointf_t, base, i, stride).strength << " "
        << "tracking_status: " << vxArrayItem(nvx_keypointf_t, base, i, stride).tracking_status << " "
        << "orientation: " << vxArrayItem(nvx_keypointf_t, base, i, stride).orientation << " "
        << "\n";

  }
  vxCommitArrayRange(keypoints, 0, num_items, base);

  std::cout
      << "num_items: " << num_items << " "
      << "n_corners_value: " << n_corners_value << " "
      << "stride: " << stride << " "
      << "\n";

  return 0;
}

