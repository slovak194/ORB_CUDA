#include "graph_mode_stabilizer.hpp"
#include "homography_smoother_node.hpp"
#include "NVX/nvx_timer.hpp"
#include "OVX/UtilityOVX.hpp"
#include <math.h>
#include <assert.h>
#include <vector>
GraphModeStabilizer::GraphModeStabilizer(vx_context context)
{
  context_ = context;
  points_ = 0;
  corresponding_points_ = 0;
  gaussian_weights_ = 0;
  homography_matrices_ = 0;
  perspective_matrix_ = 0;
  stabilized_frame_ = 0;
  gray_frames_delay_ = 0;
  pyr_delay_ = 0;
  frames_delay_ = 0;
  main_graph_ = 0;
  color_convert_node_ = 0;
  gaussian_pyramid_node_ = 0;
  opt_flow_node_ = 0;
  find_homography_node_ = 0;
  fast_corners_node_ = 0;
}
GraphModeStabilizer::~GraphModeStabilizer()
{
  vxReleaseArray(&points_);
  vxReleaseArray(&corresponding_points_);
  vxReleaseArray(&gaussian_weights_);
  vxReleaseDelay(&homography_matrices_);
  vxReleaseMatrix(&perspective_matrix_);
  vxReleaseImage(&stabilized_frame_);
  vxReleaseDelay(&gray_frames_delay_);
  vxReleaseDelay(&pyr_delay_);
  vxReleaseDelay(&frames_delay_);
  vxReleaseGraph(&main_graph_);
};
vx_status GraphModeStabilizer::initGaussianWeights()
{
  vx_status status = VX_SUCCESS;
  vx_float32 sigma = (vx_float32)params_.smoothing_window_size * 0.7;
  vx_int32 num_items = 2 * (vx_int32)params_.smoothing_window_size + 1;
  std::vector<vx_float32> gaussian_weights_data;
  gaussian_weights_data.resize(num_items);
  vx_float32 sum = 0;
  for (vx_int32 i = 0; i < num_items; ++i)
  {
    gaussian_weights_data[i] = exp(-(i - (vx_float32)params_.smoothing_window_size) * (i - (vx_float32)params_.smoothing_window_size) / (2.f * sigma * sigma));
    sum += gaussian_weights_data[i];
  }
  //normalize weights
  assert((sum > 0.00000000001) || (sum < - 0.00000000001));
  vx_float32 scaler = 1.f / sum;
  for (vx_int32 i = 0; i < num_items; i++)
  {
    gaussian_weights_data[i] *= scaler;
  }
  status |= vxAddArrayItems(gaussian_weights_, (vx_size)num_items, (void *)&gaussian_weights_data[0], sizeof(gaussian_weights_data[0]));
  return status;
}
vx_status GraphModeStabilizer::initHomographyMatrices()
{
  vx_status status = VX_SUCCESS;
  vx_float32 homography_data[3][3] = { {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f} };
  for (vx_size i = 0; i < 2 * params_.smoothing_window_size; i++)
  {
    status |= vxCopyMatrix((vx_matrix)vxGetReferenceFromDelay(homography_matrices_, -i), homography_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
  }
  return status;
}
void GraphModeStabilizer::createMainGraph()
{
  main_graph_ = vxCreateGraph(context_);
  fast_corners_node_ = vxFastCornersNode(main_graph_,
                                         (vx_image)vxGetReferenceFromDelay(gray_frames_delay_, -1),
                                         params_.s_fast_threshold,
                                         vx_true_e,
                                         points_,
                                         0);
  color_convert_node_ = vxColorConvertNode( main_graph_,
                                            (vx_image)vxGetReferenceFromDelay(frames_delay_, 0),
                                            (vx_image)vxGetReferenceFromDelay(gray_frames_delay_, 0));
  gaussian_pyramid_node_ = vxGaussianPyramidNode(main_graph_,
                                                 (vx_image)vxGetReferenceFromDelay(gray_frames_delay_, 0),
                                                 (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0));
  opt_flow_node_ = vxOpticalFlowPyrLKNode(main_graph_,
                                          (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1),
                                          (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0),
                                          points_,
                                          points_,
                                          corresponding_points_,
                                          VX_TERM_CRITERIA_BOTH,
                                          params_.s_opt_flow_epsilon,
                                          params_.s_opt_flow_num_iterations,
                                          params_.s_opt_flow_use_initial_estimate,
                                          params_.opt_flow_win_size);
  find_homography_node_ = nvxFindHomographyNode(main_graph_,points_,
                                                corresponding_points_,
                                                (vx_matrix)vxGetReferenceFromDelay(homography_matrices_, 0),
                                                params_.homography_method,
                                                params_.homography_ransac_threshold,
                                                params_.homography_max_estimate_iters,
                                                params_.homography_max_refine_iters,
                                                params_.homography_confidence,
                                                params_.homography_outlier_ratio,
                                                NULL);
  registerHomographySmootherKernel(context_);
  homography_smoother_node_ = homographySmootherNode(main_graph_,
                                                     gaussian_weights_,
                                                     homography_matrices_,
                                                     perspective_matrix_);
  warp_perspective_node_ = vxWarpPerspectiveNode(main_graph_,
                                                 (vx_image)vxGetReferenceFromDelay(frames_delay_, -params_.smoothing_window_size),
                                                 perspective_matrix_,
                                                 VX_INTERPOLATION_TYPE_BILINEAR,
                                                 stabilized_frame_);
  //
  // Graph verification
  // Note: This verification is mandatory prior to graph execution
  //
  NVXIO_SAFE_CALL(vxVerifyGraph(main_graph_));
}
void GraphModeStabilizer::init(vx_image start_frame)
{
  params_.init(context_);
  gaussian_weights_ = vxCreateArray(context_, VX_TYPE_FLOAT32, (vx_size)(params_.smoothing_window_size * 2 + 1));
  NVXIO_CHECK_REFERENCE(gaussian_weights_);
  NVXIO_SAFE_CALL(initGaussianWeights());
  const int array_type = NVX_TYPE_KEYPOINTF;
  const vx_uint32 array_capacity = 15000;
  points_ = vxCreateArray(context_, array_type, array_capacity);
  NVXIO_CHECK_REFERENCE(points_);
  corresponding_points_ = vxCreateArray(context_, array_type, array_capacity);
  NVXIO_CHECK_REFERENCE(corresponding_points_);
  frames_delay_ = vxCreateDelay(context_, (vx_reference)start_frame, params_.smoothing_window_size + 1);
  NVXIO_CHECK_REFERENCE(frames_delay_);
  NVXIO_SAFE_CALL(nvxuCopyImage(context_, start_frame, (vx_image)vxGetReferenceFromDelay(frames_delay_, 0)));
  vx_uint32 width = 0;
  vx_uint32 height = 0;
  NVXIO_SAFE_CALL(vxQueryImage(start_frame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
  NVXIO_SAFE_CALL(vxQueryImage(start_frame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
  vx_image gray_start_frame = vxCreateImage(context_, width, height, VX_DF_IMAGE_U8);
  NVXIO_CHECK_REFERENCE(gray_start_frame);
  gray_frames_delay_ = vxCreateDelay(context_, (vx_reference)gray_start_frame, 2);
  NVXIO_CHECK_REFERENCE(gray_frames_delay_);
  NVXIO_SAFE_CALL(vxuColorConvert(context_, start_frame, (vx_image)vxGetReferenceFromDelay(gray_frames_delay_, -1)));
  vx_pyramid pyramid = vxCreatePyramid(context_,
                                       (vx_size)params_.num_pyramid_levels,
                                       VX_SCALE_PYRAMID_HALF,
                                       width,
                                       height,
                                       VX_DF_IMAGE_U8);
  NVXIO_CHECK_REFERENCE(pyramid);
  pyr_delay_ = vxCreateDelay(context_, (vx_reference)pyramid, 2);
  NVXIO_CHECK_REFERENCE(pyr_delay_);
  NVXIO_SAFE_CALL(vxuGaussianPyramid(context_, gray_start_frame, (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1)));
  vx_matrix homography = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 3);
  NVXIO_CHECK_REFERENCE(homography);
  homography_matrices_ = vxCreateDelay(context_, (vx_reference)homography, 2 * params_.smoothing_window_size);
  NVXIO_CHECK_REFERENCE(homography_matrices_);
  NVXIO_SAFE_CALL(initHomographyMatrices());
  perspective_matrix_ = vxCreateMatrix(context_, VX_TYPE_FLOAT32, 3, 3);
  NVXIO_CHECK_REFERENCE(perspective_matrix_);
  stabilized_frame_ = vxCreateImage(context_, width, height, VX_DF_IMAGE_RGBX);
  NVXIO_CHECK_REFERENCE(stabilized_frame_);
  createMainGraph(/*current_frame_*/);
  vxReleasePyramid(&pyramid);
  vxReleaseImage(&gray_start_frame);
  vxReleaseMatrix(&homography);
}
vx_image GraphModeStabilizer::process(vx_image current_frame)
{
  // Initialization
  if (!frames_delay_)
  {
    init(current_frame);
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vxQueryImage(current_frame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width));
    vxQueryImage(current_frame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height));
    return vxCreateImage(context_, width, height, VX_DF_IMAGE_RGBX);
  }
  NVXIO_SAFE_CALL(vxAgeDelay(homography_matrices_));
  NVXIO_SAFE_CALL(vxAgeDelay(pyr_delay_));
  NVXIO_SAFE_CALL(vxAgeDelay(gray_frames_delay_));
  NVXIO_SAFE_CALL(vxAgeDelay(frames_delay_));
  NVXIO_SAFE_CALL(nvxuCopyImage(context_, current_frame, (vx_image)vxGetReferenceFromDelay(frames_delay_, 0)));
  NVXIO_SAFE_CALL(vxProcessGraph(main_graph_));
  return stabilized_frame_;
}

