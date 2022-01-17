//
// Created by slovak on 1/17/22.
//

#ifndef _GRAPH_MODE_STABILIZER_
#define _GRAPH_MODE_STABILIZER_
#include "NVX/nvx.h"
struct GraphModeStabilizerParams
{
  vx_size smoothing_window_size;
  vx_float32 fast_threshold;
  vx_int32 num_pyramid_levels;
  int opt_flow_win_size;
  float opt_flow_epsilon;
  int opt_flow_num_iterations;
  int opt_flow_use_initial_estimate;
  float homography_ransac_threshold;
  int homography_max_estimate_iters;
  int homography_max_refine_iters;
  float homography_confidence;
  float homography_outlier_ratio;
  vx_scalar s_fast_threshold;
  vx_scalar s_opt_flow_epsilon;
  vx_scalar s_opt_flow_num_iterations;
  vx_scalar s_opt_flow_use_initial_estimate;
  vx_enum homography_method;
  void init(vx_context context);
  ~GraphModeStabilizerParams();
};
class GraphModeStabilizer
{
 public:
  GraphModeStabilizer(vx_context context);
  virtual ~GraphModeStabilizer();
  virtual vx_image process(vx_image current_frame);
 private:
  virtual void init(vx_image start_frame);
  void createMainGraph();
  vx_status initGaussianWeights();
  vx_status initHomographyMatrices();
  GraphModeStabilizerParams params_;
  vx_context context_;
  vx_array points_;
  vx_array corresponding_points_;
  vx_array gaussian_weights_;
  vx_delay homography_matrices_;
  vx_matrix homography_;
  vx_matrix perspective_matrix_;
  vx_image stabilized_frame_;
  vx_delay gray_frames_delay_;
  vx_delay pyr_delay_;
  vx_delay frames_delay_;
  vx_graph main_graph_;
  vx_node fast_corners_node_;
  vx_node color_convert_node_;
  vx_node gaussian_pyramid_node_;
  vx_node opt_flow_node_;
  vx_node find_homography_node_;
  vx_node homography_smoother_node_;
  vx_node warp_perspective_node_;
};
#endif
