//
// Created by slovak on 1/17/22.
//

#ifndef ORB_CUDA_TEST_COMPARE_VISIONWORKS_HOMOGRAPHY_SMOOTHER_NODE_HPP_
#define ORB_CUDA_TEST_COMPARE_VISIONWORKS_HOMOGRAPHY_SMOOTHER_NODE_HPP_

#include "NVX/nvx.h"
enum
{
  USER_LIBRARY = 0x1,
  USER_KERNEL_HOMOGRAPHY_SMOOTHER = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x0
};
vx_status registerHomographySmootherKernel(vx_context context);
vx_node homographySmootherNode(vx_graph graph,
                               vx_array gaussian_weights,
                               vx_delay homography_matrices,
                               vx_matrix transformation);

#endif //ORB_CUDA_TEST_COMPARE_VISIONWORKS_HOMOGRAPHY_SMOOTHER_NODE_HPP_
