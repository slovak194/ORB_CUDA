#include "homography_smoother_node.hpp"
#include "opencv2/core/core.hpp"
#include "NVX/nvx_opencv_interop.hpp"
static cv::Matx33f getTransformation(const std::vector<cv::Matx33f> &homography_matrices, int from, int to)
{
  cv::Matx33f transformation = cv::Matx33f::eye();
  if (to > from)
  {
    for (int i = from; i < to; ++i)
    {
      transformation = transformation * homography_matrices[i];
    }
  }
  else if (to < from)
  {
    for (int i = to; i < from; ++i)
    {
      transformation = transformation * homography_matrices[i];
    }
    transformation = transformation.inv();
  }
  return transformation;
}
static cv::Matx33f getSmoothedHomography(const std::vector<float> &gauss_weights,
                                         const std::vector<cv::Matx33f> &homography_matrices)
{
  int array_size = gauss_weights.size();
  int size = homography_matrices.size();
  CV_Assert(size % 2 == 0);
  int idx = size / 2;
  cv::Matx33f transform = cv::Matx33f::zeros();
  for (int i = 0; i < array_size; ++i)
  {
    transform += gauss_weights[i] * getTransformation(homography_matrices, idx, i);
  }
  return transform;
}
static vx_status VX_CALLBACK homography_smoother_kernel(vx_node /*node*/,
                                                        const vx_reference *parameters,
                                                        vx_uint32 num)
{
  if (num != 3)
    return VX_FAILURE;
  vx_status status = VX_SUCCESS;
  vx_array gaussian_weights = (vx_array)parameters[0];
  vx_delay homography_matrices = (vx_delay)parameters[1];
  vx_matrix transformation = (vx_matrix)parameters[2];
  // converting vx_array gaussian_weights to std::vector<float> cv_gaussian_weights
  vx_size gaussian_weights_count = 0;
  status |= vxQueryArray(gaussian_weights, VX_ARRAY_ATTRIBUTE_NUMITEMS, &gaussian_weights_count, sizeof(gaussian_weights_count));
  std::vector<float> cv_gaussian_weights;
  cv_gaussian_weights.resize((int)gaussian_weights_count);
  vx_size i, stride = sizeof(vx_size);
  void *base;
  vx_map_id map_id;
  status |= vxMapArrayRange(gaussian_weights, 0, gaussian_weights_count, &map_id, &stride, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
  for (i = 0; i < gaussian_weights_count; i++)
  {
    vx_float32 cur_item = vxArrayItem(vx_float32, base, i, stride);
    cv_gaussian_weights[i] = (float)cur_item;
  }
  status |= vxUnmapArrayRange(gaussian_weights, map_id);
  // converting vx_delay of matrices to std::vector<cv::Matx33f> cv_homography_matrices
  vx_size homography_matrices_count = 0;
  status |= vxQueryDelay(homography_matrices, VX_DELAY_ATTRIBUTE_SLOTS, &homography_matrices_count, sizeof(homography_matrices_count));
  std::vector<cv::Matx33f> cv_homography_matrices;
  cv_homography_matrices.resize(homography_matrices_count);
  for (i = 0; i < homography_matrices_count; i++)
  {
    vx_matrix matrix = (vx_matrix)vxGetReferenceFromDelay(homography_matrices, i - homography_matrices_count + 1);
    CV_Assert(vxCopyMatrix(matrix, cv_homography_matrices[i].val, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) == VX_SUCCESS);
  }
  cv::Matx33f cv_transformation = getSmoothedHomography(cv_gaussian_weights, cv_homography_matrices);
  cv_transformation = cv_transformation.inv();
  nvx_cv::copyCVMatToVXMatrix(cv::Mat(cv_transformation), transformation);
  return status;
}
static vx_status VX_CALLBACK homography_smoother_validate(vx_node node, const vx_reference parameters[],
                                                          vx_uint32 num, vx_meta_format metas[])
{
  if ( num != 3 ) return VX_ERROR_INVALID_PARAMETERS;
  vx_array array = (vx_array)parameters[0];
  vx_delay delay = (vx_delay)parameters[1];
  // gaussian_weights array input
  vx_enum item_type = 0;
  vx_size num_array_items = 0;
  if (vxQueryArray(array, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &item_type, sizeof(item_type)) != VX_SUCCESS ||
      vxQueryArray(array, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_array_items, sizeof(num_array_items)) != VX_SUCCESS )
  {
    return VX_ERROR_INVALID_PARAMETERS;
  }
  if (item_type != VX_TYPE_FLOAT32)
  {
    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                  "invalid type of gaussian_weights array (expected type: VX_TYPE_FLOAT32)");
    return VX_ERROR_INVALID_PARAMETERS;
  }
  if (num_array_items % 2 != 1)
  {
    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                  "invalid number of elements of gaussian_weights array (expected number is odd)");
    return VX_ERROR_INVALID_PARAMETERS;
  }
  // homography_matrices delay input
  vx_enum delay_type = 0;
  vx_size num_delay_items = 0;
  if (vxQueryDelay(delay, VX_DELAY_ATTRIBUTE_TYPE, &delay_type, sizeof(delay_type)) != VX_SUCCESS ||
      vxQueryDelay(delay, VX_DELAY_ATTRIBUTE_SLOTS, &num_delay_items, sizeof(num_delay_items)) != VX_SUCCESS)
  {
    return VX_ERROR_INVALID_PARAMETERS;
  }
  if (delay_type != VX_TYPE_MATRIX)
  {
    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                  "invalid type of homography_matrices delay (expected type: VX_TYPE_MATRIX)");
    return VX_ERROR_INVALID_PARAMETERS;
  }
  if(num_delay_items != (num_array_items - 1))
  {
    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                  "invalid number of delay slots"
                  "(expected number is one less then number of gaussian_weights array elements)");
    return VX_ERROR_INVALID_PARAMETERS;
  }
  vx_matrix matrix = (vx_matrix)vxGetReferenceFromDelay(delay, 0);
  vx_enum data_type = 0;
  vx_size rows = 0ul, cols = 0ul;
  if (vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_TYPE, &data_type, sizeof(data_type)) != VX_SUCCESS ||
      vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)) != VX_SUCCESS ||
      vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols, sizeof(cols)) != VX_SUCCESS )
  {
    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                  "bad reference to the matrix in homography_matrices delay");
    return VX_ERROR_INVALID_PARAMETERS;
  }
  if ( (data_type != VX_TYPE_FLOAT32) || (cols != 3) || (rows != 3) )
  {
    vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_PARAMETERS,
                  "invalid type/size of matrix in homography_matrices delay"
                  "(expected type: VX_TYPE_FLOAT32, expected rows: 3, expected cols: 3)");
    return VX_ERROR_INVALID_PARAMETERS;
  }
  // transformation matrix output
  vxSetMetaFormatAttribute(metas[2], VX_MATRIX_ATTRIBUTE_COLUMNS, &cols, sizeof(cols));
  vxSetMetaFormatAttribute(metas[2], VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows));
  vxSetMetaFormatAttribute(metas[2], VX_MATRIX_ATTRIBUTE_TYPE, &data_type, sizeof(data_type));
  return VX_SUCCESS;
}
vx_status registerHomographySmootherKernel(vx_context context)
{
  vx_status status = VX_SUCCESS;
  vx_kernel kernel = vxAddUserKernel(context,
                                     "user.kernel.homography_smoother",
                                     USER_KERNEL_HOMOGRAPHY_SMOOTHER,
                                     homography_smoother_kernel,
                                     3,
                                     homography_smoother_validate,
                                     NULL,
                                     NULL);
  status = vxGetStatus((vx_reference)kernel);
  if(status != VX_SUCCESS)
  {
    vxAddLogEntry((vx_reference)context,
                  status,
                  "[%s:%u] failed to create HomographySmoother Kernel",
                  __FUNCTION__,
                  __LINE__);
    return status;
  }
  status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED); // gaussian_weights
  status |= vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_DELAY, VX_PARAMETER_STATE_REQUIRED); // homography_matrices
  status |= vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED); // transformation
  if (status != VX_SUCCESS)
  {
    vxReleaseKernel(&kernel);
    vxAddLogEntry((vx_reference)context, status, "Failed to initialize HomographySmoother Kernel parameters");
    return VX_FAILURE;
  }
  status |= vxFinalizeKernel(kernel);
  vxReleaseKernel(&kernel);
  if (status != VX_SUCCESS)
  {
    vxAddLogEntry((vx_reference)context, status, "Failed to finalize HomographySmoother Kernel");
    return VX_FAILURE;
  }
  return status;
}
vx_node homographySmootherNode(vx_graph graph,
                               vx_array gaussian_weights,
                               vx_delay homography_matrices,
                               vx_matrix transformation)
{
  vx_node node = NULL;
  vx_kernel kernel = vxGetKernelByEnum(vxGetContext((vx_reference)graph), USER_KERNEL_HOMOGRAPHY_SMOOTHER);
  if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
  {
    node = vxCreateGenericNode(graph, kernel);
    vxReleaseKernel(&kernel);
    if (vxGetStatus((vx_reference)node) == VX_SUCCESS)
    {
      vxSetParameterByIndex(node, 0, (vx_reference)gaussian_weights);
      vxSetParameterByIndex(node, 1, (vx_reference)homography_matrices);
      vxSetParameterByIndex(node, 2, (vx_reference)transformation);
    }
  }
  return node;
}