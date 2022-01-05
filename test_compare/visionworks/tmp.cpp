/*
 * Minimal example of VisionWorks problem with Tegra System Profiler.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <exception>

#include <NVX/nvx.h>
#include <NVX/Utility.hpp>
#include <OVX/UtilityOVX.hpp>

using namespace std;

int main(int argc, char **argv)
{
  int width; int height;
  if(argc < 3)
  {
    width  = 1226;
    height = 370;
  }
  else
  {
    try
    {
      width  = std::stoi(argv[1]);
      height = std::stoi(argv[2]);
    }
    catch(std::exception& e)
    {
      std::cout << "Launch: " << argv[0] << " width height" << std::endl;
      exit(1);
    }
  }
  // context
  vx_context context;
  // graph
  vx_graph global_graph;
  // images
  vx_image image_start;
  vx_image image_level_one;
  // scalars
  vx_scalar s_strength_threshold;
  vx_scalar corners;
  // array
  vx_array keypoints;
  // pyramid
  vx_pyramid gaussian_pyramid;

  context = vxCreateContext();
  NVXIO_CHECK_REFERENCE( context );
  vxDirective((vx_reference)context, NVX_DIRECTIVE_ENABLE_PERFORMANCE);

  vx_pixel_value_t p;
  p.U8 = 0;
  image_start      = vxCreateUniformImage(context, width, height, VX_DF_IMAGE_U8, &p);
  NVXIO_CHECK_REFERENCE(image_start);

  global_graph = vxCreateGraph(context);
  NVXIO_CHECK_REFERENCE( global_graph );

  float fast_t = 10;
  s_strength_threshold = vxCreateScalar(context, VX_TYPE_FLOAT32, &fast_t);
  NVXIO_CHECK_REFERENCE( s_strength_threshold );


  // allocate some space to store the keypoints
  keypoints = vxCreateVirtualArray(global_graph, VX_TYPE_KEYPOINT, 400);
  NVXIO_CHECK_REFERENCE( keypoints );
  vx_size sz_corners = 0;
  corners = vxCreateScalar(context, VX_TYPE_SIZE, &sz_corners);
  NVXIO_CHECK_REFERENCE( corners );



  // build the pyramid
  gaussian_pyramid = vxCreateVirtualPyramid(global_graph, 2, VX_SCALE_PYRAMID_ORB, width, height, VX_DF_IMAGE_U8);
  NVXIO_CHECK_REFERENCE( gaussian_pyramid );
  vxGaussianPyramidNode(global_graph, image_start, gaussian_pyramid);
  // get reference to the second level
  image_level_one = vxGetPyramidLevel(gaussian_pyramid, 1);
  NVXIO_CHECK_REFERENCE( image_level_one );
  // do the FAST corners detection




  vxFastCornersNode(global_graph, image_level_one, s_strength_threshold, vx_true_e, keypoints, corners);



  NVXIO_SAFE_CALL( vxVerifyGraph(global_graph) );

  // process 2 times the graph to arise the issue!
  for(int i = 0; i < 2; i++)
  {
    vxProcessGraph(global_graph);
  }

  vxReleaseImage(&image_level_one);
  vxReleaseImage(&image_start);

  vxReleaseScalar(&corners);
  vxReleaseScalar(&s_strength_threshold);

  vxReleasePyramid(&gaussian_pyramid);

  vxReleaseArray(&keypoints);

  vxReleaseGraph(&global_graph);
  vxReleaseContext(&context);

  return 0;

