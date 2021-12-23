#include <helper_cuda.h>
#include <orb_cuda/cuda/Cuda.hpp>

namespace ORB_CUDA { namespace cuda {
  void deviceSynchronize() {
    checkCudaErrors( cudaDeviceSynchronize() );
  }
} }
