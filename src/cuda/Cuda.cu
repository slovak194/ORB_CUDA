#include <helper_cuda.h>
#include <cuda/Cuda.hpp>

namespace ORB_CUDA { namespace cuda {
  void deviceSynchronize() {
    checkCudaErrors( cudaDeviceSynchronize() );
  }
} }
