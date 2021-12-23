import numpy as np


import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


track_times_fiel_path = "build/track_times_0.txt"

with open(track_times_fiel_path) as ff:
    track_times = np.fromfile(ff, sep='\n')

plt.plot(1/track_times, ".")
plt.plot(0, 0)

track_times_fiel_path = "build/track_times_1.txt"

with open(track_times_fiel_path) as ff:
    track_times = np.fromfile(ff, sep='\n')

plt.plot(1/track_times, ".")
plt.plot(0, 0)

# %%

# nvprof --export-profile timeline.prof app
#
# sudo nvprof -f --metrics achieved_occupancy,ipc -o metrics.prof test/test_perf
# sudo /usr/local/cuda-11.0/bin/nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java ./metrics.prof
#
# $ ./deviceQuery
# ./deviceQuery Starting...
#
# CUDA Device Query (Runtime API) version (CUDART static linking)
#
# Detected 1 CUDA Capable device(s)
#
# Device 0: "Xavier"
# CUDA Driver Version / Runtime Version          10.0 / 10.0
# CUDA Capability Major/Minor version number:    7.2
# Total amount of global memory:                 15820 MBytes (16588668928 bytes)
# ( 8) Multiprocessors, ( 64) CUDA Cores/MP:     512 CUDA Cores
# GPU Max Clock rate:                            1500 MHz (1.50 GHz)
# Memory Clock rate:                             1500 Mhz
# Memory Bus Width:                              256-bit
# L2 Cache Size:                                 524288 bytes
# Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
# Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
# Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
# Total amount of constant memory:               65536 bytes
# Total amount of shared memory per block:       49152 bytes
# Total number of registers available per block: 65536
# Warp size:                                     32
# Maximum number of threads per multiprocessor:  2048
# Maximum number of threads per block:           1024
# Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
# Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
# Maximum memory pitch:                          2147483647 bytes
# Texture alignment:                             512 bytes
# Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
# Run time limit on kernels:                     No
# Integrated GPU sharing Host Memory:            Yes
# Support host page-locked memory mapping:       Yes
# Alignment requirement for Surfaces:            Yes
# Device has ECC support:                        Disabled
# Device supports Unified Addressing (UVA):      Yes
# Device supports Compute Preemption:            Yes
# Supports Cooperative Kernel Launch:            Yes
# Supports MultiDevice Co-op Kernel Launch:      Yes
# Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 0
# Compute Mode:
# < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
#
# deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.0, CUDA Runtime Version = 10.0, NumDevs = 1
# Result = PASS

# ~ /usr/local/cuda-11.0/extras/demo_suite/deviceQuery
# /usr/local/cuda-11.0/extras/demo_suite/deviceQuery Starting...
#
# CUDA Device Query (Runtime API) version (CUDART static linking)
#
# Detected 1 CUDA Capable device(s)
#
# Device 0: "GeForce 940MX"
# CUDA Driver Version / Runtime Version          11.2 / 11.0
# CUDA Capability Major/Minor version number:    5.0
# Total amount of global memory:                 2004 MBytes (2101870592 bytes)
# ( 3) Multiprocessors, (128) CUDA Cores/MP:     384 CUDA Cores
# GPU Max Clock rate:                            1189 MHz (1.19 GHz)
# Memory Clock rate:                             2505 Mhz
# Memory Bus Width:                              64-bit
# L2 Cache Size:                                 1048576 bytes
# Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
# Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
# Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
# Total amount of constant memory:               65536 bytes
# Total amount of shared memory per block:       49152 bytes
# Total number of registers available per block: 65536
# Warp size:                                     32
# Maximum number of threads per multiprocessor:  2048
# Maximum number of threads per block:           1024
# Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
# Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
# Maximum memory pitch:                          2147483647 bytes
# Texture alignment:                             512 bytes
# Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
# Run time limit on kernels:                     Yes
# Integrated GPU sharing Host Memory:            No
# Support host page-locked memory mapping:       Yes
# Alignment requirement for Surfaces:            Yes
# Device has ECC support:                        Disabled
# Device supports Unified Addressing (UVA):      Yes
# Device supports Compute Preemption:            No
# Supports Cooperative Kernel Launch:            No
# Supports MultiDevice Co-op Kernel Launch:      No
# Device PCI Domain ID / Bus ID / location ID:   0 / 2 / 0
# Compute Mode:
# < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
#
# deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.2, CUDA Runtime Version = 11.0, NumDevs = 1, Device0 = GeForce 940MX
# Result = PASS

# 1024 * (2048 / 1024) * 3
# Out[4]: 6144.0
# 1024 * (2048 / 1024) * 8
# Out[5]: 16384.0

# ORBextractor::extract->
#     ORBextractor::ComputeKeyPointsOctTree->
#         GpuFast::detectAsync->
#             tileCalcKeypoints_kernel
