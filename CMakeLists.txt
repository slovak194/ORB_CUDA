cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.0/bin/nvcc")

project(orb_cuda LANGUAGES C CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

find_package(OpenCV COMPONENTS)
find_package(Eigen3 REQUIRED NO_MODULE)

find_package(CUDA REQUIRED)

#set(CMAKE_CUDA_FLAGS "--verbose --compiler-options -fno-strict-aliasing -use_fast_math ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "--compiler-options -fno-strict-aliasing -use_fast_math ${CMAKE_CUDA_FLAGS}")

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)


add_library(${PROJECT_NAME} SHARED
        src/ORBextractor.cc
        src/cuda/Allocator_gpu.cu
        src/cuda/Fast_gpu.cu
        src/cuda/Orb_gpu.cu
        src/cuda/Cuda.cu
        )

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "50")

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        nvToolsExt
        ${CUDA_CUDA_LIBRARY}
        )

add_subdirectory(test)
