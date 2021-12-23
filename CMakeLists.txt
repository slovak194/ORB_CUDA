cmake_minimum_required(VERSION 3.17)

execute_process(
        COMMAND bash -c "uname -m"
        OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_PROCESSOR
)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
else (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.0/bin/nvcc")
endif (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")

message(STATUS "Build for: ${CMAKE_HOST_SYSTEM_PROCESSOR}")

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
set(CMAKE_CUDA_FLAGS "--compiler-options -fno-strict-aliasing -use_fast_math -lineinfo ${CMAKE_CUDA_FLAGS}")

# TODO, replace with
#target_compile_options(target PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#        --generate-line-info
#        --use_fast_math
#        --relocatable-device-code=true
#        >)

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


if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "72")
else (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "50")
endif (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")


target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        nvToolsExt
        ${CUDA_CUDA_LIBRARY}
        )

add_subdirectory(test)


#add_subdirectory(/home/slovak/openvslam_community /home/slovak/openvslam_community/build)
