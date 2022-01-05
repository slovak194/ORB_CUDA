#cmake_minimum_required(VERSION 3.17)
cmake_minimum_required(VERSION 3.10)

project(orb_cuda LANGUAGES C CXX CUDA)

include(cmake/cuda_helpers.cmake)
detect_cuda_arch()

include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

find_package(OpenCV REQUIRED)

#set(CMAKE_CUDA_FLAGS "--verbose --compiler-options -fno-strict-aliasing -use_fast_math -lineinfo ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "--compiler-options -fno-strict-aliasing -use_fast_math ${CMAKE_CUDA_FLAGS}")

add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")


add_library(${PROJECT_NAME} SHARED
        src/ORBextractor.cc
        src/ORBextractorImpl.cc
        src/cuda/Allocator_gpu.cu
        src/cuda/Fast_gpu.cu
        src/cuda/Orb_gpu.cu
        src/cuda/Cuda.cu
        )

target_include_directories(${PROJECT_NAME} PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        )

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        )

set_target_properties(${PROJECT_NAME}
        PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
        PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/ORBextractor.h
        )

install(TARGETS ${PROJECT_NAME}
        PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
        LIBRARY DESTINATION lib
        )

add_subdirectory(test)
add_subdirectory(test_compare)

MESSAGE(STATUS)
MESSAGE(STATUS "---------------------------------")
