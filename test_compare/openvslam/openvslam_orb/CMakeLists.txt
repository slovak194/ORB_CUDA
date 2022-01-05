find_package(Eigen3 REQUIRED NO_MODULE)

find_package(OpenMP REQUIRED)
if (NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif ()

add_library(openvslam_orb
        ${OPENVSLAM_DIR}/src/openvslam/type.h
        )

# OpenMP
target_compile_options(openvslam_orb PRIVATE ${OpenMP_CXX_FLAGS})
set(USE_OPENMP ON CACHE BOOL "Use OpenMP")
if (USE_OPENMP)
    target_compile_definitions(openvslam_orb PUBLIC USE_OPENMP)
    message(STATUS "OpenMP: ENABLED")
else ()
    message(STATUS "OpenMP: DISABLED")
endif ()

if (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
else (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(USE_SSE_ORB ON CACHE BOOL "Enable SSE3 instruction for ORB extraction")
    if (USE_SSE_ORB)
        target_compile_options(openvslam_orb PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-msse3>)
        target_compile_definitions(openvslam_orb PUBLIC USE_SSE_ORB)
        message(STATUS "SSE3 for ORB extraction: ENABLED")
    else ()
        message(STATUS "SSE3 for ORB extraction: DISABLED")
    endif ()

    set(USE_SSE_FP_MATH ON CACHE BOOL "Enable SSE instruction for floating-point operation")
    if (USE_SSE_FP_MATH)
        target_compile_options(openvslam_orb PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mfpmath=sse>)
        message(STATUS "SSE for floating-point operation: ENABLED")
    else ()
        message(STATUS "SSE for floating-point operation: DISABLED")
    endif ()
endif (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")

# Include directories
target_include_directories(openvslam_orb
        PUBLIC
        ${OPENVSLAM_DIR}/src
        )

# Link to required libraries
target_link_libraries(openvslam_orb
        PUBLIC
        Threads::Threads
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        yaml-cpp
        opencv_core
        opencv_features2d
        opencv_calib3d
        PRIVATE
        ${LAPACK_LIBRARIES}
        )

add_subdirectory(feature)
add_subdirectory(util)
