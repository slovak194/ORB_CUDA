
find_package(VisionWorks QUIET)
#find_package(VisionWorks-NVXIO REQUIRED) # TODO, OLSLO, fix abs paths.

if (${VisionWorks_FOUND})

    find_package(CUDA 10.2 REQUIRED)

    set(WITH_CUDA ON)

    message("VisionWorks_FOUND: ${VisionWorks_FOUND}")
    message("VisionWorks_LIBRARIES: ${VisionWorks_LIBRARIES}")
    message("VisionWorks_INCLUDE_DIRS: ${VisionWorks_INCLUDE_DIRS}")
    message("VisionWorks_LIBRARIES: ${VisionWorks_LIBRARIES}")

    #set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-lvisionworks -lovx -lnvx -lpthread")
    #set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGE} "-lgstpbutils-1.0 -lgstapp-1.0 -lgstbase-1.0 -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0")
    #set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGE} "/usr/share/visionworks/sources/3rdparty/glfw3/libs/libglfw3.a")
    #set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGE} "-lXrandr -lXi -lXxf86vm -lX11 -lGL")

    add_executable(test_vis_works_perf test_compare_nvx_perf.cpp)

    target_compile_definitions(test_vis_works_perf PUBLIC PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

    # TODO, OLSLO, fix abs paths.

    target_link_libraries(test_vis_works_perf
            ${OpenCV_LIBS}
            ${VisionWorks_LIBRARIES}
            /home/nvidia/VisionWorks-1.6-Samples/libs/aarch64/linux/release/libovx.a
            /home/nvidia/VisionWorks-1.6-Samples/libs/aarch64/linux/release/libnvx.a
            )

    target_include_directories(test_vis_works_perf PRIVATE
            ${OpenCV_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            ${VisionWorks_INCLUDE_DIRS}
            /usr/share/visionworks/sources/nvxio/include
            )

    add_executable(test_vis_works test_compare_nvx.cpp)

    target_link_libraries(test_vis_works
            ${OpenCV_LIBS}
            ${VisionWorks_LIBRARIES}
            /home/nvidia/VisionWorks-1.6-Samples/libs/aarch64/linux/release/libovx.a
            /home/nvidia/VisionWorks-1.6-Samples/libs/aarch64/linux/release/libnvx.a
            )

    target_include_directories(test_vis_works PRIVATE
            ${OpenCV_INCLUDE_DIRS}
            ${VisionWorks_INCLUDE_DIRS}
            /usr/share/visionworks/sources/nvxio/include
            )

    add_executable(test_compare_nvx_instant test_compare_nvx_instant.cpp)

    target_link_libraries(test_compare_nvx_instant
            ${OpenCV_LIBS}
            ${VisionWorks_LIBRARIES}
            /home/nvidia/VisionWorks-1.6-Samples/libs/aarch64/linux/release/libovx.a
            /home/nvidia/VisionWorks-1.6-Samples/libs/aarch64/linux/release/libnvx.a
            )

    target_include_directories(test_compare_nvx_instant PRIVATE
            ${OpenCV_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            ${VisionWorks_INCLUDE_DIRS}
            /usr/share/visionworks/sources/nvxio/include
            )

#    add_executable(vstab_demo main.cpp
#            homography_smoother_node.cpp homography_smoother_node.hpp
#            graph_mode_stabilizer.cpp graph_mode_stabilizer.hpp
#            )
#
#    target_include_directories(vstab_demo PRIVATE
#            ${OpenCV_INCLUDE_DIRS}
#            ${CUDA_INCLUDE_DIRS}
#            ${VisionWorks_INCLUDE_DIRS}
#            /usr/share/visionworks/sources/nvxio/include
#            )
#
#    target_link_libraries(vstab_demo
#            ${OpenCV_LIBS}
#            ${VisionWorks_LIBRARIES}
#            ${CUDA_LIBRARIES}
#            ${CUDA_CUDART_LIBRARY}
#            /home/nvidia/VisionWorks-1.6-Samples/libs/aarch64/linux/release/libovx.a
#            /home/nvidia/VisionWorks-1.6-Samples/libs/aarch64/linux/release/libnvx.a
#            glfw
#            /usr/share/visionworks/sources/3rdparty/freetype/libs/libfreetype.a
#            gstpbutils-1.0
#            gstaudio-1.0
#            gstvideo-1.0
#            gstapp-1.0
#            gstbase-1.0
#            gstreamer-1.0
#            gobject-2.0
#            glib-2.0
#            GL
#            EGL
#            pthread
#            )



else (${VisionWorks_FOUND})

    message("VisionWorks               OFF")

endif (${VisionWorks_FOUND})
