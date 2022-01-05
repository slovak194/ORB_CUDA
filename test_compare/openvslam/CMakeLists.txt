set(OPENVSLAM_DIR $ENV{HOME}/openvslam_community)

add_subdirectory(openvslam_orb)

add_executable(test_compare_openvslam test_compare_openvslam.cpp)

target_link_libraries(test_compare_openvslam PUBLIC
        orb_cuda
        openvslam_orb
        )
