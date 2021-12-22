# Add sources
target_sources(openvslam_orb
        PRIVATE

        ${OPENVSLAM_DIR}/src/openvslam/feature/orb_params.h
        ${OPENVSLAM_DIR}/src/openvslam/feature/orb_extractor.h
        ${OPENVSLAM_DIR}/src/openvslam/feature/orb_extractor_node.h
        ${OPENVSLAM_DIR}/src/openvslam/feature/orb_params.cc
        ${OPENVSLAM_DIR}/src/openvslam/feature/orb_extractor.cc
        ${OPENVSLAM_DIR}/src/openvslam/feature/orb_extractor_node.cc)

