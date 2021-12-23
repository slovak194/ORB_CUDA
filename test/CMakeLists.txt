add_executable(test_extraction test_extraction.cpp)
target_link_libraries(test_extraction PUBLIC orb_cuda)

add_executable(test_perf test_perf.cpp)
target_link_libraries(test_perf PUBLIC orb_cuda)
