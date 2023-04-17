#
# gpt-j

set(LIBRARY_TARGET gpt-j-core)
add_library(${LIBRARY_TARGET} SHARED gpt-j.cpp)
target_link_libraries(${LIBRARY_TARGET} PRIVATE ggml ggml_utils)

set(TEST_TARGET gpt-j)
add_executable(${TEST_TARGET} gpt-j.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)


#
# gpt-j-quantize

set(TEST_TARGET gpt-j-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)
