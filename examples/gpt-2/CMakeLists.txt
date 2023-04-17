#
# gpt-2

set(LIBRARY_TARGET gpt-2-core)
add_library(${LIBRARY_TARGET} SHARED gpt-2.cpp)
target_link_libraries(${LIBRARY_TARGET} PRIVATE ggml ggml_utils)

set(TEST_TARGET gpt-2)
add_executable(${TEST_TARGET} gpt-2.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)

#
# gpt-2-quantize

set(TEST_TARGET gpt-2-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)
