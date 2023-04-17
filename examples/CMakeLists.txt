add_library(ggml_utils STATIC utils.cpp)
target_include_directories(ggml_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(ggml_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(gpt-2)
add_subdirectory(gpt-j)
add_subdirectory(whisper)
add_subdirectory(mnist)
