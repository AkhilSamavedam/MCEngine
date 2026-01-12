# tests/CMakeLists.txt

add_executable(test_d20_omp
        test_d20_omp.cpp
)

add_executable(test_d20_cuda
        test_d20_cuda.cpp
)

set_source_files_properties(
        test_d20_cuda.cpp
        PROPERTIES
        LANGUAGE CUDA
)

target_link_libraries(test_d20_omp
        PRIVATE
        MCEngine
)

target_link_libraries(test_d20_cuda
        PRIVATE
        MCEngine
)

target_include_directories(test_d20_omp
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(test_d20_cuda
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)


add_test(
        NAME omp_d20_mean_test
        COMMAND test_d20_omp
)

add_test(
        NAME cuda_d20_mean_test
        COMMAND test_d20_cuda
)
