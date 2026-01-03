# tests/CMakeLists.txt

add_executable(test_d20_omp
        test_d20_omp.cpp
)

target_link_libraries(test_d20_omp
        PRIVATE
        MCEngine
)

target_include_directories(test_d20_omp
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
        NAME omp_d20_mean_test
        COMMAND test_d20_omp
)
