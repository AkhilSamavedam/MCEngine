# tests/CMakeLists.txt

function(add_mc_test target src use_cuda)
        add_executable(${target} ${src})
        target_link_libraries(${target} PRIVATE MCEngine)
        target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)

        if (use_cuda)
                set_source_files_properties(${src} PROPERTIES LANGUAGE CUDA)
                target_compile_definitions(${target} PRIVATE MCENGINE_ENABLE_CUDA=1)
        endif()
endfunction()

add_mc_test(test_d20_omp test_d20_omp.cpp OFF)
add_mc_test(test_d20_cuda test_d20_cuda.cpp ON)
add_mc_test(test_dice_until_6_omp test_dice_until_6_omp.cpp OFF)
add_mc_test(test_dice_until_6_cuda test_dice_until_6_cuda.cpp ON)


add_test(
        NAME omp_d20_mean
        COMMAND test_d20_omp
)

add_test(
        NAME cuda_d20_mean
        COMMAND test_d20_cuda
)

add_test(
        NAME omp_dice_until_6
        COMMAND test_dice_until_6_omp
)

add_test(
        NAME cuda_dice_until_6
        COMMAND test_dice_until_6_cuda
)
