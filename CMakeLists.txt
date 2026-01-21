cmake_minimum_required(VERSION 4.1)

project(MCEngine
        VERSION 0.1
        LANGUAGES CXX
)

# ----------------------------
# Language standards
# ----------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------
# Options
# ----------------------------
option(MCENGINE_ENABLE_CUDA "Enable CUDA backend" ON)

if(MCENGINE_ENABLE_CUDA)
    enable_language(CUDA)

    # Find CUDA toolkit only when needed
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "CUDA toolkit found: ${CUDAToolkit_INCLUDE_DIRS}")
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
endif()

# ----------------------------
# Library target
# ----------------------------
add_library(MCEngine STATIC
        src/omp_backend.cpp
)

target_include_directories(MCEngine
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ----------------------------
# OpenMP
# ----------------------------
find_package(OpenMP REQUIRED)
target_link_libraries(MCEngine PUBLIC OpenMP::OpenMP_CXX)

# ----------------------------
# CUDA backend (optional)
# ----------------------------
if (MCENGINE_ENABLE_CUDA)

    target_sources(MCEngine
            PRIVATE
            src/cuda_backend.cu
    )

    set_target_properties(MCEngine PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
    set_target_properties(MCEngine PROPERTIES
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_compile_definitions(MCEngine PUBLIC MCENGINE_ENABLE_CUDA=1)

    target_link_libraries(MCEngine PUBLIC CUDA::curand)
    target_compile_definitions(MCEngine PUBLIC MCENGINE_USE_CURAND=1)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

    target_compile_options(MCEngine PUBLIC
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
    )
else()
    target_compile_definitions(MCEngine PUBLIC MCENGINE_ENABLE_CUDA=0)
endif()

# ----------------------------
# Warnings (recommended)
# ----------------------------
target_compile_options(MCEngine PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

# ----------------------------
# Testing
# ----------------------------
include(CTest)
enable_testing()

add_subdirectory(tests)
