cmake_minimum_required(VERSION 4.1)

project(MCEngine
        VERSION 0.1
        LANGUAGES CXX CUDA
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
    enable_language(CUDA)

    target_sources(MCEngine
            PRIVATE
            src/cuda_backend.cu
    )

    set_target_properties(MCEngine PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_definitions(MCEngine PUBLIC MCENGINE_HAS_CUDA=1)
else()
    target_compile_definitions(MCEngine PUBLIC MCENGINE_HAS_CUDA=0)
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