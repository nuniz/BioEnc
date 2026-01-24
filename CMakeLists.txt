cmake_minimum_required(VERSION 3.18)
project(bioenc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find pybind11 (minimum version 2.11 for modern C++ support)
find_package(pybind11 2.11 CONFIG REQUIRED)

# Find OpenMP (optional)
find_package(OpenMP)

# Create the Python module
pybind11_add_module(_bioenc src/bindings.cpp)

target_include_directories(_bioenc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(_bioenc PRIVATE OpenMP::OpenMP_CXX)
endif()

# Optimization flags for Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(_bioenc PRIVATE -O3 -march=native)
endif()

# Install target
install(TARGETS _bioenc LIBRARY DESTINATION bioenc)
