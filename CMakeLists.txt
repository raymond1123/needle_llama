cmake_minimum_required(VERSION 3.12)
project(needle_llama LANGUAGES CXX CUDA)  # set project name

# Set the C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_ARCHITECTURES 86)
SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_BUILD_TYPE Release)  # enable debug symbols for gdb debugging
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(THIRD_PARTY "${PROJECT_SOURCE_DIR}/3rdparty")

# Find CUDA package
find_package(CUDA REQUIRED)

# for cublas
find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include 
                    ${PROJECT_SOURCE_DIR}/encap/include
                    ${CUDA_INCLUDE_DIRS})

message(STATUS "Found cuda, building cuda backend")
message(STATUS "cuda header: ${CUDA_INCLUDE_DIRS}")

# Add your source files
file(GLOB_RECURSE SRC_FILES
    ${PROJECT_SOURCE_DIR}/encap/encap_tensor.cu)

# Set DEBUG to ON by default
option(DEBUG "Enable debug mode" ON)

# Add a definition if DEBUG is enabled
if (DEBUG)
    add_definitions(-DDEBUG)
endif()

# pybind11
add_subdirectory(${THIRD_PARTY}/pybind11)
pybind11_add_module(needle ${SRC_FILES})

# Create an executable
#add_executable(unittest ${TEST_SRC_FILES})

# Link against the CUDA library
target_link_libraries(needle PUBLIC ${CUDA_LIBRARIES} CUDA::cublas CUDA::cudart)

# directly output to ffi folder
set_target_properties(needle
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/unittest/)
    #LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/)

