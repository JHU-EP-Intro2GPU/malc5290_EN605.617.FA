cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(EN605_617_PROJECT LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
# include(CTest)
set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -std=c++11 " )
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" "-std=c++11")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" "-x cuda")
include_directories(
    SYSTEM
    /usr/local/cuda/include
    ./cpp/src
)

add_subdirectory(./src)
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
# target_compile_features(particles PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
#set_target_properties( particles
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#                      )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
