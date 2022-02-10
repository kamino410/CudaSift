cmake_minimum_required(VERSION 2.6)

project(cudaSift)
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

option(CUDASIFT_BUILD_SAMPLE "" OFF)

find_package(Eigen3 REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE")
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT")
  endif()
endif()

set(cuda_sources
  cudaImage.cu
  cudaSiftH.cu
  matching.cu
)
cuda_add_library(cudasift ${cuda_sources} OPTIONS -arch=sm_35)
set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
)
target_include_directories(cudasift PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cudasift PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY})

if(${BUILD_SAMPLE})
  find_package(OpenCV REQUIRED)

  set(sources
    geomFuncs.cpp
    mainSift.cpp
  )
  add_executable(sample ${sources})
  target_link_libraries(sample cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
endif()

install(FILES
  ${cuda_sources}
  ${sources}
  cudaSiftD.cu
  CMakeLists.txt
  Copyright.txt
  DESTINATION .
)
install(FILES data/left.pgm data/righ.pgm
  DESTINATION data
)
