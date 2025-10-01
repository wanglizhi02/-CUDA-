cmake_minimum_required(VERSION 3.18)
project(myab LANGUAGES CXX CUDA)

INCLUDE_DIRECTORIES(./include)
# add_compile_options(-fmad=false)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})



# find_library(SHTNS libshtns.a ./lib)
# find_library(SHTNS_CUDA libshtns_cuda.a ./lib)


# Add all source files to the library
# file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB SOURCES1 "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")



# Generate the library from the sources
# add_library(MYLIB STATIC ${SOURCES})
add_library(MYCUDALIB STATIC ${SOURCES1})

ADD_EXECUTABLE(ab ./t.cu)

TARGET_LINK_LIBRARIES(ab PRIVATE  fftw3 cublas cufft MYCUDALIB ${CUDA_LIBRARIES} )

# TARGET_LINK_LIBRARIES(ab PRIVATE MYLIB MYCUDALIB fftw3 cublas cufft ${SHTNS} ${SHTNS_CUDA} ${CUDA_LIBRARIES} )
# TARGET_LINK_LIBRARIES(ab PRIVATE  ${SHTNS} ${SHTNS_CUDA} ${CUDA_LIBRARIES})
# TARGET_LINK_LIBRARIES(ab PRIVATE fftw3 cublas cufft)