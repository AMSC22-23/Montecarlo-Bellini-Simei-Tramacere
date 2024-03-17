cmake_minimum_required(VERSION 3.10)

project(OptionPricing VERSION 1.0)

set(CMAKE_C_COMPILER "/usr/local/Cellar/llvm/17.0.6_1/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/17.0.6_1/bin/clang++")

option(USE_OpenMP "Use OpenMP" ON)

set(OpenMP_C_FLAGS "-fopenmp -I/usr/local/include")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/llvm/17.0.6_1/lib")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/usr/local/Cellar/llvm/17.0.6_1/lib/libomp.dylib")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(DEBUG_MODE "Enable Debug Mode" OFF)

if(DEBUG_MODE)
    set(CMAKE_BUILD_TYPE "Debug")
else()
    set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
# add_definitions(-DNO_AFFINITY=1)
# add_compile_options(-DNO_AFFINITY=1)
SET(CMAKE_CXX_FLAGS"-Ofast -Wall -mavx2 -flto -lpthread -march=znver2 -mtune=znver2 -qopt-zmm-usage=high -qopt-prefetch -qopt-dynamic-align")

IF (MY_COMPILER_OPTION STREQUAL "Intel")
  MESSAGE(STATUS "** Compiling with Intel settings **")
  SET(CMAKE_CXX_COMPILER "icpx")
  SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -Ofast -xHost -ipo -march=znver2 -mtune=znver2 -axarch -no-prec-div -fp-model fast=2 -qopt-zmm-usage=high -qopt-prefetch -qopt-dynamic-align")
  SET(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
  MESSAGE(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
ENDIF ()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-g -O0 -Wall -DDEBUG)
else()
    add_compile_options(-O3 -Wall -march=native)
endif()

include_directories(/include/muparser-2.3.4/include)

# Library files
add_library(OptionPricing STATIC
    src/asset.cpp
    src/functionevaluator.cpp
    src/hypercube.cpp
    src/hyperrectangle.cpp
    src/hypersphere.cpp
    src/inputmanager.cpp
)

add_executable(mainOmp
src/main.cpp
${HEADER_FILES}
)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    target_link_libraries(OptionPricing OpenMP::OpenMP_CXX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/Cellar/llvm/17.0.6_1/lib -lomp")
endif()

target_link_libraries(mainOmp OptionPricing)

# Include directory
target_include_directories(OptionPricing PRIVATE include)
target_include_directories(mainOmp PRIVATE include)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(OPENMP_FLAGS "-fopenmp -Wopenmp-simd")
    target_link_libraries(OptionPricing OpenMP::OpenMP_CXX ${OPENMP_FLAGS} )
    target_link_libraries(mainOmp OptionPricing OpenMP::OpenMP_CXX ${OPENMP_FLAGS} )
endif()



