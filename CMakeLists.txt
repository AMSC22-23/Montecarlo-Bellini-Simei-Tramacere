cmake_minimum_required(VERSION 3.10)

project(OptionPricing VERSION 1.0)
  
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

#sCheck if CUDA is available and add CUDA source files
find_package(CUDA)
if(CUDA_FOUND)
        message(STATUS "CUDA found")

        # Enable CUDA language
        enable_language(CUDA) #enable NVCC compiler

        # Set policy to use CUDA_ARCHITECTURES
        cmake_policy(SET CMP0104 NEW)

        # Set CUDA_ARCHITECTURES to compute capability 6.1
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75 80 86)
endif()

add_subdirectory(CUDA)


include_directories(/include/muparser-2.3.4/include)

# Library files
add_library(OptionPricing STATIC
    src/main.cpp
    src/inputmanager.cpp
    src/integration/geometry/hypercube.cpp
    src/integration/geometry/hypersphere.cpp
    src/integration/geometry/hyperrectangle.cpp
    src/integration/functionevaluator.cpp
    src/integration/integralcalculator.cpp
    src/optionpricing/finance_inputmanager.cpp
    src/optionpricing/finance_montecarlo.cpp
    src/optionpricing/optionparameters.cpp
    src/optionpricing/optionpricer.cpp
    src/optionpricing/finance_montecarloutils.cpp
    src/optionpricing/finance_pricingutils.cpp
    )

add_executable(mainOmp
src/main.cpp
${HEADER_FILES}
)

# Include directory
target_include_directories(OptionPricing PRIVATE include)
target_include_directories(mainOmp PRIVATE include)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(OPENMP_FLAGS "-fopenmp -Wopenmp-simd")
    target_link_libraries(OptionPricing OpenMP::OpenMP_CXX ${OPENMP_FLAGS} )
    target_link_libraries(mainOmp OptionPricing OpenMP::OpenMP_CXX ${OPENMP_FLAGS} )
endif()