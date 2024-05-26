cmake_minimum_required(VERSION 3.10)
project(CUDAProject)

# Compiler
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_COMPILER nvcc)

# Compiler flags
set(CMAKE_CXX_FLAGS "-Ofast -funroll-loops -march=native -fpie -std=c++17 -fopenmp -Wopenmp-simd -ggdb3")
set(CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp")

# Targets
set(TARGET mainCUDA)
set(CUDA_OBJ mainCUDA.cpp optionpricer.cu hyperrectangle.cpp ../utils/financecomputation.cpp financemontecarlo.cpp optionparameters.cpp ../utils/financeinputmanager.cpp)

# Libraries
set(LIBS -lcurand)

# Add executable
add_executable(${TARGET} ${CUDA_OBJ})

# Link libraries
target_link_libraries(${TARGET} ${LIBS})

# Compile CUDA files
set_source_files_properties(optionpricer.o PROPERTIES LANGUAGE CUDA)