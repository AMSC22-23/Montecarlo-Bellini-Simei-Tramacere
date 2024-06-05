# Locate all source files in the src directory
file(GLOB_RECURSE SOURCES "maincuda.cpp")

if(CUDA_FOUND)
    
    # Enable CUDA language
    enable_language(CUDA)

    # Add the CUDA library
    add_library(CudaOptionPricing STATIC
    ../src/integration/geometry/hyperrectangle.cpp
    ../src/optionpricing/finance_inputmanager.cpp
    ../src/optionpricing/finance_montecarlo.cpp
    ../src/optionpricing/optionparameters.cpp
    ../src/optionpricing/optionpricer.cpp
    optionpricer.cu
)
    add_executable(mainCUDA
    mainCUDA.cpp
    ${HEADER_FILES}
    )

    # Include directories for the mainCUDA executable
    
    target_include_directories(CudaOptionPricing PRIVATE include)
    target_include_directories(mainCUDA PRIVATE include)

    # Link CUDA libraries to the CUDA_LIBRARY
    #target_link_libraries(mainCUDA PUBLIC CudaOptionPricing )

    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(OPENMP_FLAGS "-fopenmp -Wopenmp-simd")
        target_link_libraries(CudaOptionPricing OpenMP::OpenMP_CXX ${OPENMP_FLAGS} )
        target_link_libraries(mainCUDA PUBLIC CudaOptionPricing OpenMP::OpenMP_CXX ${OPENMP_FLAGS} )
    endif()
endif()