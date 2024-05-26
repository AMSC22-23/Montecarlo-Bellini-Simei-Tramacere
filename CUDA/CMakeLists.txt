# Locate all source files in the src directory
file(GLOB_RECURSE SOURCES "maincuda.cpp")

if(CUDA_FOUND)
    # Locate CUDA source files
    # file(GLOB_RECURSE CUDA_SOURCES "optionpricer.cu")

    # Append CUDA source files to the list of sources
    # list(APPEND SOURCES ${CUDA_SOURCES})

    # Enable CUDA language
    enable_language(CUDA)

    # Create a shared library from the source files
    # add_library(CUDA_LIBRARY SHARED ${SOURCES})

    add_library(CudaOptionPricing STATIC
    ../src/hyperrectangle.cpp
    ../utils/finance_inputmanager.cpp
    ../src/finance_montecarlo.cpp
    ../src/optionparameters.cpp
    ../utils/finance_computation.cpp
    optionpricer.cu
)

    add_executable(mainCUDA
    mainCUDA.cpp
    ${HEADER_FILES}
    )

    # Include directories for the mainCUDA executable
    
    target_include_directories(CudaOptionPricing PRIVATE include)
    target_include_directories(mainCUDA PRIVATE include)

    # Link the CUDA library and necessary libraries to the mainCUDA executable
    # target_link_directories(CUDA_LIBRARY PUBLIC "${CudaOptionPricing}/lib")

    # Link CUDA libraries to the CUDA_LIBRARY
    target_link_libraries(mainCUDA PUBLIC CudaOptionPricing )
endif()