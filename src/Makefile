# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -Ofast -funroll-loops -march=native -fpie -std=c++17 -fopenmp -Wopenmp-simd -ggdb3
NVCCFLAGS = -Xcompiler -fopenmp

# Targets
TARGET = ../buildCUDA/mainCUDA
CUDA_OBJ = mainCUDA.o optionpricer.o asset.o csvhandler.o functionevaluator.o hypercube.o hyperrectangle.o hypersphere.o inputmanager.o

# Libraries
LIBS = -lcurand

# Make rules
all: $(TARGET)

$(TARGET): $(CUDA_OBJ)
	$(NVCC) -o $@ $^ $(LIBS) $(NVCCFLAGS)

mainCUDA.o: mainCUDA.cpp
	$(CXX) -c $< $(CXXFLAGS)

asset.o: asset.cpp
	$(CXX) -c $< $(CXXFLAGS)

csvhandler.o: csvhandler.cpp
	$(CXX) -c $< $(CXXFLAGS)

functionevaluator.o: functionevaluator.cpp
	$(CXX) -c $< $(CXXFLAGS)

hypercube.o: hypercube.cpp
	$(CXX) -c $< $(CXXFLAGS)

hyperrectangle.o: hyperrectangle.cpp
	$(CXX) -c $< $(CXXFLAGS)

hypersphere.o: hypersphere.cpp
	$(CXX) -c $< $(CXXFLAGS)

inputmanager.o: inputmanager.cpp
	$(CXX) -c $< $(CXXFLAGS)

optionpricer.o: optionpricer.cu
	$(NVCC) -c $< $(NVCCFLAGS)

clean:
	rm -f $(TARGET) $(CUDA_OBJ)
