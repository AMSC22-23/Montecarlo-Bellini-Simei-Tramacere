# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -Ofast -funroll-loops -march=native -fpie -std=c++17 -fopenmp -Wopenmp-simd -ggdb3
NVCCFLAGS = -Xcompiler -fopenmp

# Targets
TARGET = ../buildCUDA/mainCUDA
CUDA_OBJ = mainCUDA.o optionpricer.o hyperrectangle.o financecomputation.o financemontecarlo.o optionparameters.o financeinputmanager.o
# Libraries
LIBS = -lcurand

# Make rules
all: $(TARGET)

$(TARGET): $(CUDA_OBJ)
	$(NVCC) -o $@ $^ $(LIBS) $(NVCCFLAGS)

mainCUDA.o: mainCUDA.cpp
	$(CXX) -c $< $(CXXFLAGS)

optionpricer.o: optionpricer.cu
	$(NVCC) -c $< $(NVCCFLAGS)

hyperrectangle.o: hyperrectangle.cpp
	$(CXX) -c $< $(CXXFLAGS)

financecomputation.o: ../utils/financecomputation.cpp
	$(CXX) -c $< $(CXXFLAGS)

financemontecarlo.o: financemontecarlo.cpp
	$(CXX) -c $< $(CXXFLAGS)

optionparameters.o: optionparameters.cpp
	$(CXX) -c $< $(CXXFLAGS)

financeinputmanager.o: ../utils/financeinputmanager.cpp
	$(CXX) -c $< $(CXXFLAGS)


clean:
	rm -f $(CUDA_OBJ)
