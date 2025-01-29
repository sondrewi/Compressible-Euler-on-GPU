# Paths and compiler settings
NVCC=nvcc
EIGEN_INCLUDE_PATH=/lsc/opt/eigen-3.4.0
NVCCFLAGS=-O0 -arch=sm_80
INCLUDES=-isystem $(EIGEN_INCLUDE_PATH)
SUPPRESS_FLAGS=-Xcudafe --diag_suppress=20012

# Output executable
OUTPUT=MUSCL_exec

# Source and Object files
CPP_SOURCES=Grid.cpp Slope_limiter.cpp Riemann.cpp Solver.cpp
CU_SOURCES=DeviceKernels.cu CUDA_Grid.cu CUDAtests.cu
OBJECTS=$(CPP_SOURCES:.cpp=.o) $(CU_SOURCES:.cu=.o)

# Default target
all: $(OUTPUT)

# Compile CPP sources
Grid.o: Grid.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c Grid.cpp -o Grid.o

Slope_limiter.o: Slope_limiter.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c Slope_limiter.cpp -o Slope_limiter.o

Riemann.o: Riemann.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c Riemann.cpp -o Riemann.o

Solver.o: Solver.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c Solver.cpp -o Solver.o

# Compile CU sources
DeviceKernels.o: DeviceKernels.cu
	$(NVCC) $(NVCCFLAGS) -c DeviceKernels.cu -o DeviceKernels.o

CUDA_Grid.o: CUDA_Grid.cu
	$(NVCC) $(NVCCFLAGS) -c CUDA_Grid.cu -o CUDA_Grid.o

CUDAtests.o: CUDAtests.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c CUDAtests.cu -o CUDAtests.o $(SUPPRESS_FLAGS)

# Link objects into the final executable
$(OUTPUT): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o $(OUTPUT)

# Clean up compiled files
clean:
	rm -f $(OBJECTS) $(OUTPUT)