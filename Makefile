NVCC     := nvcc
INCLUDES := -Iinclude
CXXFLAGS := -O2 -std=c++17 $(INCLUDES)

SRC      := src/main.cu src/BellmanFordCudaOpt.cu  src/DijkstraCuda.cu src/BellmanFordCuda.cu src/FloydWarshallCuda.cu src/FloydWarshallCudaOpt.cu src/DijkstraCudaOpt.cu src/DijkstraCudaOpt2.cu src/DijkstraCudaOpt3.cu
TARGET   := test_sp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TARGET) results_*.csv
