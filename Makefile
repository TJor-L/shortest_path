NVCC     := nvcc
INCLUDES := -Iinclude
CXXFLAGS := -O2 -std=c++17 $(INCLUDES)

SRC      := src/main.cu src/DijkstraCuda.cu
TARGET   := test_sp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TARGET) results_*.csv
