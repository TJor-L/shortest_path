# CUDA Shortest Path Test Harness

This repository provides a simple framework for benchmarking single‐source shortest‑path (SSSP) algorithms on both CPU and GPU. It comes with:

- **Graph generator** (random directed graphs with configurable vertex count, density, and weight range)  
- **Abstract base class** `ShortestPathAlgorithm`  
- **Implementations**:
  - Naïve Dijkstra on GPU (`DijkstraCuda`)  
  - Standard Dijkstra on CPU (`DijkstraCpu`)  
- **Automated testing** over multiple randomly generated graphs  
- **Timestamped CSV** output in `results/`  
- **Makefile** for one‑step build

---

## Project Structure

```bash
project/
├── config.ini             # test parameters
├── Makefile               # build rules
├── .gitignore
├── include/               # public headers
│   ├── GraphGenerator.h
│   ├── ShortestPathAlgorithm.h
│   ├── DijkstraCuda.h
│   └── DijkstraCpu.h
└── src/                   # implementation files
    ├── main.cu
    └── DijkstraCuda.cu
```

---

## Prerequisites

- CUDA Toolkit installed and in your `PATH`  
- A C++ compiler supporting C++17 (e.g. `g++` or `clang++`)  
- GNU Make  

---

## Configuration

Edit **config.ini** to set your test parameters:

```ini
# Number of vertices (V)
V=1000

# Edge density (0.0–1.0)
density=0.01

# Maximum weight for each edge (1..maxWeight)
maxWeight=10

# Number of different random graphs to test
runs=5

# Source vertex index for SSSP
source=0
```

---

## Build

From the project root:

```bash
make
```

This compiles all `.cu` files into the `test_sp` executable.

---

## Run

```bash
./test_sp
```

- Creates a `results/` directory if missing  
- Runs each algorithm (`BellmanFordCuda`, `DijkstraCuda`, `DijkstraCpu`) on `runs` different random graphs  
- Writes a timestamped CSV:  

  ```
  results/results_YYYYMMDD_HHMMSS.csv
  ```

- CSV columns:  

  ```
  Algorithm,Run,V,E,Density,TimeMs
  ```

---

## Adding a New Algorithm

1. **Create a header** `include/MyAlgoCuda.h`:

   ```cpp
   #pragma once
   #include "ShortestPathAlgorithm.h"

   class MyAlgoCuda : public ShortestPathAlgorithm {
   public:
     void run(int source, std::vector<int>& dist) override {
       // 1) Allocate & initialize device data if needed
       // 2) Launch CUDA kernels, measure time in lastTimeMs
       // 3) Copy results back to dist
     }
   };
   ```

2. **Implement it** in `src/MyAlgoCuda.cu`:

   ```cpp
   #include "MyAlgoCuda.h"

   // [define any kernels here]

   void MyAlgoCuda::run(int source, std::vector<int>& dist) {
     // your implementation…
   }
   ```

3. **Register it** in `src/main.cu`:

   ```cpp
   #include "MyAlgoCuda.h"
   // …
   algos.emplace_back("MyAlgoCuda", std::make_unique<MyAlgoCuda>());
   ```

4. **Add to Makefile**:

   ```Makefile
    # Add to the list of source files
    SRC      := src/main.cu src/DijkstraCuda.cu [src/MyAlgoCuda.cu]
    ```

5. **Rebuild & rerun**:

   ```bash
   make
   ./test_sp
   ```
