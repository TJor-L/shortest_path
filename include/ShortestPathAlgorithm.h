#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include "GraphGenerator.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__              \
                << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n";\
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)


class ShortestPathAlgorithm {
protected:
  int V = 0, E = 0;
  int *d_src = nullptr, *d_dst = nullptr, *d_w = nullptr;
  float lastTimeMs = 0.0f;

public:
  virtual ~ShortestPathAlgorithm() {
    if (d_src) cudaFree(d_src);
    if (d_dst) cudaFree(d_dst);
    if (d_w)   cudaFree(d_w);
  }

  virtual void setupGraph(int V_, int E_, const std::vector<Edge>& edges) {
    V = V_; E = E_;
    if (d_src) cudaFree(d_src);
    if (d_dst) cudaFree(d_dst);
    if (d_w)   cudaFree(d_w);

    std::vector<int> h_src(E), h_dst(E), h_w(E);
    for (int i = 0; i < E; ++i) {
      h_src[i] = edges[i].u;
      h_dst[i] = edges[i].v;
      h_w[i]   = edges[i].w;
    }
    CUDA_CHECK(cudaMalloc(&d_src, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dst, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_w,   E * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), E * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, h_dst.data(), E * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,   h_w.data(),   E * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  virtual void run(int source, std::vector<int>& dist) = 0;

  float getLastTimeMs() const { return lastTimeMs; }
};
