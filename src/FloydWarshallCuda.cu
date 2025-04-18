// src/FloydWarshallCuda.cu
#include "FloydWarshallCuda.h"
#include <climits>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// 1) Initialize dense V×V matrix: INF everywhere, 0 on diagonal
__global__ static void initDense(int V, int* mat) {
  size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  size_t total = size_t(V)*V;
  if (idx >= total) return;
  int i = idx / V, j = idx % V;
  mat[idx] = (i==j ? 0 : INT_MAX);
}

// 2) Scatter edges (d_src, d_dst, d_w provided by base setupGraph)
__global__ static void scatterEdges(int E, int V,
                                    const int* src,
                                    const int* dst,
                                    const int* w,
                                    int* mat) {
  int ei = blockIdx.x*blockDim.x + threadIdx.x;
  if (ei >= E) return;
  int offset = src[ei]*V + dst[ei];
  mat[offset] = w[ei];
}

// 3) Floyd–Warshall inner update for fixed k
__global__ static void fwKernel(int k, int V, int* mat) {
  size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  size_t total = size_t(V)*V;
  if (idx >= total) return;
  int i = idx / V, j = idx % V;

  int dik = mat[i*V + k];
  int dkj = mat[k*V + j];
  int dij = mat[i*V + j];
  if (dik < INT_MAX && dkj < INT_MAX) {
    int via = dik + dkj;
    if (via < dij) mat[i*V + j] = via;
  }
}

void FloydWarshallCuda::run(int source, std::vector<int>& h_dist) {
  // --- 1) Build & initialize dense matrix on GPU ---
  size_t total = size_t(V)*V;
  int* d_mat = nullptr;
  CUDA_CHECK(cudaMalloc(&d_mat, total * sizeof(int)));

  const int threads = 65536;
  int blocksInit = (total + threads - 1) / threads;
  initDense<<<blocksInit, threads>>>(V, d_mat);
  CUDA_CHECK(cudaDeviceSynchronize());

  int blocksScatter = (E + threads - 1) / threads;
  scatterEdges<<<blocksScatter, threads>>>(E, V, d_src, d_dst, d_w, d_mat);
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- 2) Time the FW loop ---
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  int blocksFW = blocksInit;
  for (int k = 0; k < V; ++k) {
    fwKernel<<<blocksFW, threads>>>(k, V, d_mat);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&lastTimeMs, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // --- 3) Copy back & extract the 'source' row ---
  std::vector<int> hostMat(total);
  CUDA_CHECK(cudaMemcpy(hostMat.data(),
                        d_mat,
                        total * sizeof(int),
                        cudaMemcpyDeviceToHost));

  h_dist.resize(V);
  for (int v = 0; v < V; ++v) {
    h_dist[v] = hostMat[source * V + v];
  }

  // --- 4) Clean up ---
  CUDA_CHECK(cudaFree(d_mat));
}
