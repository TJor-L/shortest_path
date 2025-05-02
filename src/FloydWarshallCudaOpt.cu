// FloydWarshallCudaOptimized.cu
// Optimized blocked Floyd–Warshall using shared memory and 2D thread blocks

#include "FloydWarshallCudaOpt.h"
#include <climits>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Tile size (shared-memory block dimension)
constexpr int TILE = 32;

// 1) Combined init + scatter in one kernel, 2D launch
__global__ void initScatter(int V, int E,
                            const int* src,
                            const int* dst,
                            const int* w,
                            int* mat) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < V && j < V) {
    // initialize
    int idx = i * V + j;
    mat[idx] = (i == j ? 0 : INT_MAX);
  }
  // scatter edges
  int ei = blockIdx.y * blockDim.y + threadIdx.x;  // reuse x dimension
  if (ei < E) {
    int o = src[ei] * V + dst[ei];
    mat[o] = w[ei];
  }
}

// 2) Blocked FW kernel: each block processes a TILE×TILE submatrix
__global__ void fwBlocked(int V, int* mat) {
  __shared__ int tile_i[TILE][TILE];
  __shared__ int tile_j[TILE][TILE];

  int bi = blockIdx.y;
  int bj = blockIdx.x;
  int ti = threadIdx.y;
  int tj = threadIdx.x;

  int row = bi * TILE + ti;
  int col = bj * TILE + tj;

  // 1) read once into register (or set to INF if out-of-bounds)
  int d = (row < V && col < V)
          ? mat[row * V + col]
          : INT_MAX;

  // 2) loop over k-blocks, only update 'd'
  for (int bk = 0; bk < V; bk += TILE) {
    // load row-tile and col-tile into shared memory
    if (row < V && (bk + tj) < V)
      tile_i[ti][tj] = mat[row * V + (bk + tj)];
    else
      tile_i[ti][tj] = INT_MAX;

    if ((bk + ti) < V && col < V)
      tile_j[ti][tj] = mat[(bk + ti) * V + col];
    else
      tile_j[ti][tj] = INT_MAX;

    __syncthreads();

    if (row < V && col < V) {
      #pragma unroll
      for (int k = 0; k < TILE && (bk + k) < V; ++k) {
        int via = tile_i[ti][k] + tile_j[k][tj];
        if (via < d) d = via;
      }
    }

    __syncthreads();
  }

  // 3) write back exactly once
  if (row < V && col < V) {
    mat[row * V + col] = d;
  }
}

void FloydWarshallCudaOpt::run(int source, std::vector<int>& h_dist) {
    size_t total = size_t(V) * V;
    int* d_mat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mat, total * sizeof(int)));

    // launch init+scatter
    dim3 threads(TILE, TILE);
    dim3 blocks((V + TILE - 1) / TILE, (V + TILE - 1) / TILE);
    initScatter<<<blocks, threads>>>(V, E, d_src, d_dst, d_w, d_mat);
    CUDA_CHECK(cudaDeviceSynchronize());

    // time the blocked FW
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // single launch covers all k-blocks internally
    fwBlocked<<<blocks, threads>>>(V, d_mat);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&lastTimeMs, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy back and extract source row
    std::vector<int> hostMat(total);
    CUDA_CHECK(cudaMemcpy(hostMat.data(), d_mat,
                            total * sizeof(int),
                            cudaMemcpyDeviceToHost));

    h_dist.resize(V);
    for (int v = 0; v < V; ++v)
        h_dist[v] = hostMat[source * V + v];

    CUDA_CHECK(cudaFree(d_mat));
}
