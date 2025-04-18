// DijkstraCudaOpt.cu
#include "DijkstraCudaOpt.h"
#include <climits>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

// (identical) neighbor‐relaxation kernel
__global__ static void relaxNeighbors(int E, int u,
                                      const int* src,
                                      const int* dst,
                                      const int* w,
                                      int* dist) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= E) return;
  if (src[idx] == u) {
    int du = dist[u];
    if (du < INT_MAX) {
      int v  = dst[idx];
      int nd = du + w[idx];
      atomicMin(&dist[v], nd);
    }
  }
}

// block‐level reduction: each block finds its local best (dist, idx)
__global__ static void findBlockMin(int V,
                                    const int* dist,
                                    const char* visited,
                                    int* blockBest,
                                    int* blockIdxOut) {
  extern __shared__ int sdata[];           // size = 2 * blockDim.x
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  int bestDist = INT_MAX;
  int bestIdx  = -1;
  if (idx < V && !visited[idx]) {
    bestDist = dist[idx];
    bestIdx  = idx;
  }

  int* sdist = sdata;                     // first half
  int* sidx  = sdata + blockDim.x;        // second half
  sdist[tid] = bestDist;
  sidx[tid]  = bestIdx;
  __syncthreads();

  for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (sdist[tid + stride] < sdist[tid]) {
        sdist[tid] = sdist[tid + stride];
        sidx[tid]  = sidx[tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    blockBest[blockIdx.x]   = sdist[0];
    blockIdxOut[blockIdx.x] = sidx[0];
  }
}

void DijkstraCudaOpt::run(int source, std::vector<int>& h_dist) {
  int*   d_dist      = nullptr;
  char*  d_visited   = nullptr;
  int*   d_blockBest = nullptr;
  int*   d_blockIdx  = nullptr;

  // allocate device arrays
  CUDA_CHECK(cudaMalloc(&d_dist,      V * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_visited,   V * sizeof(char)));

  const int threads = 65536;
  const int blocksE = (E + threads - 1) / threads;
  const int blocksV = (V + threads - 1) / threads;
  CUDA_CHECK(cudaMalloc(&d_blockBest, blocksV * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_blockIdx,  blocksV * sizeof(int)));

  // host‐side init
  std::vector<int>  hostDist(V, INT_MAX);
  std::vector<char> hostVis(V, 0);
  hostDist[source] = 0;

  // copy initial state to device
  CUDA_CHECK(cudaMemcpy(d_dist,    hostDist.data(), V * sizeof(int),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_visited, hostVis.data(),   V * sizeof(char),  cudaMemcpyHostToDevice));

  // timers
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  std::vector<int>  h_blockBest(blocksV);
  std::vector<int>  h_blockIdx(blocksV);

  for (int iter = 0; iter < V; ++iter) {
    // 1) find next u via block reduction
    findBlockMin<<<blocksV, threads, 2*threads*sizeof(int)>>>(V, d_dist, d_visited, d_blockBest, d_blockIdx);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2) copy block minima back
    CUDA_CHECK(cudaMemcpy(h_blockBest.data(), d_blockBest, blocksV * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blockIdx.data(),  d_blockIdx,  blocksV * sizeof(int), cudaMemcpyDeviceToHost));

    // 3) host‐side final reduction
    int u = -1, best = INT_MAX;
    for (int b = 0; b < blocksV; ++b) {
      if (h_blockBest[b] < best) {
        best = h_blockBest[b];
        u    = h_blockIdx[b];
      }
    }
    if (u < 0 || best == INT_MAX) break;

    // 4) mark visited
    hostVis[u] = 1;
    char one = 1;
    CUDA_CHECK(cudaMemcpy(d_visited + u, &one, sizeof(char), cudaMemcpyHostToDevice));

    // 5) relax neighbors of u
    relaxNeighbors<<<blocksE, threads>>>(E, u, d_src, d_dst, d_w, d_dist);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // stop timer
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&lastTimeMs, start, stop));

  // copy final distances back
  CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, V * sizeof(int), cudaMemcpyDeviceToHost));

  // cleanup
  cudaFree(d_dist);
  cudaFree(d_visited);
  cudaFree(d_blockBest);
  cudaFree(d_blockIdx);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
