#include "DijkstraCuda.h"
#include <climits>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

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

void DijkstraCuda::run(int source, std::vector<int>& h_dist) {

  int* d_dist = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dist, V * sizeof(int)));
  std::vector<int> hostDist(V, INT_MAX);
  hostDist[source] = 0;
  CUDA_CHECK(cudaMemcpy(d_dist, hostDist.data(),
                        V * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<char> visited(V, 0);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  const int threads = 65536;
  const int blocks  = (E + threads - 1) / threads;

  for (int iter = 0; iter < V; ++iter) {
    CUDA_CHECK(cudaMemcpy(hostDist.data(), d_dist,
                          V * sizeof(int), cudaMemcpyDeviceToHost));

    int u = -1, best = INT_MAX;
    for (int v = 0; v < V; ++v) {
      if (!visited[v] && hostDist[v] < best) {
        best = hostDist[v];
        u = v;
      }
    }
    if (u < 0 || best == INT_MAX) break;
    visited[u] = 1;

    relaxNeighbors<<<blocks, threads>>>(E, u, d_src, d_dst, d_w, d_dist);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&lastTimeMs, start, stop));

  CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist,
                        V * sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(d_dist);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
