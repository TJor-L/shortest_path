#include "DijkstraCudaOpt2.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <utility>
#include <functional>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                     \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
              cudaGetErrorString(err));                                      \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)
#endif

// Neighbor-relaxation kernel (same as original)
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

void DijkstraCudaOpt2::run(int source, std::vector<int>& h_dist) {
    // Allocate and initialize distance array on GPU
    int* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, V * sizeof(int)));
    std::vector<int> hostDist(V, INT_MAX);
    hostDist[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_dist, hostDist.data(),
                          V * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate and initialize visited flags on host and device
    std::vector<char> visited(V, 0);
    unsigned char* d_visited = nullptr;
    CUDA_CHECK(cudaMalloc(&d_visited, V * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_visited, 0, V * sizeof(unsigned char)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Main Dijkstra loop using CPU priority queue for min selection
    const int TPB = 65536;
    int blocks = (E + TPB - 1) / TPB;

    for (int iter = 0; iter < V; ++iter) {
        // Copy updated distances back to host
        CUDA_CHECK(cudaMemcpy(hostDist.data(), d_dist,
                              V * sizeof(int), cudaMemcpyDeviceToHost));

        // Build a min-heap of (dist, vertex)
        using PII = std::pair<int,int>;
        std::priority_queue<PII, std::vector<PII>, std::greater<PII>> pq;
        for (int v = 0; v < V; ++v) {
            if (!visited[v]) pq.emplace(hostDist[v], v);
        }

        // Extract the smallest unvisited vertex
        int u = -1;
        int bestDist = INT_MAX;
        while (!pq.empty()) {
            auto [d, idx] = pq.top(); pq.pop();
            if (visited[idx]) continue;
            u = idx;
            bestDist = d;
            break;
        }
        if (u < 0 || bestDist == INT_MAX) break;

        // Mark visited on host and device
        visited[u] = 1;
        unsigned char one = 1;
        CUDA_CHECK(cudaMemcpy(d_visited + u, &one,
                              sizeof(unsigned char),
                              cudaMemcpyHostToDevice));

        // Relax neighbors of u on GPU
        relaxNeighbors<<<blocks, TPB>>>(E, u,
                                        d_src,
                                        d_dst,
                                        d_w,
                                        d_dist);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Record timing and clean up events
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&lastTimeMs, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Copy final distances back to host
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist,
                          V * sizeof(int), cudaMemcpyDeviceToHost));

    // Clean up GPU memory
    cudaFree(d_dist);
    cudaFree(d_visited);
}
