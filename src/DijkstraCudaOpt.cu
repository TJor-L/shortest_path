#include "DijkstraCudaOpt.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

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

// Stage 1: per-block reduction to find local minima
__global__ void findBlockMins(const int* __restrict__ dist,
                              const unsigned char* __restrict__ visited,
                              int* blockMinDist,
                              int* blockMinIdx,
                              int N) {
    extern __shared__ int s[];
    int* sdist = s;
    int* sidx  = s + blockDim.x;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int ld  = INT_MAX, li = -1;
    if (gid < N && !visited[gid]) {
        ld = dist[gid];
        li = gid;
    }
    sdist[threadIdx.x] = ld;
    sidx[threadIdx.x]  = li;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            int rd = sdist[threadIdx.x + offset];
            if (rd < sdist[threadIdx.x]) {
                sdist[threadIdx.x] = rd;
                sidx[threadIdx.x]  = sidx[threadIdx.x + offset];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        blockMinDist[blockIdx.x] = sdist[0];
        blockMinIdx[blockIdx.x]  = sidx[0];
    }
}

// Stage 2: global reduction over block minima
__global__ void findGlobalMin(const int* blockMinDist,
                              const int* blockMinIdx,
                              int* outDist,
                              int* outIdx,
                              int M) {
    extern __shared__ int s[];
    int* sdist = s;
    int* sidx  = s + blockDim.x;

    int tid = threadIdx.x;
    int ld  = (tid < M ? blockMinDist[tid] : INT_MAX);
    int li  = (tid < M ? blockMinIdx[tid]  : -1);

    sdist[tid] = ld;
    sidx[tid]  = li;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            int rd = sdist[tid + offset];
            if (rd < sdist[tid]) {
                sdist[tid] = rd;
                sidx[tid]  = sidx[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        outDist[0] = sdist[0];
        outIdx[0]  = sidx[0];
    }
}

// Original neighbor-relaxation kernel
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

void DijkstraCudaOpt::run(int source, std::vector<int>& h_dist) {
    // Allocate and initialize distance array
    int* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, V * sizeof(int)));
    std::vector<int> hostDist(V, INT_MAX);
    hostDist[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_dist, hostDist.data(),
                          V * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate and initialize visited flags on device
    std::vector<unsigned char> visited(V, 0);
    unsigned char* d_visited = nullptr;
    CUDA_CHECK(cudaMalloc(&d_visited, V * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_visited, visited.data(),
                          V * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Prepare per-block minima buffers
    const int TPB = 256;
    int blocks1 = (V + TPB - 1) / TPB;
    int* d_blockMinDist = nullptr;
    int* d_blockMinIdx  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockMinDist, blocks1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockMinIdx,  blocks1 * sizeof(int)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Main Dijkstra loop
    for (int iter = 0; iter < V; ++iter) {
        // Stage 1: find local minima per block
        size_t shared1 = 2 * TPB * sizeof(int);
        findBlockMins<<<blocks1, TPB, shared1>>>(d_dist,
                                                 d_visited,
                                                 d_blockMinDist,
                                                 d_blockMinIdx,
                                                 V);
        CUDA_CHECK(cudaGetLastError());

        // Stage 2: reduce block minima to global minimum
        size_t shared2 = 2 * blocks1 * sizeof(int);
        findGlobalMin<<<1, blocks1, shared2>>>(d_blockMinDist,
                                               d_blockMinIdx,
                                               d_blockMinDist,
                                               d_blockMinIdx,
                                               blocks1);
        CUDA_CHECK(cudaGetLastError());

        // Copy back the global minimum distance and index
        int bestDist, u;
        CUDA_CHECK(cudaMemcpy(&bestDist, d_blockMinDist, sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&u,        d_blockMinIdx,  sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Terminate if no reachable vertex remains
        if (u < 0 || bestDist == INT_MAX) break;

        // Mark visited[u] = true on device
        unsigned char one = 1;
        CUDA_CHECK(cudaMemcpy(d_visited + u, &one,
                              sizeof(unsigned char),
                              cudaMemcpyHostToDevice));

        // Relax neighbors of u
        relaxNeighbors<<<(E + TPB - 1)/TPB, TPB>>>(E, u,
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

    // Clean up
    cudaFree(d_dist);
    cudaFree(d_visited);
    cudaFree(d_blockMinDist);
    cudaFree(d_blockMinIdx);
}
