#include "BellmanFordCudaOpt.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

#define INF 1000000000

// -----------------------------------------------------------------------------
// Kernel: one warp per vertex in the **current frontier**
// -----------------------------------------------------------------------------
__global__ void bf_frontier_warp(
    int                V,
    const int         *rowPtrIn,
    const int         *colIn,
    const int         *wIn,
    const int         *dist_cur,
          int         *dist_next,
    const int         *frontier_curr,
    int                frontier_size,
          int         *frontier_next,
          int         *frontier_next_size
) {
    int globalLane = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId     = globalLane >> 5;      // which warp
    int lane       = globalLane &  31;     // which lane in that warp
    if (warpId >= frontier_size) return;

    int v = frontier_curr[warpId];

    int start = rowPtrIn[v], end = rowPtrIn[v+1];
    int best = dist_cur[v];

    for (int e = start + lane; e < end; e += 32) {
        int u = colIn[e], w = wIn[e];
        int du = dist_cur[u];
        if (du != INF) {
            int cand = du + w;
            if (cand < best) best = cand;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(0xFFFFFFFF, best, offset);
        best = min(best, other);
    }

    if (lane == 0) {
        int old = dist_cur[v];
        dist_next[v] = best;
        if (best < old) {
            int idx = atomicAdd(frontier_next_size, 1);
            frontier_next[idx] = v;
        }
    }
}

// -----------------------------------------------------------------------------
// Host: run() builds inbound‐CSR from the original arrays, then does
// the frontier‐driven warp kernel.
// -----------------------------------------------------------------------------
void BellmanFordCudaOpt::run(int source, std::vector<int>& dist) {
    if (V <= 0 || E <= 0) {
        std::cerr << "Error: Graph uninitialized!" << std::endl;
        return;
    }

    // --- 0) Copy original edge arrays back to host ---
    std::vector<int> h_src(E), h_dst(E), h_w(E);
    CUDA_CHECK(cudaMemcpy(h_src.data(), d_src, E * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, E * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w.data(),   d_w,   E * sizeof(int), cudaMemcpyDeviceToHost));

    // --- 1) Build inbound CSR on host ---
    std::vector<int> rowPtrIn(V+1, 0), colIn(E), wIn(E);

    // 1a) count incoming edges per vertex
    for (int i = 0; i < E; ++i) {
        ++rowPtrIn[ h_dst[i] + 1 ];
    }
    // 1b) prefix-sum
    for (int v = 1; v <= V; ++v) {
        rowPtrIn[v] += rowPtrIn[v-1];
    }
    // 1c) scatter into colIn/wIn
    std::vector<int> writePtr = rowPtrIn;
    for (int i = 0; i < E; ++i) {
        int v   = h_dst[i];
        int pos = writePtr[v]++;
        colIn[pos] = h_src[i];
        wIn[pos]   = h_w[i];
    }

    // --- 2) Upload CSR to the device ---
    int *d_rowPtrIn = nullptr, *d_colIn = nullptr, *d_wIn = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowPtrIn, (V+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIn,     E    * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wIn,       E    * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rowPtrIn, rowPtrIn.data(), (V+1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIn,    colIn.data(),    E    * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wIn,      wIn.data(),      E    * sizeof(int), cudaMemcpyHostToDevice));

    // --- 3) Allocate + init distance buffers (double‐buffer) ---
    int *d_dist_cur = nullptr, *d_dist_next = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist_cur,  V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist_next, V * sizeof(int)));
    std::vector<int> hostDist(V, INF);
    hostDist[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_dist_cur, hostDist.data(), V * sizeof(int), cudaMemcpyHostToDevice));

    // --- 4) Allocate + init frontier buffers (start all active) ---
    std::vector<int> h_frontier(V);
    std::iota(h_frontier.begin(), h_frontier.end(), 0);
    int *d_frontier_curr = nullptr, *d_frontier_next = nullptr, *d_frontier_next_size = nullptr;
    CUDA_CHECK(cudaMalloc(&d_frontier_curr,      V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_next,      V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_next_size,    sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_frontier_curr, h_frontier.data(), V * sizeof(int), cudaMemcpyHostToDevice));
    int frontier_size = V;

    // --- 5) Main loop: warp-per-vertex over frontier, early exit ---
    cudaEvent_t tic, toc;
    CUDA_CHECK(cudaEventCreate(&tic));
    CUDA_CHECK(cudaEventCreate(&toc));
    CUDA_CHECK(cudaEventRecord(tic));

    const int WARPS_PER_BLOCK   = 4;
    const int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

    for (int iter = 0; iter < V - 1 && frontier_size > 0; ++iter) {
        // reset next-frontier counter
        CUDA_CHECK(cudaMemset(d_frontier_next_size, 0, sizeof(int)));

        int blocks = (frontier_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        bf_frontier_warp<<<blocks, THREADS_PER_BLOCK>>>(
            V,
            d_rowPtrIn, d_colIn, d_wIn,
            d_dist_cur, d_dist_next,
            d_frontier_curr, frontier_size,
            d_frontier_next, d_frontier_next_size
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // fetch new frontier size
        CUDA_CHECK(cudaMemcpy(&frontier_size, d_frontier_next_size, sizeof(int), cudaMemcpyDeviceToHost));

        // swap buffers
        std::swap(d_dist_cur,       d_dist_next);
        std::swap(d_frontier_curr,  d_frontier_next);
    }

    CUDA_CHECK(cudaEventRecord(toc));
    CUDA_CHECK(cudaEventSynchronize(toc));
    CUDA_CHECK(cudaEventElapsedTime(&lastTimeMs, tic, toc));

    // --- 6) Copy back final distances & clean up ---
    CUDA_CHECK(cudaMemcpy(hostDist.data(), d_dist_cur, V * sizeof(int), cudaMemcpyDeviceToHost));
    dist = hostDist;

    CUDA_CHECK(cudaFree(d_rowPtrIn));
    CUDA_CHECK(cudaFree(d_colIn));
    CUDA_CHECK(cudaFree(d_wIn));
    CUDA_CHECK(cudaFree(d_dist_cur));
    CUDA_CHECK(cudaFree(d_dist_next));
    CUDA_CHECK(cudaFree(d_frontier_curr));
    CUDA_CHECK(cudaFree(d_frontier_next));
    CUDA_CHECK(cudaFree(d_frontier_next_size));
    CUDA_CHECK(cudaEventDestroy(tic));
    CUDA_CHECK(cudaEventDestroy(toc));
}
