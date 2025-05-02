#include "BellmanFordCuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define INF 1000000

__global__ void relaxEdges(int V, int E, int* d_dist, int* d_u, int* d_v, int* d_w, bool* d_hasUpdate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;

    int u = d_u[idx];
    int v = d_v[idx];
    int weight = d_w[idx];

    // Relaxation condition: if dist[u] + weight < dist[v], update dist[v]
    if (d_dist[u] != INF && d_dist[u] + weight < d_dist[v]) {
        d_dist[v] = d_dist[u] + weight;
        *d_hasUpdate = true;  // Indicate that an update happened
    }
}

void BellmanFordCuda::run(int source, std::vector<int>& dist) {
    if (V <= 0 || E <= 0) {
        std::cerr << "Error: Number of vertices or edges is zero or uninitialized!" << std::endl;
        return;
    }
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));


    // Device arrays for storing distances, edges, and update flag
    int* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, V * sizeof(int)));
    std::vector<int> hostDist(V, INT_MAX);
    hostDist[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_dist, hostDist.data(),
                            V * sizeof(int), cudaMemcpyHostToDevice));
    bool* d_hasUpdate;
    CUDA_CHECK(cudaMalloc((void**)&d_hasUpdate, sizeof(bool)));


    bool h_hasUpdate = false;
    CUDA_CHECK(cudaMemcpy(d_hasUpdate, &h_hasUpdate, sizeof(bool), cudaMemcpyHostToDevice));


    // Bellman-Ford for V-1 iterations
    for (int iter = 0; iter < V - 1; ++iter) {
        h_hasUpdate = false;
        CUDA_CHECK(cudaMemcpy(d_hasUpdate, &h_hasUpdate, sizeof(bool), cudaMemcpyHostToDevice));

        // Launch the CUDA kernel
        int blockSize = 1024;  // Number of threads per block
        int numBlocks = (E + blockSize - 1) / blockSize;
        relaxEdges<<<numBlocks, blockSize>>>(V, E, d_dist, d_src, d_dst, d_w, d_hasUpdate);

        // Check for kernel errors
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check if any updates were made in this iteration
        CUDA_CHECK(cudaMemcpy(&h_hasUpdate, d_hasUpdate, sizeof(bool), cudaMemcpyDeviceToHost));
        if (!h_hasUpdate) {
            break;  // No updates, so we can exit early
        }
    }

    // Copy the final distances back to host memory
    CUDA_CHECK(cudaMemcpy(hostDist.data(), d_dist,
                          V * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&lastTimeMs, start, stop));
    // Clean up device memory
    cudaFree(d_dist);
    cudaFree(d_hasUpdate);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}