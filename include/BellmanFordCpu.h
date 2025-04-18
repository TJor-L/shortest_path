#pragma once
#include "ShortestPathAlgorithm.h"
#include <vector>
#include <limits>
#include <chrono>
#include <iostream>

class BellmanFordCpu : public ShortestPathAlgorithm {
private:
  std::vector<Edge> edges_; 

public:
  void setupGraph(int V_, int E_, const std::vector<Edge>& edges) override {
    V = V_;
    E = E_;
    edges_ = edges;  // Initialize edges
  }

  void run(int source, std::vector<int>& dist) override {
    dist.assign(V, std::numeric_limits<int>::max());
    dist[source] = 0;  // Distance from source to itself is 0

    // Relax edges for V - 1 iterations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < V - 1; ++i) {
        bool updated = false;
        for (auto& e : edges_) {
            if (dist[e.u] != std::numeric_limits<int>::max() && dist[e.u] + e.w < dist[e.v]) {
                dist[e.v] = dist[e.u] + e.w;
                updated = true;
            }
        }
        if (!updated) break;

    }
    auto end = std::chrono::high_resolution_clock::now();
    lastTimeMs = std::chrono::duration<float, std::milli>(end - start).count();  // Calculate time taken
    
  }
};
