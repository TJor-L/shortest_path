#pragma once
#include "ShortestPathAlgorithm.h"
#include <vector>
#include <limits>
#include <chrono>

class FloydWarshallCpu : public ShortestPathAlgorithm {
private:
  // weight matrix: weight[u][v] is initial edge weight (INF if none)
  std::vector<std::vector<int>> weight;

public:
  void setupGraph(int V_, int E_, const std::vector<Edge>& edges) override {
    V = V_; E = E_;
    const int INF = std::numeric_limits<int>::max();
    // initialize V×V matrix to INF, 0 on diagonal
    weight.assign(V, std::vector<int>(V, INF));
    for (int i = 0; i < V; ++i) {
      weight[i][i] = 0;
    }
    // fill in given edges (directed)
    for (auto &e : edges) {
      weight[e.u][e.v] = e.w;
    }
  }

  void run(int source, std::vector<int>& dist) override {
    // make a working copy so original weights stay intact
    std::vector<std::vector<int>> distMat = weight;
    const int INF = std::numeric_limits<int>::max();

    auto start = std::chrono::high_resolution_clock::now();

    // Floyd–Warshall core
    for (int k = 0; k < V; ++k) {
      for (int i = 0; i < V; ++i) {
        if (distMat[i][k] == INF) continue;
        for (int j = 0; j < V; ++j) {
          if (distMat[k][j] == INF) continue;
          int throughK = distMat[i][k] + distMat[k][j];
          if (throughK < distMat[i][j]) {
            distMat[i][j] = throughK;
          }
        }
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    lastTimeMs = std::chrono::duration<float, std::milli>(end - start).count();

    // extract the 'source' row into dist[], mapping INF to max again
    dist.resize(V);
    for (int v = 0; v < V; ++v) {
      dist[v] = (distMat[source][v] == INF
                  ? std::numeric_limits<int>::max()
                  : distMat[source][v]);
    }
  }
};
