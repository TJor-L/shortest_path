#pragma once
#include <vector>
#include <random>

struct Edge {
  int u, v, w;
};

static std::vector<Edge>
generateRandomGraph(int V, float density, int maxWeight = 10) {
  std::mt19937                     gen(std::random_device{}());
  std::uniform_real_distribution<> prob(0.0f, 1.0f);
  std::uniform_int_distribution<>  wdist(1, maxWeight);

  std::vector<Edge> edges;
  edges.reserve(static_cast<size_t>(V) * V * density);

  for (int u = 0; u < V; ++u) {
    for (int v = 0; v < V; ++v) {
      if (u == v) continue;
      if (prob(gen) < density) {
        edges.push_back({u, v, wdist(gen)});
      }
    }
  }
  return edges;
}
