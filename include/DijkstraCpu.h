#pragma once
#include "ShortestPathAlgorithm.h"
#include <vector>
#include <queue>
#include <limits>
#include <chrono>
#include <utility>

class DijkstraCpu : public ShortestPathAlgorithm {
private:
  std::vector<std::vector<std::pair<int,int>>> adj;
public:

  void setupGraph(int V_, int E_, const std::vector<Edge>& edges) override {
    V = V_; E = E_;
    adj.assign(V, {});
    for (auto &e : edges) {
      adj[e.u].emplace_back(e.v, e.w);
    }
  }

  void run(int source, std::vector<int>& dist) override {
    dist.assign(V, std::numeric_limits<int>::max());
    dist[source] = 0;

    auto cmp = [](auto &a, auto &b){ return a.first > b.first; };
    std::priority_queue<std::pair<int,int>,
        std::vector<std::pair<int,int>>,
        decltype(cmp)> pq(cmp);
    pq.emplace(0, source);

    auto start = std::chrono::high_resolution_clock::now();
    while (!pq.empty()) {
      auto [d,u] = pq.top(); pq.pop();
      if (d != dist[u]) continue;
      for (auto &pr : adj[u]) {
        int v = pr.first, w = pr.second;
        if (dist[u] + w < dist[v]) {
          dist[v] = dist[u] + w;
          pq.emplace(dist[v], v);
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    lastTimeMs = std::chrono::duration<float,std::milli>(end - start).count();
  }
};
