#pragma once
#include "ShortestPathAlgorithm.h"


class BellmanFordCuda : public ShortestPathAlgorithm {
public:
  // Run the Bellman-Ford algorithm on the GPU
  void run(int source, std::vector<int>& dist) override;
};
