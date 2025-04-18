#pragma once
#include "ShortestPathAlgorithm.h"

class FloydWarshallCuda : public ShortestPathAlgorithm {
public:
  void run(int source, std::vector<int>& h_dist) override;
};
