// DijkstraCudaOpt.h
#pragma once
#include "ShortestPathAlgorithm.h"

class DijkstraCudaOpt : public ShortestPathAlgorithm {
public:
  void run(int source, std::vector<int>& h_dist) override;
};
