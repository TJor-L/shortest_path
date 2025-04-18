#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <ctime>
#include <limits>
#include <filesystem>
#include "GraphGenerator.h"
#include "DijkstraCuda.h"
#include "DijkstraCpu.h"
#include "BellmanFordCuda.h"
#include "BellmanFordCpu.h"
#include "FloydWarshallCuda.h"
#include "FloydWarshallCpu.h"
#include "DijkstraCudaOpt.h"

struct Config {
  int   V         = 1000;
  float density   = 0.01f;
  int   maxWeight = 10;
  int   runs      = 5;
  int   source    = 0;
};

static Config readConfig(const std::string& fn) {
  Config cfg;
  std::ifstream in(fn);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0]=='#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    auto key = line.substr(0, eq);
    auto val = line.substr(eq + 1);
    if      (key=="V")         cfg.V         = std::stoi(val);
    else if (key=="density")   cfg.density   = std::stof(val);
    else if (key=="maxWeight") cfg.maxWeight = std::stoi(val);
    else if (key=="runs")      cfg.runs      = std::stoi(val);
    else if (key=="source")    cfg.source    = std::stoi(val);
  }
  return cfg;
}

int main() {
  auto cfg = readConfig("config.ini");
  std::filesystem::path out_dir("results");
  if (!std::filesystem::exists(out_dir)) {
    std::filesystem::create_directories(out_dir);
  }

  auto now = std::time(nullptr);
  char buf[64];
  std::strftime(buf, sizeof(buf),
                "results/results_%Y%m%d_%H%M%S.csv",
                std::localtime(&now));

  std::ofstream out(buf);

  out << "Algorithm,Run,V,E,Density,TimeMs\n";

  std::vector<std::pair<std::string, std::unique_ptr<ShortestPathAlgorithm>>> algos;

  algos.emplace_back("DijkstraCpu",
                     std::make_unique<DijkstraCpu>());
  algos.emplace_back("BellmanFordCuda",
                     std::make_unique<BellmanFordCuda>());
  algos.emplace_back("BellmanFordCpu",
                     std::make_unique<BellmanFordCpu>());
  algos.emplace_back("FloydWarshallCuda",
                     std::make_unique<FloydWarshallCuda>());
  algos.emplace_back("FloydWarshallCpu",
                     std::make_unique<FloydWarshallCpu>());
  algos.emplace_back("DijkstraCudaOpt",
                    std::make_unique<DijkstraCudaOpt>());
  algos.emplace_back("DijkstraCuda",
                    std::make_unique<DijkstraCuda>());

  for (int run = 0; run < cfg.runs; ++run) {
    auto edges = generateRandomGraph(
                   cfg.V, cfg.density, cfg.maxWeight);
    int E = static_cast<int>(edges.size());

    for (auto& [name, algo] : algos) {
      algo->setupGraph(cfg.V, E, edges);
      std::vector<int> dist(cfg.V);
      algo->run(cfg.source, dist);
      out << name      << ","
          << run       << ","
          << cfg.V     << ","
          << E         << ","
          << cfg.density << ","
          << algo->getLastTimeMs()
          << "\n";
    }
  }

  out.close();
  std::cout << "Results saved to " << buf << "\n";
  return 0;
}
