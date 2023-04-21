#ifndef MCMC_H
#define MCMC_H

#include <cassert>
#include <random>
#include <vector>
using Parameters = std::vector<double>;
using Data = std::vector<double>;

using Indexes = std::vector<std::size_t>;

using IndexedData = std::pair<Indexes, Data>;

auto operator+(const Parameters &x, const Parameters &y) {
  assert((x.size() == y.size()) && "sum of vector fields of different sizes");
  auto out = x;
  for (std::size_t i = 0; i < x.size(); ++i)
    out[i] = x[i] + y[i];
  return out;
}
auto operator-(const Parameters &x, const Parameters &y) {
  assert((x.size() == y.size()) && "sum of vector fields of different sizes");
  auto out = x;
  for (std::size_t i = 0; i < x.size(); ++i)
    out[i] = x[i] - y[i];
  return out;
}

auto calc_seed(typename std::mt19937_64::result_type initseed) {

  if (initseed == 0) {
    std::random_device rd;
    std::uniform_int_distribution<typename std::mt19937_64::result_type> useed;

    return useed(rd);
  } else
    return initseed;
}

auto init_mt(typename std::mt19937_64::result_type initseed) {
  initseed = calc_seed(initseed);
  return std::mt19937_64(initseed);
}

using sample_Parameters = auto (*)(std::mt19937_64 &) -> Parameters;

template <class Variable>
using samples_Data = auto (*)(std::mt19937_64 &, const Parameters &,
                              const Variable &) -> IndexedData;

using calculates_PriorProb = auto (*)(const Parameters &) -> double;

template <class Variable>
using calculates_Likelihood = auto (*)(const Parameters &, const IndexedData &,
                                       const Variable &) -> double;

struct mcmc {
  Parameters parameter;
  double logP;
  double logL;
};

template <class Variable>
auto init_mcmc(std::mt19937_64 &mt, sample_Parameters modelsample,
               calculates_PriorProb priorfunction,
               calculates_Likelihood<Variable> likfunction,
               const IndexedData &y, const Variable &x) {
  auto par = modelsample(mt);
  double logP = priorfunction(par);
  double logL = likfunction(par, y, x);
  return mcmc{std::move(par), logP, logL};
}

#endif // MCMC_H
