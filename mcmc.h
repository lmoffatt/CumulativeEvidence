#ifndef MCMC_H
#define MCMC_H

#include "matrix.h"
#include <cassert>
#include <random>
#include <vector>
#include "distributions.h"
using Parameters = Matrix<double>;
using Data = Matrix<double>;

using Indexes = std::vector<std::size_t>;

using IndexedData = std::pair<Indexes, Data>;

/*
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
*/
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



template <class Model, class Parameters,class Variables,class IndexedData>
concept is_model = requires(Model const &m_const,
                            Model& m,
                            const Parameters& p,
                            const Variables& var,
                            const IndexedData& y) {
  {
    sample(std::declval<std::mt19937_64 &>(),m)
  } -> std::convertible_to<Parameters>;

  {
    simulate(std::declval<std::mt19937_64 &>(),m_const,p,var)
  }-> std::convertible_to<IndexedData>;

  {
    logPrior(m_const,p)
  }->std::convertible_to<double>;

  {
    logLikelihood(m_const,p,var,y)
  }->std::convertible_to<double>;
};


template<class D>
requires(is_Distribution<D>)
auto sample(std::mt19937_64& mt, D&d)
{return d(mt);}

template<class D, class T>
requires(is_Distribution_of<D,T>)
auto logPrior(const D&d, const T& x )
{return d.logP(x);}





template<class Parameters>
struct mcmc {
  Parameters parameter;
  double logP;
  double logL;
};

template <class Model, class Variables,class IndexedData,
         class Parameters=std::decay_t<
             decltype(sample(std::declval<std::mt19937_64 &>(), std::declval<Model&>()))>>
requires (is_model<Model,Parameters,Variables,IndexedData>)
auto init_mcmc(std::mt19937_64 &mt, Model& m,
               const IndexedData &y, const Variables &x) {
  auto par = sample(mt,m);
  double logP = logPrior(m,par);
  double logL = logLikelihood(m,par, y,x);
  return mcmc{std::move(par), logP, logL};
}

#endif // MCMC_H
