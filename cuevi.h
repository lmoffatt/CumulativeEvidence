#ifndef CUEVI_H
#define CUEVI_H
#include <algorithm>
#include <cassert>
#include <random>
#include <vector>
#include "mcmc.h"
#include "parallel_tempering.h"

auto stretch_move(const Parameters &Xk, const Parameters &Xj, double z) {
  assert((Xj.size() == Xk.size()) && "sum of vector fields of different sizes");
  auto out = Xj;
  for (std::size_t i = 0; i < Xj.size(); ++i)
    out[i] += z * (Xk[i] - Xj[i]);
  return out;
}


auto random_portion_of_Index(const Indexes &indexes, std::mt19937_64 &mt,
                             double portion) {
  auto index = indexes;
  std::shuffle(index.begin(), index.end(), mt);
  std::size_t nh = index.size() * portion;
  auto out = Indexes(index.begin(), index.begin() + nh);
  std::sort(out.begin(), out.end());
  return out;
}

auto generate_Indexes(std::mt19937_64 &mt, std::size_t num_samples,
                      std::size_t num_parameters, double num_jumps_per_decade) {

  std::size_t n_jumps =
      std::floor(num_jumps_per_decade *
                 (std::log10(num_samples) - std::log10(2 * num_parameters)));
  double portion = std::pow(10, 1.0 / num_jumps_per_decade);
  auto out = std::vector<Indexes>(n_jumps);
  auto index = Indexes(num_samples);
  std::iota(index.begin(), index.end(), 0u);
  for (auto i = 0u; i < n_jumps; ++i) {
    out[i] = index;
    index = random_portion_of_Index(index, mt, portion);
  }
  return index;
}

struct initseed {
  constexpr static std::string name() { return "initseed"; }
};

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

auto init_mts(std::mt19937_64 &mt, std::size_t n) {
  std::uniform_int_distribution<typename std::mt19937_64::result_type> useed;
  std::vector<std::mt19937_64> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
    out.emplace_back(useed(mt));
  return out;
}

auto get_beta_list(double jump_factor, double stops_at, bool includes_zero) {
  std::size_t num_beta = std::ceil(stops_at / std::log(jump_factor)) + 1;

  auto beta_size = num_beta;
  if (includes_zero)
    beta_size = beta_size + 1;

  auto out = std::vector<double>(beta_size, 0.0);
  for (std::size_t i = 0; i < beta_size; ++i)
    out[beta_size-1-i] = std::pow(jump_factor, i);
  return out;
}

template <class T> using ensemble = std::vector<T>;
template <class T> using by_fraction = std::vector<T>;
template <class T> using by_beta = std::vector<T>;

using sample_Parameters = auto (*)(std::mt19937_64 &) -> Parameters;

template <class Variable>
using samples_Data = auto (*)(std::mt19937_64 &, const Parameters &,
                              const Variable &) -> IndexedData;

using calculates_PriorProb = auto (*)(const Parameters &) -> double;

template <class Variable>
using calculates_Likelihood = auto (*)(const Parameters &, const IndexedData &,
                                       const Variable &) -> double;

auto init_parameters(std::mt19937_64 &mts, sample_Parameters model) {
  return model(mts);
}

auto calc_prior(calculates_PriorProb priorfunct, const Parameters &par) {
  return priorfunct(par);
}

template <class Variable>
auto calc_lik(calculates_Likelihood<Variable> likfunct, const Parameters &par,
              const IndexedData &y, const Variable &x) {
  return likfunct(par, y, x);
}

struct mcmc {
  Parameters parameter;
  double logP;
  double logL;
};

struct thermo_mcmc {
  ensemble<by_beta<mcmc>> walkers;
  ensemble<by_beta<std::size_t>> i_walkers;
};

struct cuevi_mcmc {

  by_fraction<Indexes> ind;

  by_fraction<ensemble<Parameters>> parameter;
  by_beta<ensemble<double>> logP;
  by_beta<ensemble<double>> logL_0;
  by_beta<ensemble<double>> logL_1;
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


template <class Variable>
auto init_thermo_mcmc(std::size_t n_walkers,
                      std::size_t n_beta,
                      ensemble<std::mt19937_64> &mt,
                      sample_Parameters modelsample,
                      calculates_PriorProb priorfunction,
                      calculates_Likelihood<Variable> likfunction,
                      const IndexedData &y, const Variable &x) {

  ensemble<by_beta<std::size_t>> i_walker(n_walkers,
                                          by_beta<std::size_t>(n_beta));
  ensemble<by_beta<mcmc>> walker(n_walkers, by_beta<mcmc>(n_beta));

  for (std::size_t half=0; half<2; ++half)
#pragma omp parallel for
  for (std::size_t iiw = 0; iiw < n_walkers/2; ++iiw) {
    auto iw=iiw+half*n_walkers/2;
    for (std::size_t i = 0; i < n_beta; ++i) {
      i_walker[iw][i] = iw + i * n_walkers;
      walker[iw][i] =
          init_mcmc(mt[iiw], modelsample, priorfunction, likfunction, y, x);
    }
  }
  return thermo_mcmc{walker, i_walker};
}


std::pair<std::pair<std::size_t,std::size_t>, bool> check_iterations(std::pair<std::size_t,std::size_t> current_max,const thermo_mcmc&)
{
  if (current_max.first>=current_max.second)
      return std::pair(std::pair(0ul,current_max.second),true);
  else
      return std::pair(std::pair(current_max.first+1,current_max.second),false);
};



using walker_distribution =
    ensemble<std::uniform_int_distribution<std::size_t>>;

using z_stretch_distribution = ensemble<std::uniform_real_distribution<double>>;

using p_distribution = ensemble<std::uniform_real_distribution<double>>;

template <class Variable>
auto &step_stretch_thermo_mcmc(const by_beta<double>& beta,
    ensemble<std::mt19937_64> &mt,
                               thermo_mcmc &current,
                               calculates_PriorProb priorfunction,
                               calculates_Likelihood<Variable> likfunction,
                               const IndexedData &y, const Variable &x,
                               double alpha_stretch = 2) {
  auto n_walkers = mt.size();
  auto n_beta = beta.size();
  auto n_par = current.walkers[0][0].parameter.size();

  std::uniform_int_distribution<std::size_t> uniform_walker(0, n_walkers / 2);
  std::vector<std::uniform_int_distribution<std::size_t>> udist(n_walkers,
                                                                uniform_walker);

  std::uniform_real_distribution<double> uniform_stretch_zdist(
      1.0 / alpha_stretch, alpha_stretch);
  std::vector<std::uniform_real_distribution<double>> zdist(
      n_walkers, uniform_stretch_zdist);

  std::uniform_real_distribution<double> uniform_real(0, 1);
  std::vector<std::uniform_real_distribution<double>> rdist(n_walkers,
                                                            uniform_real);

  for (bool half : {false, true})
#pragma omp parallel for
    for (std::size_t i = 0; i < n_walkers / 2; ++i) {
      auto iw = half ? i + n_walkers / 2 : i;
      auto j = udist[i](mt[i]);
      auto jw = half ? j : j + n_walkers / 2;
      for (std::size_t ib = 0; ib < n_beta; ++ib) {
        // we can try in the outer loop

        auto z = zdist[i](mt[i]);
        auto r = rdist[i](mt[i]);

        // candidate[ib].walkers[iw].
        auto ca_par = stretch_move(current.walkers[iw][ib].parameter,
                                   current.walkers[jw][ib].parameter, z);
        auto ca_logP = priorfunction(ca_par);
        auto ca_logL = likfunction(ca_par, y, x);
        auto dthLogL =
            ca_logP - current.walkers[iw][ib].logP +
            beta[ib] * (ca_logL - current.walkers[iw][ib].logL);
        auto pJump = std::min(1.0, std::pow(z, n_par - 1) * std::exp(dthLogL));
        if (pJump > r) {
          current.walkers[iw][ib].parameter = std::move(ca_par);
          current.walkers[iw][ib].logP = ca_logP;
          current.walkers[iw][ib].logL = ca_logL;
        }
      }
    }
  return current;
}

auto &thermo_jump_mcmc(const by_beta<double>& beta,
                       std::mt19937_64 &mt, ensemble<std::mt19937_64> &mts,
                       thermo_mcmc &current) {
  std::uniform_real_distribution<double> uniform_real(0, 1);
  auto n_walkers = mts.size();
  auto n_beta = beta.size();
  auto n_par = current.walkers[0][0].parameter.size();
  std::uniform_int_distribution<std::size_t> booldist(0, 1);
  auto half = booldist(mt) == 1;

  Indexes landing_walker(n_walkers / 2);
  std::iota(landing_walker.begin(), landing_walker.end(), 0);
  std::shuffle(landing_walker.begin(), landing_walker.end(), mt);
  std::vector<std::uniform_real_distribution<double>> rdist(n_walkers,
                                                            uniform_real);

#pragma omp parallel for
  for (std::size_t i = 0; i < n_walkers / 2; ++i) {
    auto iw = half ? i + n_walkers / 2 : i;
    auto j = landing_walker[i];
    auto jw = half ? j : j + n_walkers / 2;
    for (std::size_t ib = 0; ib < n_beta - 1; ++ib) {

      auto r = rdist[i](mts[i]);
      double logA =
          -(beta[ib] - beta[ib + 1]) *
          (current.walkers[iw][ib].logL - current.walkers[jw][ib + 1].logL);
      auto pJump = std::min(1.0, std::exp(logA));
      if (pJump > r) {
        std::swap(current.walkers[iw][ib], current.walkers[jw][ib + 1]);
        std::swap(current.i_walkers[iw][ib], current.i_walkers[jw][ib + 1]);
      }
    }
  }
  return current;
}


template<class Parameters,class Variable>
auto push_back_new_beta(thermo_mcmc<Parameters>& current, ensemble<std::mt19937_64>& mts, sample_Parameters modelsample,
                   calculates_PriorProb priorfunction,
                   calculates_Likelihood<Variable> likfunction,
                   const IndexedData &y, const Variable &x)
{
  auto n_walkers=current.walkers.size();
  auto n_beta_old=current.walkers[0].size();
  for (std::size_t half=0; half<2; ++half)
  for (std::size_t i=0; i<n_walkers/2; ++i)
  {
      auto iw= i+ half*n_walkers/2;
      current.walkers[iw].push_back(init_mcmc(mts[i],modelsample,priorfunction,likfunction,y,x));
      current.i_walkers[iw].push_back(n_beta_old*n_walkers+iw);
  }
  return current;
}



template <class Variable, class ConvergenceConditions>
auto thermo_impl(sample_Parameters modelsample,
                 calculates_PriorProb priorfunction,
                 calculates_Likelihood<Variable> likfunction,
                 checks_convergence<ConvergenceConditions> does_converge,
                 ConvergenceConditions converge_cond, const IndexedData &y,
                 const Variable &x, std::size_t num_scouts_per_ensemble,
                 double jump_factor, double stops_at, bool includes_zero,
                 std::size_t initseed) {

  auto mt = init_mt(initseed);
  auto n_walkers= num_scouts_per_ensemble;
  auto mts = init_mts(mt, num_scouts_per_ensemble / 2);

  auto beta = get_beta_list(jump_factor, stops_at, includes_zero);



  auto n_beta=beta.size();

  auto beta_run = by_beta<double>(beta.rend()-2, beta.rend());

  auto current = init_thermo_mcmc(n_walkers,beta_run.size(),mts,  modelsample, priorfunction,
                                  likfunction, y, x);

  auto n_par = current.walkers[0][0].parameter.size();

  auto mcmc_run = does_converge(converge_cond, current);

  std::size_t thermo_jumps_every = n_par;


  while (beta_run.size() < beta.size() || !mcmc_run.second) {
    while (!mcmc_run.second) {
      for (std::size_t i_p = 0; i_p < thermo_jumps_every; ++i_p) {
        current = step_stretch_thermo_mcmc(beta_run,mts, current, priorfunction,
                                           likfunction, y, x);
        mcmc_run = does_converge(mcmc_run.first, current);
      }
      current = thermo_jump_mcmc(beta_run,mt, mts, current);
      mcmc_run = does_converge(mcmc_run.first, current);
    }
    if (beta_run.size()<beta.size())
    {
      beta_run.insert(beta_run.begin(),beta[beta_run.size()]);
      current=push_back_new_beta(current,mts,modelsample,priorfunction,likfunction,y,x);
      mcmc_run = does_converge(mcmc_run.first, current);
    }

  }

  return std::pair(mcmc_run,current);

}




template <class Variable, class ConvergenceConditions>
auto thermo_max_iter(sample_Parameters modelsample,
                 calculates_PriorProb priorfunction,
                 calculates_Likelihood<Variable> likfunction,
                  const IndexedData &y,
                 const Variable &x, std::size_t num_scouts_per_ensemble,
                 double jump_factor, double stops_at, bool includes_zero,
                 std::size_t initseed)
{
  return thermo_impl(modelsample,priorfunction,likfunction,check_iterations,   std::pair(0,1000),y,x,num_scouts_per_ensemble,jump_factor,stops_at,includes_zero,initseed);
}







#endif // CUEVI_H
