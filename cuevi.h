#ifndef CUEVI_H
#define CUEVI_H
#include "mcmc.h"
#include "parallel_tempering.h"
#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

template <class T> using by_fraction = std::vector<T>;

auto random_portion_of_Index(const DataIndexes &indexes, std::mt19937_64 &mt,
                             double portion) {
  auto index = indexes;
  std::shuffle(index.begin(), index.end(), mt);
  std::size_t nh = index.size() * portion;
  auto out = DataIndexes(index.begin(), index.begin() + nh);
  std::sort(out.begin(), out.end());
  return out;
}

auto generate_Indexes(std::mt19937_64 &mt, std::size_t num_samples,
                      std::size_t num_parameters, double num_jumps_per_decade) {

  std::size_t n_jumps =
      std::floor(num_jumps_per_decade *
                 (std::log10(num_samples) - std::log10(2 * num_parameters)));
  double portion = std::pow(10, 1.0 / num_jumps_per_decade);
  auto out = std::vector<DataIndexes>(n_jumps);
  auto index = DataIndexes(num_samples);
  std::iota(index.begin(), index.end(), 0u);
  for (auto i = 0u; i < n_jumps; ++i) {
    out[i] = index;
    index = random_portion_of_Index(index, mt, portion);
  }
  return index;
}

template <class Parameters> struct mcmc2 : public mcmc<Parameters> {
  double logPa;
};

template <class Parameters> struct cuevi_mcmc {
  by_fraction<by_beta<double>> beta;
  ensemble<by_fraction<by_beta<mcmc2<Parameters>>>> walkers;
  ensemble<by_fraction<by_beta<std::size_t>>> i_walkers;
};

template <class Observer, class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
void single_step_stretch_cuevi_mcmc(
    cuevi_mcmc<Parameters> &current, Observer &obs,
    ensemble<std::mt19937_64> &mt,
    std::vector<std::uniform_real_distribution<double>> &rdist,
    Model const &model, const by_fraction<DataType> &y,
    const by_fraction<Variables> &x, std::size_t n_par, std::size_t i,
    std::size_t iw, std::size_t jw, std::size_t ib, std::size_t i_fr) {
  auto z = std::pow(rdist[i](mt[i]) + 1, 2) / 2.0;
  auto r = rdist[i](mt[i]);
  // candidate[ib].walkers[iw].
  auto ca_par = stretch_move(current.walkers[iw][i_fr][ib].parameter,
                             current.walkers[jw][i_fr][ib].parameter, z);
  auto ca_logP = logPrior(model, ca_par);
  auto ca_logL = logLikelihood(model, ca_par, y[i_fr], x[i_fr]);

  if ((ca_logP) && (ca_logL)) {
    auto dthLogL = ca_logP.value() - current.walkers[iw][i_fr][ib].logP +
                   current.beta[0][ib] *
                       (ca_logL.value() - current.walkers[iw][i_fr][ib].logL);
    auto pJump = std::min(1.0, std::pow(z, n_par - 1) * std::exp(dthLogL));
    observe_step_stretch_thermo_mcmc(
        obs[iw][i_fr][ib], jw, z, r, current.walkers[iw][i_fr][ib].parameter,
        current.walkers[jw][ib].parameter, current.walkers[iw][ib].logP,
        ca_logP, current.walkers[iw][i_fr][ib].logL, ca_logL, pJump >= r);
    if (pJump >= r) {
      current.walkers[iw][i_fr][ib].parameter = std::move(ca_par);
      current.walkers[iw][i_fr][ib].logPa = ca_logP.value();
      current.walkers[iw][i_fr][ib].logP = ca_logP.value();
      current.walkers[iw][i_fr][ib].logL = ca_logL.value();
    }
  }
}

template <class Observer, class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
void double_step_stretch_cuevi_mcmc(
    cuevi_mcmc<Parameters> &current, Observer &obs,
    ensemble<std::mt19937_64> &mt,
    std::vector<std::uniform_real_distribution<double>> &rdist,
    Model const &model, const by_fraction<DataType> &y,
    const by_fraction<Variables> &x, std::size_t n_par, std::size_t i,
    std::size_t iw, std::size_t jw, std::size_t ib, std::size_t i_fr)

{

  auto z = std::pow(rdist[i](mt[i]) + 1, 2) / 2.0;
  auto r = rdist[i](mt[i]);

  // candidate[ib].walkers[iw].
  auto ca_par = stretch_move(current.walkers[iw][i_fr][ib].parameter,
                             current.walkers[jw][i_fr][ib].parameter, z);
  auto ca_logP = logPrior(model, ca_par);
  auto ca_logL0 = logLikelihood(model, ca_par, y[i_fr], x[i_fr]);
  auto ca_logL1 = logLikelihood(model, ca_par, y[i_fr + 1], x[i_fr + 1]);

  if ((ca_logP) && (ca_logL0) && (ca_logL1)) {
    auto dthLogL = ca_logP.value() - current.walkers[iw][i_fr][ib].logP +
                   current.beta[i_fr][ib] *
                       (ca_logL0.value() - current.walkers[iw][i_fr][ib].logL);
    auto pJump = std::min(1.0, std::pow(z, n_par - 1) * std::exp(dthLogL));
    observe_step_stretch_thermo_mcmc(
        obs[iw][i_fr][ib], jw, z, r, current.walkers[iw][i_fr][ib].parameter,
        current.walkers[jw][ib].parameter, current.walkers[iw][ib].logP,
        ca_logP, current.walkers[iw][i_fr][ib].logL, ca_logL1, pJump >= r);
    if (pJump >= r) {
      current.walkers[iw][i_fr][ib].parameter = std::move(ca_par);
      current.walkers[iw][i_fr][ib].logPa = ca_logP.value();
      current.walkers[iw][i_fr][ib].logP = ca_logP.value();
      current.walkers[iw][i_fr][ib].logL = ca_logL0.value();
      current.walkers[iw][i_fr + 1][0].parameter = std::move(ca_par);
      current.walkers[iw][i_fr + 1][0].logPa = ca_logP.value();
      current.walkers[iw][i_fr + 1][0].logP =
          ca_logP.value() + ca_logL0.value();
      current.walkers[iw][i_fr + 1][0].logL =
          ca_logL1.value() - ca_logL0.value();
    }
  }
}

template <class Observer, class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
void middle_step_stretch_cuevi_mcmc(
    cuevi_mcmc<Parameters> &current, Observer &obs,
    ensemble<std::mt19937_64> &mt,
    std::vector<std::uniform_real_distribution<double>> &rdist,
    Model const &model, const by_fraction<DataType> &y,
    const by_fraction<Variables> &x, std::size_t n_par, std::size_t i,
    std::size_t iw, std::size_t jw, std::size_t ib, std::size_t i_fr)

{

  auto z = std::pow(rdist[i](mt[i]) + 1, 2) / 2.0;
  auto r = rdist[i](mt[i]);

  // candidate[ib].walkers[iw].
  auto ca_par = stretch_move(current.walkers[iw][i_fr][ib].parameter,
                             current.walkers[jw][i_fr][ib].parameter, z);
  auto ca_logP = logPrior(model, ca_par);
  auto ca_logL0 = logLikelihood(model, ca_par, y[i_fr - 1], x[i_fr - 1]);
  auto ca_logL1 = logLikelihood(model, ca_par, y[i_fr], x[i_fr]);

  if ((ca_logP) && (ca_logL0) && (ca_logL1)) {
    auto dthLogL =
        ca_logP.value() + ca_logL0.value() -
        current.walkers[iw][i_fr][ib].logP +
        current.beta[i_fr][ib] * (ca_logL1.value() - ca_logL0.value() -
                                  current.walkers[iw][i_fr][ib].logL);
    auto pJump = std::min(1.0, std::pow(z, n_par - 1) * std::exp(dthLogL));
    observe_step_stretch_thermo_mcmc(
        obs[iw][ib], jw, z, r, current.walkers[iw][ib].parameter,
        current.walkers[jw][ib].parameter, current.walkers[iw][ib].logP,
        ca_logP, current.walkers[iw][ib].logL, ca_logL0, pJump >= r);
    if (pJump >= r) {
      current.walkers[iw][i_fr][ib].parameter = std::move(ca_par);
      current.walkers[iw][i_fr][ib].logPa = ca_logP.value();
      current.walkers[iw][i_fr][ib].logP = ca_logP.value() + ca_logL0.value();
      current.walkers[iw][i_fr][ib].logL = ca_logL1.value() - ca_logL0.value();
    }
  }
}

template <class Observer, class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
void triple_step_stretch_cuevi_mcmc(
    cuevi_mcmc<Parameters> &current, Observer &obs,
    ensemble<std::mt19937_64> &mt,
    std::vector<std::uniform_real_distribution<double>> &rdist,
    Model const &model, const by_fraction<DataType> &y,
    const by_fraction<Variables> &x, std::size_t n_par, std::size_t i,
    std::size_t iw, std::size_t jw, std::size_t ib, std::size_t i_fr) {

  auto z = std::pow(rdist[i](mt[i]) + 1, 2) / 2.0;
  auto r = rdist[i](mt[i]);

  // candidate[ib].walkers[iw].
  auto ca_par = stretch_move(current.walkers[iw][i_fr][ib].parameter,
                             current.walkers[jw][i_fr][ib].parameter, z);
  auto ca_logP = logPrior(model, ca_par);
  auto ca_logL0 = logLikelihood(model, ca_par, y[i_fr - 1], x[i_fr - 1]);
  auto ca_logL1 = logLikelihood(model, ca_par, y[i_fr], x[i_fr]);
  auto ca_logL2 = logLikelihood(model, ca_par, y[i_fr + 1], x[i_fr + 1]);

  if ((ca_logP) && (ca_logL0) && (ca_logL1) && (ca_logL2)) {
    auto dthLogL =
        ca_logP.value() + ca_logL0.value() -
        current.walkers[iw][i_fr][ib].logP +
        current.beta[i_fr][ib] * (ca_logL1.value() - ca_logL0.value() -
                                  current.walkers[iw][i_fr][ib].logL);
    auto pJump = std::min(1.0, std::pow(z, n_par - 1) * std::exp(dthLogL));
    observe_step_stretch_thermo_mcmc(
        obs[iw][ib], jw, z, r, current.walkers[iw][ib].parameter,
        current.walkers[jw][ib].parameter, current.walkers[iw][ib].logP,
        ca_logP, current.walkers[iw][ib].logL, ca_logL0, pJump >= r);
    if (pJump >= r) {
      current.walkers[iw][i_fr][ib].parameter = std::move(ca_par);
      current.walkers[iw][i_fr][ib].logPa = ca_logP.value();
      current.walkers[iw][i_fr][ib].logP = ca_logP.value() + ca_logL0.value();
      current.walkers[iw][i_fr][ib].logL = ca_logL1.value() - ca_logL0.value();
      current.walkers[iw][i_fr + 1][0].parameter = std::move(ca_par);
      current.walkers[iw][i_fr + 1][0].logPa = ca_logP.value();
      current.walkers[iw][i_fr + 1][0].logP =
          ca_logP.value() + ca_logL1.value();
      current.walkers[iw][i_fr + 1][0].logL =
          ca_logL2.value() - ca_logL1.value();
    }
  }
}

template <class Observer, class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
void last_step_stretch_cuevi_mcmc(
    cuevi_mcmc<Parameters> &current, Observer &obs,
    ensemble<std::mt19937_64> &mt,
    std::vector<std::uniform_real_distribution<double>> &rdist,
    Model const &model, const by_fraction<DataType> &y,
    const by_fraction<Variables> &x, std::size_t n_par, std::size_t i,
    std::size_t iw, std::size_t jw, std::size_t ib, std::size_t i_fr) {

  auto z = std::pow(rdist[i](mt[i]) + 1, 2) / 2.0;
  auto r = rdist[i](mt[i]);

  auto ca_par = stretch_move(current.walkers[iw][i_fr][ib].parameter,
                             current.walkers[jw][i_fr][ib].parameter, z);
  auto ca_logP = logPrior(model, ca_par);
  auto ca_logL0 = logLikelihood(model, ca_par, y[i_fr - 1], x[i_fr - 1]);
  auto ca_logL1 = logLikelihood(model, ca_par, y[i_fr], x[i_fr]);

  if ((ca_logP) && (ca_logL0) && (ca_logL1)) {
    auto dthLogL =
        ca_logP.value() + ca_logL0.value() -
        current.walkers[iw][i_fr][ib].logP +
        current.beta[i_fr][ib] * (ca_logL1.value() - ca_logL0.value() -
                                  current.walkers[iw][i_fr][ib].logL);
    auto pJump = std::min(1.0, std::pow(z, n_par - 1) * std::exp(dthLogL));
    observe_step_stretch_thermo_mcmc(
        obs[iw][ib], jw, z, r, current.walkers[iw][ib].parameter,
        current.walkers[jw][ib].parameter, current.walkers[iw][ib].logP,
        ca_logP, current.walkers[iw][ib].logL, ca_logL0, pJump >= r);
    if (pJump >= r) {
      current.walkers[iw][i_fr][ib].parameter = std::move(ca_par);
      current.walkers[iw][i_fr][ib].logPa = ca_logP.value();
      current.walkers[iw][i_fr][ib].logP = ca_logP.value() + ca_logL0.value();
      current.walkers[iw][i_fr][ib].logL = ca_logL1.value() - ca_logL0.value();
    }
  }
}

template <class Observer, class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
void step_stretch_cuevi_mcmc(std::size_t &iter, cuevi_mcmc<Parameters> &current,
                             Observer &obs, ensemble<std::mt19937_64> &mt,
                             Model const &model, const by_fraction<DataType> &y,
                             const by_fraction<Variables> &x,
                             double alpha_stretch = 2) {
  assert(beta.size() == num_betas(current));
  auto n_walkers = num_walkers(current);

  auto n_par = current.mcmc[0].walkers[0][0].parameter.size();

  std::uniform_int_distribution<std::size_t> uniform_walker(0,
                                                            n_walkers / 2 - 1);
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
      for (std::size_t i_fr = 0; i_fr < 1; ++i_fr) {
        for (std::size_t ib = 0; ib + 1 < current.beta[i_fr].size(); ++ib)
          single_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                         n_par, i, iw, jw, ib, i_fr);

        for (std::size_t ib = current.beta[i_fr].size() - 1;
             ib < current.beta[i_fr].size(); ++ib)
          double_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                         n_par, i, iw, jw, ib, i_fr);
      }
      for (std::size_t i_fr = 1; i_fr + 1 < current.walkers.size(); ++i_fr) {
        for (std::size_t ib = 1; ib + 1 < current.walkers[i_fr].size(); ++ib)
          middle_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                         n_par, i, iw, jw, ib, i_fr);

        for (std::size_t ib = current.walkers[i_fr].size() - 1;
             ib < current.walkers[i_fr].size(); ++ib)
          triple_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                         n_par, i, iw, jw, ib, i_fr);
      }
      for (std::size_t i_fr = current.walkers.size() - 1;
           i_fr < current.walkers.size(); ++i_fr) {
        for (std::size_t ib = 1; ib < current.walkers[i_fr].size(); ++ib)
          last_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                       n_par, i, iw, jw, ib, i_fr);
      }
      ++iter;
    }
}

template <class Observer, class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
void thermo_cuevi_jump_mcmc(std::size_t iter, cuevi_mcmc<Parameters> &current,
                            Observer &obs, const by_beta<double> &beta,
                            std::mt19937_64 &mt, ensemble<std::mt19937_64> &mts,
                            Model const &model, const by_fraction<DataType> &y,
                            const by_fraction<Variables> &x,
                            std::size_t thermo_jumps_every) {
  if (iter % (thermo_jumps_every) == 0) {
    std::uniform_real_distribution<double> uniform_real(0, 1);
    auto n_walkers = mts.size() * 2;
    auto n_beta = beta.size();
    auto n_par = current.walkers[0][0].parameter.size();
    std::uniform_int_distribution<std::size_t> booldist(0, 1);
    auto half = booldist(mt) == 1;

    WalkerIndexes landing_walker(n_walkers / 2);
    std::iota(landing_walker.begin(), landing_walker.end(), 0);
    std::shuffle(landing_walker.begin(), landing_walker.end(), mt);
    std::vector<std::uniform_real_distribution<double>> rdist(n_walkers,
                                                              uniform_real);

#pragma omp parallel for
    for (std::size_t i = 0; i < n_walkers / 2; ++i) {
      auto iw = half ? i + n_walkers / 2 : i;
      auto j = landing_walker[i];
      auto jw = half ? j : j + n_walkers / 2;

      for (std::size_t i_fr = 0; i_fr < 1; ++i_fr) {
        for (std::size_t ib = 0; ib < current.beta[i_fr].size() - 2; ++ib) {

          auto r = rdist[i](mts[i]);
          double logA =
              calc_logA(current.beta[i_fr][ib], current.beta[i_fr][ib + 1],
                        current.walkers[iw][i_fr][ib].logL,
                        current.walkers[jw][i_fr][ib + 1].logL);
          auto pJump = std::min(1.0, std::exp(logA));
          observe_thermo_jump_mcmc(
              obs[iw][ib], jw, current.walkers[iw][i_fr][ib].parameter,
              current.walkers[jw][i_fr][ib + 1].parameter,
              current.walkers[iw][i_fr][ib].logL,
              current.walkers[jw][i_fr][ib + 1].logL,
              -(current.beta[i_fr][ib] - current.beta[i_fr][ib + 1]), logA,
              pJump, r, pJump > r);
          if (pJump > r) {
            std::swap(current.walkers[iw][i_fr][ib],
                      current.walkers[jw][i_fr][ib + 1]);
            std::swap(current.i_walkers[iw][i_fr][ib],
                      current.i_walkers[jw][i_fr][ib + 1]);
          }
        }
        for (std::size_t ib = n_beta - 1; ib < current.beta[i_fr].size();
             ++ib) {

          auto r = rdist[i](mts[i]);
          double logA =
              calc_logA(current.beta[i_fr][ib], current.beta[i_fr][ib + 1],
                        current.walkers[iw][i_fr][ib].logL,
                        current.walkers[jw][i_fr][ib + 1].logL);
          auto pJump = std::min(1.0, std::exp(logA));
          observe_thermo_jump_mcmc(
              obs[iw][ib], jw, current.walkers[iw][i_fr][ib].parameter,
              current.walkers[jw][i_fr][ib + 1].parameter,
              current.walkers[iw][i_fr][ib].logL,
              current.walkers[jw][i_fr][ib + 1].logL,
              -(current.beta[i_fr][ib] - current.beta[i_fr][ib + 1]), logA,
              pJump, r, pJump > r);
          if (pJump > r) {
            auto ca_par = current.walkers[iw][i_fr][ib].parameter;
            auto ca_logL1 =
                logLikelihood(model, ca_par, y[i_fr + 1], x[i_fr + 1]);
            if (ca_logL1) {
              auto ca_logPa = current.walkers[iw][i_fr][ib].logPa;
              auto ca_logP = current.walkers[iw][i_fr][ib].logP;
              auto ca_logL0 = current.walkers[iw][i_fr][ib].logL;
              std::swap(current.walkers[iw][i_fr][ib],
                        current.walkers[jw][i_fr][ib + 1]);
              std::swap(current.i_walkers[iw][i_fr][ib],
                        current.i_walkers[jw][i_fr][ib + 1]);
              current.walkers[iw][i_fr + 1][0].logPa = ca_logPa;
              current.walkers[iw][i_fr + 1][0].logP = ca_logP + ca_logL0;
              current.walkers[iw][i_fr + 1][0].logL =
                  ca_logL1.value() - ca_logL0 - ca_logP + ca_logPa;
            }
          }
        }
      }
      for (std::size_t i_fr = 1; i_fr + 1 < current.walkers.size(); ++i_fr) {
        if (current.beta[i_fr].size() < 3) {
          for (std::size_t ib = 0; ib + 1 < current.beta[i_fr].size(); ++ib) {

            auto r = rdist[i](mts[i]);
            double logA =
                calc_logA(current.beta[i_fr][ib], current.beta[i_fr][ib + 1],
                          current.walkers[iw][i_fr][ib].logL,
                          current.walkers[jw][i_fr][ib + 1].logL);
            auto pJump = std::min(1.0, std::exp(logA));
            observe_thermo_jump_mcmc(
                obs[iw][ib], jw, current.walkers[iw][i_fr][ib].parameter,
                current.walkers[jw][i_fr][ib + 1].parameter,
                current.walkers[iw][i_fr][ib].logL,
                current.walkers[jw][i_fr][ib + 1].logL,
                -(current.beta[i_fr][ib] - current.beta[i_fr][ib + 1]), logA,
                pJump, r, pJump > r);
            if (pJump > r) {
              auto ca_par_1 = current.walkers[iw][i_fr][ib].parameter;
              auto ca_logL_11 =
                  logLikelihood(model, ca_par_1, y[i_fr + 1], x[i_fr + 1]);
              auto ca_par_0 = current.walkers[jw][i_fr][ib + 1].parameter;
              auto ca_logL_00 =
                  logLikelihood(model, ca_par_0, y[i_fr - 1], x[i_fr - 1]);
              if ((ca_logL_11) && (ca_logL_00)) {
                auto ca_logPa_1 = current.walkers[iw][i_fr][ib].logPa;
                auto ca_logP_1 = current.walkers[iw][i_fr][ib].logP;
                auto ca_logL_1 = current.walkers[iw][i_fr][ib].logL;
                auto ca_logPa_0 = current.walkers[jw][i_fr][ib + 1].logPa;
                auto ca_logP_0 = current.walkers[jw][i_fr][ib + 1].logP;
                auto ca_logL_0 = current.walkers[jw][i_fr][ib + 1].logL;
                std::swap(current.walkers[iw][i_fr][ib],
                          current.walkers[jw][i_fr][ib + 1]);
                std::swap(current.i_walkers[iw][i_fr][ib],
                          current.i_walkers[jw][i_fr][ib + 1]);
                current.walkers[jw][i_fr + 1][0].parameter = ca_par_1;
                current.walkers[jw][i_fr + 1][0].logPa = ca_logPa_1;
                current.walkers[jw][i_fr + 1][0].logP = ca_logP_1 + ca_logL_1;
                current.walkers[jw][i_fr + 1][0].logL =
                    ca_logL_11.value() - ca_logL_1 - ca_logP_1 + ca_logPa_1;
                auto ib0 = current.beta[i_fr - 1].size() - 1;
                current.walkers[iw][i_fr - 1][ib0].parameter = ca_par_0;
                current.walkers[iw][i_fr - 1][ib0].logPa = ca_logPa_0;
                current.walkers[iw][i_fr - 1][ib0].logP =
                    ca_logPa_0 + ca_logL_00;
                current.walkers[iw][i_fr - 1][ib0].logL =
                    ca_logP_0 - ca_logPa_0 - ca_logL_00;
              }
            }
          }
        } else {
          for (std::size_t ib = 0; ib < 1; ++ib) {

            auto r = rdist[i](mts[i]);
            double logA =
                calc_logA(current.beta[i_fr][ib], current.beta[i_fr][ib + 1],
                          current.walkers[iw][i_fr][ib].logL,
                          current.walkers[jw][i_fr][ib + 1].logL);
            auto pJump = std::min(1.0, std::exp(logA));
            observe_thermo_jump_mcmc(
                obs[iw][ib], jw, current.walkers[iw][i_fr][ib].parameter,
                current.walkers[jw][i_fr][ib + 1].parameter,
                current.walkers[iw][i_fr][ib].logL,
                current.walkers[jw][i_fr][ib + 1].logL,
                -(current.beta[i_fr][ib] - current.beta[i_fr][ib + 1]), logA,
                pJump, r, pJump > r);
            if (pJump > r) {

              auto ca_par_0 = current.walkers[jw][i_fr][ib + 1].parameter;
              auto ca_logL_00 =
                  logLikelihood(model, ca_par_0, y[i_fr - 1], x[i_fr - 1]);
              if (ca_logL_00) {
                auto ca_logPa_0 = current.walkers[jw][i_fr][ib + 1].logPa;
                auto ca_logP_0 = current.walkers[jw][i_fr][ib + 1].logP;
                auto ca_logL_0 = current.walkers[jw][i_fr][ib + 1].logL;
                std::swap(current.walkers[iw][i_fr][ib],
                          current.walkers[jw][i_fr][ib + 1]);
                std::swap(current.i_walkers[iw][i_fr][ib],
                          current.i_walkers[jw][i_fr][ib + 1]);
                auto ib0 = current.beta[i_fr - 1].size() - 1;
                current.walkers[iw][i_fr - 1][ib0].parameter = ca_par_0;
                current.walkers[iw][i_fr - 1][ib0].logPa = ca_logPa_0;
                current.walkers[iw][i_fr - 1][ib0].logP =
                    ca_logPa_0 + ca_logL_00;
                current.walkers[iw][i_fr - 1][ib0].logL =
                    ca_logP_0 - ca_logPa_0 - ca_logL_00;
              }
            }
          }
          for (std::size_t ib = 1; ib + 2 < current.beta[i_fr].size(); ++ib) {

            auto r = rdist[i](mts[i]);
            double logA =
                calc_logA(current.beta[i_fr][ib], current.beta[i_fr][ib + 1],
                          current.walkers[iw][i_fr][ib].logL,
                          current.walkers[jw][i_fr][ib + 1].logL);
            auto pJump = std::min(1.0, std::exp(logA));
            observe_thermo_jump_mcmc(
                obs[iw][ib], jw, current.walkers[iw][i_fr][ib].parameter,
                current.walkers[jw][i_fr][ib + 1].parameter,
                current.walkers[iw][i_fr][ib].logL,
                current.walkers[jw][i_fr][ib + 1].logL,
                -(current.beta[i_fr][ib] - current.beta[i_fr][ib + 1]), logA,
                pJump, r, pJump > r);
            if (pJump > r) {
              std::swap(current.walkers[iw][i_fr][ib],
                        current.walkers[jw][i_fr][ib + 1]);
              std::swap(current.i_walkers[iw][i_fr][ib],
                        current.i_walkers[jw][i_fr][ib + 1]);
            }
          }
        }
        for (std::size_t ib = current.beta[i_fr].size() - 2;
             ib + 1 < current.beta[i_fr].size(); ++ib) {

          auto r = rdist[i](mts[i]);
          double logA =
              calc_logA(current.beta[i_fr][ib], current.beta[i_fr][ib + 1],
                        current.walkers[iw][i_fr][ib].logL,
                        current.walkers[jw][i_fr][ib + 1].logL);
          auto pJump = std::min(1.0, std::exp(logA));
          observe_thermo_jump_mcmc(
              obs[iw][ib], jw, current.walkers[iw][i_fr][ib].parameter,
              current.walkers[jw][i_fr][ib + 1].parameter,
              current.walkers[iw][i_fr][ib].logL,
              current.walkers[jw][i_fr][ib + 1].logL,
              -(current.beta[i_fr][ib] - current.beta[i_fr][ib + 1]), logA,
              pJump, r, pJump > r);
          if (pJump > r) {
            auto ca_par_1 = current.walkers[iw][i_fr][ib].parameter;
            auto ca_logL_11 =
                logLikelihood(model, ca_par_1, y[i_fr + 1], x[i_fr + 1]);
            if (ca_logL_11) {
              auto ca_logPa_1 = current.walkers[iw][i_fr][ib].logPa;
              auto ca_logP_1 = current.walkers[iw][i_fr][ib].logP;
              auto ca_logL_1 = current.walkers[iw][i_fr][ib].logL;
              std::swap(current.walkers[iw][i_fr][ib],
                        current.walkers[jw][i_fr][ib + 1]);
              std::swap(current.i_walkers[iw][i_fr][ib],
                        current.i_walkers[jw][i_fr][ib + 1]);
              current.walkers[jw][i_fr + 1][0].parameter = ca_par_1;
              current.walkers[jw][i_fr + 1][0].logPa = ca_logPa_1;
              current.walkers[jw][i_fr + 1][0].logP = ca_logP_1 + ca_logL_1;
              current.walkers[jw][i_fr + 1][0].logL =
                  ca_logL_11.value() - ca_logL_1 - ca_logP_1 + ca_logPa_1;
            }
          }
        }
      }
      for (std::size_t i_fr = current.walkers.size() - 1;
           i_fr < current.walkers.size(); ++i_fr) {
        for (std::size_t ib = 0; ib < 1; ++ib) {

          auto r = rdist[i](mts[i]);
          double logA =
              calc_logA(current.beta[i_fr][ib], current.beta[i_fr][ib + 1],
                        current.walkers[iw][i_fr][ib].logL,
                        current.walkers[jw][i_fr][ib + 1].logL);
          auto pJump = std::min(1.0, std::exp(logA));
          observe_thermo_jump_mcmc(
              obs[iw][ib], jw, current.walkers[iw][i_fr][ib].parameter,
              current.walkers[jw][i_fr][ib + 1].parameter,
              current.walkers[iw][i_fr][ib].logL,
              current.walkers[jw][i_fr][ib + 1].logL,
              -(current.beta[i_fr][ib] - current.beta[i_fr][ib + 1]), logA,
              pJump, r, pJump > r);
          if (pJump > r) {

            auto ca_par_0 = current.walkers[jw][i_fr][ib + 1].parameter;
            auto ca_logL_00 =
                logLikelihood(model, ca_par_0, y[i_fr - 1], x[i_fr - 1]);
            if (ca_logL_00) {
              auto ca_logPa_0 = current.walkers[jw][i_fr][ib + 1].logPa;
              auto ca_logP_0 = current.walkers[jw][i_fr][ib + 1].logP;
              auto ca_logL_0 = current.walkers[jw][i_fr][ib + 1].logL;
              std::swap(current.walkers[iw][i_fr][ib],
                        current.walkers[jw][i_fr][ib + 1]);
              std::swap(current.i_walkers[iw][i_fr][ib],
                        current.i_walkers[jw][i_fr][ib + 1]);
              auto ib0 = current.beta[i_fr - 1].size() - 1;
              current.walkers[iw][i_fr - 1][ib0].parameter = ca_par_0;
              current.walkers[iw][i_fr - 1][ib0].logPa = ca_logPa_0;
              current.walkers[iw][i_fr - 1][ib0].logP = ca_logPa_0 + ca_logL_00;
              current.walkers[iw][i_fr - 1][ib0].logL =
                  ca_logP_0 - ca_logPa_0 - ca_logL_00;
            }
          }
        }
        for (std::size_t ib = 1; ib + 1 < current.beta[i_fr].size(); ++ib) {

          auto r = rdist[i](mts[i]);
          double logA =
              calc_logA(current.beta[i_fr][ib], current.beta[i_fr][ib + 1],
                        current.walkers[iw][i_fr][ib].logL,
                        current.walkers[jw][i_fr][ib + 1].logL);
          auto pJump = std::min(1.0, std::exp(logA));
          observe_thermo_jump_mcmc(
              obs[iw][ib], jw, current.walkers[iw][i_fr][ib].parameter,
              current.walkers[jw][i_fr][ib + 1].parameter,
              current.walkers[iw][i_fr][ib].logL,
              current.walkers[jw][i_fr][ib + 1].logL,
              -(current.beta[i_fr][ib] - current.beta[i_fr][ib + 1]), logA,
              pJump, r, pJump > r);
          if (pJump > r) {
            std::swap(current.walkers[iw][i_fr][ib],
                      current.walkers[jw][i_fr][ib + 1]);
            std::swap(current.i_walkers[iw][i_fr][ib],
                      current.i_walkers[jw][i_fr][ib + 1]);
          }
        }
      }
    }
  }
}

template <class Algorithm, class Model, class Variables, class DataType,
          class Reporter,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_Algorithm_conditions<Algorithm> &&
           is_model<Model, Parameters, Variables, DataType>)

auto cuevi_impl(const Algorithm &alg, Model &model, const DataType &y,
                const Variables &x, Reporter rep,
                std::size_t num_scouts_per_ensemble,
                std::size_t thermo_jumps_every, double n_points_per_decade,
                double stops_at, bool includes_zero, std::size_t initseed) {

  auto a = alg;
  auto mt = init_mt(initseed);
  auto n_walkers = num_scouts_per_ensemble;
  auto mts = init_mts(mt, num_scouts_per_ensemble / 2);
  auto beta = get_beta_list(n_points_per_decade, stops_at, includes_zero);
  auto beta_run = by_beta<double>(beta.rend() - 2, beta.rend());
  auto current = init_thermo_mcmc(n_walkers, beta_run, mts, model, y, x);
  auto n_par = current.walkers[0][0].parameter.size();
  auto mcmc_run = checks_convergence(std::move(a), current);
  report_title(rep, current);

  while (beta_run.size() < beta.size() || !mcmc_run.second) {
    while (!mcmc_run.second) {
      step_stretch_cuevi_mcmc(current, rep, beta_run, mts, model, y, x);
      thermo_jump_mcmc(current, rep, beta_run, mt, mts, thermo_jumps_every);
      report(rep, current);
      mcmc_run = checks_convergence(std::move(mcmc_run.first), current);
    }
    if (beta_run.size() < beta.size()) {
      beta_run.insert(beta_run.begin(), beta[beta_run.size()]);
      current = push_back_new_beta(current, mts, beta_run, model, y, x);
      std::cerr << "\n  beta_run=" << beta_run[0] << "\n";
      mcmc_run = checks_convergence(std::move(mcmc_run.first), current);
    }
  }

  return std::pair(mcmc_run, current);
}

#endif // CUEVI_H
