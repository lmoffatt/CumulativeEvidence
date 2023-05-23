#ifndef CUEVI_H
#define CUEVI_H
#include "mcmc.h"
#include "parallel_tempering.h"
#include "random_samplers.h"
#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

template <class T> using by_fraction = std::vector<T>;

template <class Parameters> struct mcmc2 : public mcmc<Parameters> {
  double logPa;
};

template <class Parameters> struct cuevi_mcmc {
  by_fraction<std::size_t> nsamples;
  by_fraction<by_beta<double>> beta;
  ensemble<by_fraction<by_beta<mcmc2<Parameters>>>> walkers;
  ensemble<by_fraction<by_beta<std::size_t>>> i_walkers;
  friend void report_title(save_likelihood &s, cuevi_mcmc<Parameters> const &) {

    s.f << "n_fractions" << s.sep << "n_betas" << s.sep << "iter" << s.sep
        << "nsamples" << s.sep << "beta" << s.sep << "i_walker" << s.sep
        << "id_walker" << s.sep << "logP" << s.sep << "logLik"
        << "\n";
  }
  friend void report_title(save_Evidence &s, cuevi_mcmc<Parameters> const &) {

    s.f << "n_fractions" << s.sep << "n_betas" << s.sep << "iter" << s.sep
        << "beta" << s.sep << "meanPrior" << s.sep << "meanLik" << s.sep
        << "varLik" << s.sep << "Evidence_mean" << s.sep << "Evidence_var"
        << "\n";
  }

  friend void report_title(save_Parameter &s, cuevi_mcmc<Parameters> const &) {

    s.f << "n_fractions" << s.sep << "n_betas" << s.sep << "iter" << s.sep
        << "beta" << s.sep << "i_walker" << s.sep << "id_walker" << s.sep
        << "i_par" << s.sep << "par_value"
        << "\n";
  }
  template <class... saving>
  friend void report_title(save_mcmc<saving...> &f,
                           cuevi_mcmc<Parameters> const &data) {
    (report_title(static_cast<saving &>(f), data), ..., 1);
  }
};

template <class Parameters> auto mean_logL(cuevi_mcmc<Parameters> const &mcmc) {
  auto out = by_fraction<by_beta<double>>{};
  out.reserve(size(mcmc.beta));
  auto n_walkers = size(mcmc.walkers);
  for (std::size_t i_frac = 0; i_frac < size(mcmc.beta); ++i_frac) {
    out.emplace_back(size(mcmc.beta[i_frac]), 0.0);
    for (std::size_t iwalker = 0; iwalker < n_walkers; ++iwalker)
      for (std::size_t ibeta = 0; ibeta < size(mcmc.beta[i_frac]); ++ibeta)
        out[i_frac][ibeta] +=
            mcmc.walkers[iwalker][i_frac][ibeta].logL / n_walkers;
  }
  return out;
}
template <class Parameters> auto mean_logP(cuevi_mcmc<Parameters> const &mcmc) {
  auto out = by_fraction<by_beta<double>>{};
  out.reserve(size(mcmc.beta));
  auto n_walkers = size(mcmc.walkers);
  for (std::size_t i_frac = 0; i_frac < size(mcmc.beta); ++i_frac) {
    out.emplace_back(size(mcmc.beta[i_frac]), 0.0);
    for (std::size_t iwalker = 0; iwalker < n_walkers; ++iwalker)
      for (std::size_t ibeta = 0; ibeta < size(mcmc.beta[i_frac]); ++ibeta)
        out[i_frac][ibeta] +=
            mcmc.walkers[iwalker][i_frac][ibeta].logP / n_walkers;
  }
  return out;
}

template <class Parameters>
auto var_logL(cuevi_mcmc<Parameters> const &mcmc,
              by_fraction<by_beta<double>> const &mean) {
  auto out = by_fraction<by_beta<double>>{};
  out.reserve(size(mcmc.beta));
  auto n_walkers = size(mcmc.walkers);
  for (std::size_t i_frac = 0; i_frac < size(mcmc.beta); ++i_frac) {
    out.emplace_back(size(mcmc.beta[i_frac]), 0.0);
    for (std::size_t iwalker = 0; iwalker < n_walkers; ++iwalker)
      for (std::size_t ibeta = 0; ibeta < size(mcmc.beta[i_frac]); ++ibeta)
        out[i_frac][ibeta] +=
            std::pow(mcmc.walkers[iwalker][i_frac][ibeta].logL -
                         mean[i_frac][ibeta],
                     2) /
            n_walkers;
  }
  return out;
}

template <class Parameters>
auto mean_logL(by_iteration<cuevi_mcmc<Parameters>> const &series) {
  auto const &mcmc = series[0];
  auto out = by_fraction<by_beta<double>>(size(mcmc.beta));
  auto n_walkers = size(mcmc.walkers);
  auto n_iters = size(series);
  for (std::size_t i = 0; i < size(series); ++i)
    for (std::size_t i_frac = 0; i_frac < size(mcmc.beta); ++i_frac) {

      out[i_frac] = by_beta<double>(size(mcmc.beta[i_frac]), 0.0);
      for (std::size_t iwalker = 0; iwalker < size(mcmc.walkers); ++iwalker)
        for (std::size_t ibeta = 0; ibeta < size(mcmc.beta[i_frac]); ++ibeta)
          out[i_frac][ibeta] += series[i].walkers[iwalker][i_frac][ibeta].logL /
                                n_iters / n_walkers;
    }
  return out;
}

template <class Parameters>
auto mean_logP(by_iteration<cuevi_mcmc<Parameters>> const &series) {
  auto const &mcmc = series[0];
  auto out = by_fraction<by_beta<double>>(size(mcmc.beta));
  auto n_walkers = size(mcmc.walkers);
  auto n_iters = size(series);
  for (std::size_t i = 0; i < size(series); ++i)
    for (std::size_t i_frac = 0; i_frac < size(mcmc.beta); ++i_frac) {

      out[i_frac] = by_beta<double>(size(mcmc.beta[i_frac]), 0.0);
      for (std::size_t iwalker = 0; iwalker < size(mcmc.walkers); ++iwalker)
        for (std::size_t ibeta = 0; ibeta < size(mcmc.beta[i_frac]); ++ibeta)
          out[i_frac][ibeta] += series[i].walkers[iwalker][i_frac][ibeta].logP /
                                n_iters / n_walkers;
    }
  return out;
}

template <class Parameters>
auto var_logL(by_iteration<cuevi_mcmc<Parameters>> const &series,
              by_fraction<by_beta<double>> const &mean) {
  auto const &mcmc = series[0];
  auto out = by_fraction<by_beta<double>>(size(mcmc.beta));
  auto n_walkers = size(mcmc.walkers);
  auto n_iters = size(series);
  for (std::size_t i = 0; i < size(series); ++i)
    for (std::size_t i_frac = 0; i_frac < size(mcmc.beta); ++i_frac) {

      out[i_frac] = by_beta<double>(size(mcmc.beta[i_frac]), 0.0);
      for (std::size_t iwalker = 0; iwalker < size(series[0].beta); ++iwalker)
        for (std::size_t ibeta = 0; ibeta < size(series[0].beta[i_frac]);
             ++ibeta)
          out[i_frac][ibeta] +=
              std::pow(series[i].walkers[iwalker][i_frac][ibeta].logL -
                           mean[i_frac][ibeta],
                       2) /
              n_iters / n_walkers;
    }
  return out;
}

by_fraction<double>
calculate_Evidence(by_fraction<by_beta<double>> const &beta,
                   by_fraction<by_beta<double>> const &meanLik) {
  auto nfraction = beta.size();
  auto out = by_fraction<double>(nfraction, 0.0);
  for (std::size_t i_frac = 0; i_frac < nfraction; ++i_frac) {
    out[i_frac] = calculate_Evidence(beta[i_frac], meanLik[i_frac]);
  }
  return out;
}

by_fraction<double>
calculate_Evidence(by_fraction<by_beta<double>> const &beta,
                   by_fraction<by_beta<double>> const &meanLik,
                   by_fraction<by_beta<double>> const &varLik) {
  auto nfraction = beta.size();
  auto out = by_fraction<double>(nfraction, 0.0);
  for (std::size_t i_frac = 0; i_frac < nfraction; ++i_frac) {
    out[i_frac] =
        calculate_Evidence(beta[i_frac], meanLik[i_frac], varLik[i_frac]);
  }
  return out;
}
void report(std::size_t iter, save_likelihood &s,
            cuevi_mcmc<Parameters> const &data) {
  if (iter % s.save_every == 0)
    for (std::size_t i_frac = 0; i_frac < size(data.beta); ++i_frac)
      for (std::size_t i_beta = 0; i_beta < size(data.beta[i_frac]); ++i_beta)
        for (std::size_t i_walker = 0; i_walker < size(data.walkers);
             ++i_walker)

          s.f << size(data.beta) << s.sep << size(data.beta[i_frac]) << s.sep
              << iter << s.sep << data.nsamples[i_frac] << s.sep
              << data.beta[i_frac][i_beta] << s.sep << i_walker << s.sep
              << data.i_walkers[i_walker][i_frac][i_beta] << s.sep
              << data.walkers[i_walker][i_frac][i_beta].logP << s.sep
              << data.walkers[i_walker][i_frac][i_beta].logL << "\n";
}

void report(std::size_t iter, save_Parameter &s,
            cuevi_mcmc<Parameters> const &data) {
  if (iter % s.save_every == 0)
    for (std::size_t i_frac = 0; i_frac < size(data.beta); ++i_frac)
      for (std::size_t i_beta = 0; i_beta < size(data.beta[i_frac]); ++i_beta)
        for (std::size_t i_walker = 0; i_walker < size(data.walkers);
             ++i_walker)
          for (std::size_t i_par = 0;
               i_par < size(data.walkers[i_walker][i_frac][i_beta].parameter);
               ++i_par)

            s.f << size(data.beta) << s.sep << size(data.beta[i_frac]) << s.sep
                << iter << s.sep << data.nsamples[i_frac] << s.sep
                << data.beta[i_frac][i_beta] << s.sep
                << data.i_walkers[i_walker][i_frac][i_beta] << s.sep << i_par
                << s.sep
                << data.walkers[i_walker][i_frac][i_beta].parameter[i_par]
                << "\n";
}

void report(std::size_t iter, save_Evidence &s,
            cuevi_mcmc<Parameters> const &data) {
  if (iter % s.save_every == 0) {

    auto meanLik = mean_logL(data);
    auto meanPrior = mean_logP(data);
    auto varLik = var_logL(data, meanLik);
    auto Evidence2 = calculate_Evidence(data.beta, meanLik, varLik);
    auto Evidence1 = calculate_Evidence(data.beta, meanLik);
    for (std::size_t i_frac = 0; i_frac < size(data.beta); ++i_frac)
      for (std::size_t i_beta = 0; i_beta < size(data.beta[i_frac]); ++i_beta)
        if (data.beta[i_frac][0] == 1) {
          s.f << size(data.beta) << s.sep << size(data.beta[i_frac]) << s.sep
              << iter << s.sep << data.nsamples[i_frac] << s.sep
              << data.beta[i_frac][i_beta] << s.sep << meanPrior[i_frac][i_beta]
              << s.sep << meanLik[i_frac][i_beta] << s.sep
              << varLik[i_frac][i_beta] << s.sep << Evidence1[i_frac] << s.sep
              << Evidence2[i_frac] << "\n";
        }
  }
}

template <class Parameters, class... saving>
void report(std::size_t iter, save_mcmc<saving...> &f,
            cuevi_mcmc<Parameters> const &data) {
  (report(iter, static_cast<saving &>(f), data), ..., 1);
}

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
        current.walkers[jw][i_fr][ib].parameter,
        current.walkers[iw][i_fr][ib].logP, ca_logP,
        current.walkers[iw][i_fr][ib].logL, ca_logL, pJump >= r);
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
        current.walkers[jw][i_fr][ib].parameter,
        current.walkers[iw][i_fr][ib].logP, ca_logP,
        current.walkers[iw][i_fr][ib].logL, ca_logL1, pJump >= r);
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
        obs[iw][ib], jw, z, r, current.walkers[iw][i_fr][ib].parameter,
        current.walkers[jw][i_fr][ib].parameter,
        current.walkers[iw][i_fr][ib].logP, ca_logP,
        current.walkers[iw][i_fr][ib].logL, ca_logL0, pJump >= r);
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
        obs[iw][ib], jw, z, r, current.walkers[iw][i_fr][ib].parameter,
        current.walkers[jw][i_fr][ib].parameter,
        current.walkers[iw][i_fr][ib].logP, ca_logP,
        current.walkers[iw][i_fr][ib].logL, ca_logL0, pJump >= r);
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
        obs[iw][ib], jw, z, r, current.walkers[iw][i_fr][ib].parameter,
        current.walkers[jw][i_fr][ib].parameter,
        current.walkers[iw][i_fr][ib].logP, ca_logP,
        current.walkers[iw][i_fr][ib].logL, ca_logL0, pJump >= r);
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
void step_stretch_cuevi_mcmc(cuevi_mcmc<Parameters> &current, Observer &obs,
                             ensemble<std::mt19937_64> &mt, Model const &model,
                             const by_fraction<DataType> &y,
                             const by_fraction<Variables> &x,
                             double alpha_stretch = 2) {
  assert(current.beta.size() == num_betas(current));
  auto n_walkers = size(current.walkers);

  auto n_par = size(current.walkers[0][0][0].parameter);

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
      if (size(current.beta) == 1) {
        for (std::size_t i_fr = 0; i_fr < 1; ++i_fr) {
          for (std::size_t ib = 0; ib < current.beta[i_fr].size(); ++ib)
            single_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                           n_par, i, iw, jw, ib, i_fr);
        }
      } else {
        for (std::size_t i_fr = 0; i_fr < 1; ++i_fr) {
          for (std::size_t ib = 0; ib + 1 < current.beta[i_fr].size(); ++ib)
            single_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                           n_par, i, iw, jw, ib, i_fr);

          for (std::size_t ib = current.beta[i_fr].size() - 1;
               ib < current.beta[i_fr].size(); ++ib)
            double_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                           n_par, i, iw, jw, ib, i_fr);
        }
        for (std::size_t i_fr = 1; i_fr + 1 < current.beta.size(); ++i_fr) {
          for (std::size_t ib = 1; ib + 1 < current.beta[i_fr].size(); ++ib)
            middle_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                           n_par, i, iw, jw, ib, i_fr);

          for (std::size_t ib = current.beta[i_fr].size() - 1;
               ib < current.beta[i_fr].size(); ++ib)
            triple_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                           n_par, i, iw, jw, ib, i_fr);
        }
        for (std::size_t i_fr = current.beta.size() - 1;
             i_fr < current.beta.size(); ++i_fr) {
          for (std::size_t ib = 1; ib < current.walkers[i_fr].size(); ++ib)
            last_step_stretch_cuevi_mcmc(current, obs, mt, rdist, model, y, x,
                                         n_par, i, iw, jw, ib, i_fr);
        }
      }
    }
}

using DataIndexes = std::vector<std::size_t>;

auto generate_random_Indexes(std::mt19937_64 &mt, std::size_t num_samples,
                             std::size_t num_parameters,
                             double num_jumps_per_decade) {

  std::size_t n_jumps =
      std::floor(num_jumps_per_decade *
                 (std::log10(num_samples) - std::log10(2 * num_parameters)));
  auto indexsizes = DataIndexes(n_jumps);

  for (std::size_t i = 0; i < n_jumps; ++i)
    indexsizes[i] = num_samples * std::pow(10.0, -(1.0 * (n_jumps - i)) /
                                                     num_jumps_per_decade);
  auto out = std::vector<DataIndexes>(n_jumps);
  auto index = DataIndexes(num_samples);
  std::iota(index.begin(), index.end(), 0u);
  auto it = index.begin();
  std::size_t n = 0;
  for (auto i = 0u; i < n_jumps; ++i) {
    n = indexsizes[i] - n;
    it = randomly_extract_n(mt, it, index.end(), n);
    std::sort(index.begin(), it);
    out[i] = DataIndexes(index.begin(), it);
  }
  return out;
}

struct fractioner {
  auto operator()(const Matrix<double> &y, const Matrix<double> &x,
                  std::mt19937_64 &mt, std::size_t num_parameters,
                  double num_jumps_per_decade, double stops_at,
                  bool includes_zero) {
    assert(y.nrows() == x.nrows());
    std::size_t num_samples = size(y);
    auto indexes = generate_random_Indexes(mt, num_samples, num_parameters,
                                           num_jumps_per_decade);
    auto n_frac = size(indexes) + 1;
    by_fraction<Matrix<double>> y_out(n_frac);
    by_fraction<Matrix<double>> x_out(n_frac);
    for (std::size_t i = 0; i < n_frac - 1; ++i) {
      auto n = size(indexes[i]);
      auto ii = indexes[i];
      Matrix<double> yi(n, 1, false);
      Matrix<double> xi(n, x.ncols(), false);

      for (std::size_t j = 0; j < n; ++j) {
        yi[j] = y[ii[j]];
        for (std::size_t k = 0; k < x.ncols(); ++k)
          xi(j, k) = x(ii[j], k);
      }
      y_out[i] = std::move(yi);
      x_out[i] = std::move(xi);
    }
    x_out[n_frac - 1] = x;
    y_out[n_frac - 1] = y;

    auto beta0 = get_beta_list(num_jumps_per_decade, stops_at, includes_zero);
    by_beta<double> betan = {0, 1};
    by_fraction<by_beta<double>> beta(n_frac, betan);
    beta[0] = std::move(beta0);

    return std::tuple(std::move(y_out), std::move(x_out), std::move(beta));
  }
};

template <class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
auto init_mcmc2(std::mt19937_64 &mt, Model &m, const by_fraction<DataType> &y,
                const by_fraction<Variables> &x) {
  auto par = sample(mt, m);
  auto logP = logPrior(m, par);
  auto logL = logLikelihood(m, par, y[0], x[0]);
  auto logPa = logP;
  while (!(logP) || !(logL)) {
    par = sample(mt, m);
    logP = logPrior(m, par);
    logL = logLikelihood(m, par, y[0], x[0]);
  }
  return mcmc2{mcmc{std::move(par), logP.value(), logL.value()}, logPa.value()};
}

template <class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
auto init_cuevi_mcmc(std::size_t n_walkers, by_beta<double> const &beta,
                     ensemble<std::mt19937_64> &mt, Model &model,
                     const by_fraction<DataType> &y,
                     const by_fraction<Variables> &x) {
  by_fraction<std::size_t> nsamples_out(1, size(y[0]));
  by_fraction<by_beta<double>> beta_out(1, beta);
  ensemble<by_fraction<by_beta<std::size_t>>> i_walker(
      n_walkers,
      by_fraction<by_beta<std::size_t>>(1, by_beta<std::size_t>(beta.size())));
  ensemble<by_fraction<by_beta<mcmc2<Parameters>>>> walker(
      n_walkers, by_fraction<by_beta<mcmc2<Parameters>>>(
                     1, by_beta<mcmc2<Parameters>>(beta.size())));

  for (std::size_t half = 0; half < 2; ++half)
#pragma omp parallel for
    for (std::size_t iiw = 0; iiw < n_walkers / 2; ++iiw) {
      auto iw = iiw + half * n_walkers / 2;
      for (std::size_t i = 0; i < beta.size(); ++i) {
        i_walker[iw][0][i] = iw + i * n_walkers;
        walker[iw][0][i] = init_mcmc2(mt[iiw], model, y, x);
      }
    }
  return cuevi_mcmc<Parameters>{nsamples_out, beta_out, walker, i_walker};
}

template <class Parameters>
std::size_t last_walker(const cuevi_mcmc<Parameters> &c) {
  std::size_t tot = 0;
  for (std::size_t i_frac = 0; i_frac < size(c.walkers[0]); ++i_frac)
    tot += size(c.walkers[0][i_frac]);
  return tot * size(c.walkers);
}

template <class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
Maybe_error<cuevi_mcmc<Parameters>> push_back_new_fraction(
    const cuevi_mcmc<Parameters> &current_old, ensemble<std::mt19937_64> &mts,
    const by_fraction<by_beta<double>> &final_beta, Model &model,
    const by_fraction<DataType> &y, const by_fraction<Variables> &x) {
  auto current = current_old;
  auto n_walkers = current.walkers.size();
  auto n_frac_old = size(current.beta);
  auto i_frac_old = n_frac_old - 1;
  auto sum_walkers = last_walker(current);
  ensemble<mcmc2<Parameters>> new_walkers(n_walkers);
  ensemble<std::size_t> new_i_walkers(n_walkers);
  for (std::size_t half = 0; half < 2; ++half)
    for (std::size_t i = 0; i < n_walkers / 2; ++i) {
      auto iw = i + half * n_walkers / 2;
      new_walkers[iw] = init_mcmc2(mts[i], model, y, x);
      new_i_walkers[iw] = sum_walkers + iw;
    }

  auto beta_first = size(current.beta[0]);

  for (std::size_t half = 0; half < 2; ++half)
    for (std::size_t i = 0; i < n_walkers / 2; ++i) {
      auto iw = i + half * n_walkers / 2;
      for (std::size_t i_b = 0; i_b < beta_first; ++i_b) {
        std::swap(current.walkers[iw][0][i_b], new_walkers[iw]);
        std::swap(current.i_walkers[iw][0][i_b], new_i_walkers[iw]);
      }
    }

  if (beta_first < size(final_beta[0])) {
    for (std::size_t half = 0; half < 2; ++half)
      for (std::size_t i = 0; i < n_walkers / 2; ++i) {
        auto iw = i + half * n_walkers / 2;
        current.walkers[iw][0].push_back(new_walkers[iw]);
        current.i_walkers[iw][0].push_back(new_i_walkers[iw]);
      }
    current.beta[0].push_back(final_beta[0][beta_first]);
    return current;
  } else {
    for (auto i_frac = 1ul; i_frac < n_frac_old; ++i_frac) {
      for (std::size_t i_b = 0; i_b < 1; ++i_b) {
        for (std::size_t half = 0; half < 2; ++half)
          for (std::size_t i = 0; i < n_walkers / 2; ++i) {
            auto iw = i + half * n_walkers / 2;
            std::swap(current.walkers[iw][i_frac][i_b], new_walkers[iw]);
            std::swap(current.i_walkers[iw][i_frac][i_b], new_i_walkers[iw]);
            auto &ca_wa = current.walkers[iw][i_frac - 1].back();
            auto ca_logPa = ca_wa.logPa;
            auto ca_par = ca_wa.parameter;
            auto ca_logP = ca_wa.logPa + ca_wa.logL;
            auto ca_logL1 = logLikelihood(model, ca_par, y[i_frac], x[i_frac]);
            if (!(ca_logL1))
              return ca_logL1.error() + " push back new fraction at walker " +
                     std::to_string(iw) + "of fraction " +
                     std::to_string(i_frac) + " beta 0";
            auto ca_logL = ca_logL1.value() - ca_logP + ca_logPa;
            current.walkers[iw][i_frac][i_b] = {{ca_par, ca_logP, ca_logL},
                                                ca_logPa};
          }
      }

      for (std::size_t i_b = 1; i_b < size(current.beta[i_frac]); ++i_b) {
        for (std::size_t half = 0; half < 2; ++half)
          for (std::size_t i = 0; i < n_walkers / 2; ++i) {
            auto iw = i + half * n_walkers / 2;
            std::swap(current.walkers[iw][i_frac][i_b], new_walkers[iw]);
            std::swap(current.i_walkers[iw][i_frac][i_b], new_i_walkers[iw]);
          }
      }
    }
    auto n_beta_current = size(current.beta[i_frac_old]);
    auto n_beta_current_final = size(final_beta[i_frac_old]);
    auto n_frac_final = size(final_beta);

    if ((n_beta_current < n_beta_current_final) ||
        (n_frac_final == n_frac_old)) {
      for (std::size_t half = 0; half < 2; ++half)
        for (std::size_t i = 0; i < n_walkers / 2; ++i) {
          auto iw = i + half * n_walkers / 2;
          current.walkers[iw][n_frac_old].push_back(new_walkers[iw]);
          current.i_walkers[iw][n_frac_old].push_back(new_i_walkers[iw]);
        }
      current.beta[n_frac_old].push_back(
          final_beta[n_frac_old][n_beta_current]);
      return current;
    } else {
      for (std::size_t half = 0; half < 2; ++half)
        for (std::size_t i = 0; i < n_walkers / 2; ++i) {
          auto iw = i + half * n_walkers / 2;
          current.walkers[iw].push_back(by_beta<mcmc2<Parameters>>(2));
          current.i_walkers[iw].push_back(by_beta<std::size_t>(2));

          auto &ca_wa0 = current.walkers[iw][i_frac_old].back();
          auto ca_logPa0 = ca_wa0.logPa;
          auto ca_par0 = ca_wa0.parameter;
          auto ca_logP0 = ca_wa0.logPa + ca_wa0.logL;
          auto ca_logL10 = logLikelihood(model, ca_par0, y[i_frac_old + 1],
                                         x[i_frac_old + 1]);
          if (!(ca_logL10))
            return ca_logL10.error() + " push back new fraction at walker " +
                   std::to_string(iw) + "of fraction " +
                   std::to_string(i_frac_old + 1) + " beta 0";
          auto ca_logL0 = ca_logL10.value() - ca_logP0 + ca_logPa0;
          current.walkers[iw][i_frac_old + 1][0] = {
              {ca_par0, ca_logP0, ca_logL0}, ca_logPa0};
          current.i_walkers[iw][i_frac_old + 1][0] =
              current.i_walkers[iw][i_frac_old].back();

          auto &ca_wa1 = new_walkers[iw];
          auto ca_logPa1 = ca_wa1.logPa;
          auto ca_par1 = ca_wa1.parameter;
          auto ca_logP1 = ca_wa1.logPa + ca_wa1.logL;
          auto ca_logL11 = logLikelihood(model, ca_par1, y[i_frac_old + 1],
                                         x[i_frac_old + 1]);
          if (!(ca_logL11))
            return ca_logL11.error() + " push back new fraction at walker " +
                   std::to_string(iw) + "of fraction " +
                   std::to_string(i_frac_old + 1) + " beta 0";
          auto ca_logL1 = ca_logL11.value() - ca_logP1 + ca_logPa1;
          current.walkers[iw][i_frac_old + 1][1] = {
              {ca_par1, ca_logP1, ca_logL1}, ca_logPa1};
          current.i_walkers[iw][i_frac_old + 1][1] = new_i_walkers[iw];
        }
      current.beta[i_frac_old + 1].insert(current.beta[i_frac_old + 1].end(),
                                          final_beta[i_frac_old + 1].begin(),
                                          final_beta[i_frac_old + 1].begin() +
                                              2);
      return current;
    }
  }
}

template <class Observer, class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
void thermo_cuevi_jump_mcmc(std::size_t iter, cuevi_mcmc<Parameters> &current,
                            Observer &obs, std::mt19937_64 &mt,
                            ensemble<std::mt19937_64> &mts, Model const &model,
                            const by_fraction<DataType> &y,
                            const by_fraction<Variables> &x,
                            std::size_t thermo_jumps_every) {
  if (iter % (thermo_jumps_every) == 0) {
    std::uniform_real_distribution<double> uniform_real(0, 1);
    auto n_walkers = mts.size() * 2;
    auto n_par = current.walkers[0][0][0].parameter.size();
    std::uniform_int_distribution<std::size_t> booldist(0, 1);
    auto half = booldist(mt) == 1;

    WalkerIndexes landing_walker(n_walkers / 2);
    std::iota(landing_walker.begin(), landing_walker.end(), 0);
    std::shuffle(landing_walker.begin(), landing_walker.end(), mt);
    std::vector<std::uniform_real_distribution<double>> rdist(n_walkers,
                                                              uniform_real);

    // #pragma omp parallel for
    for (std::size_t i = 0; i < n_walkers / 2; ++i) {
      auto iw = half ? i + n_walkers / 2 : i;
      auto j = landing_walker[i];
      auto jw = half ? j : j + n_walkers / 2;

      if (size(current.beta) == 1)
        for (std::size_t i_fr = 0; i_fr < 1; ++i_fr) {
          for (std::size_t ib = 0; ib < current.beta[i_fr].size() - 1; ++ib) {

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
      else {
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
          for (std::size_t ib = current.beta[i_fr].size() - 1;
               ib < current.beta[i_fr].size(); ++ib) {

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
                if (size(current.beta) > 1) {
                  current.walkers[iw][i_fr + 1][0].parameter = ca_par;
                  current.walkers[iw][i_fr + 1][0].logPa = ca_logPa;
                  current.walkers[iw][i_fr + 1][0].logP = ca_logP + ca_logL0;
                  current.walkers[iw][i_fr + 1][0].logL =
                      ca_logL1.value() - ca_logL0 - ca_logP + ca_logPa;
                }
              }
            }
          }
        }
        for (std::size_t i_fr = 1; i_fr + 1 < current.beta.size(); ++i_fr) {
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
                      ca_logPa_0 + ca_logL_00.value();
                  current.walkers[iw][i_fr - 1][ib0].logL =
                      ca_logP_0 - ca_logPa_0 - ca_logL_00.value();
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
                      ca_logPa_0 + ca_logL_00.value();
                  current.walkers[iw][i_fr - 1][ib0].logL =
                      ca_logP_0 - ca_logPa_0 - ca_logL_00.value();
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
        for (std::size_t i_fr = std::max(1ul, current.beta.size() - 1);
             i_fr < current.beta.size(); ++i_fr) {
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
                    ca_logPa_0 + ca_logL_00.value();
                current.walkers[iw][i_fr - 1][ib0].logL =
                    ca_logP_0 - ca_logPa_0 - ca_logL_00.value();
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
}

template <class Parameters>
auto derivative_var_ratio(by_fraction<by_beta<double>> const &mean,
                          by_fraction<by_beta<double>> const &var,
                          cuevi_mcmc<Parameters> const &current) {

  auto &beta = current.beta;
  by_fraction<by_beta<double>> out(size(mean));
  for (std::size_t i = 0; i < mean.size(); ++i) {
    out[i] = derivative_var_ratio_beta(mean[i], var[i], beta[i]);
  }
  return out;
}

template <>
bool compare_to_max_ratio(by_fraction<by_beta<double>> const &beta,
                          by_fraction<by_beta<double>> const &var_ratio,
                          double max_ratio) {
  for (std::size_t i_frac = 0; i_frac < size(var_ratio); ++i_frac) {
    for (std::size_t ib = 0; ib < size(var_ratio[i_frac]); ++ib) {
      std::cerr << "(" << beta[i_frac][ib] << " => " << var_ratio[i_frac][ib]
                << ")  ";
      if (var_ratio[i_frac][ib] > max_ratio) {
        std::cerr << "  FALSE \n";
        return false;
      }
    }
  }
  std::cerr << " TRUE\n";
  return true;
}

template <class Algorithm, class Model, class Variables, class DataType,
          class Fractioner, class Reporter,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_Algorithm_conditions<Algorithm, cuevi_mcmc> &&
           is_model<Model, Parameters, Variables, DataType>)

auto cuevi_impl(const Algorithm &alg, Model &model, const DataType &y,
                const Variables &x, const Fractioner frac, Reporter rep,
                std::size_t num_scouts_per_ensemble,
                std::size_t thermo_jumps_every, double n_points_per_decade,
                double stops_at, bool includes_zero, std::size_t initseed) {

  auto a = alg;
  auto mt = init_mt(initseed);
  auto n_walkers = num_scouts_per_ensemble;
  auto mts = init_mts(mt, num_scouts_per_ensemble / 2);
  auto [ys, xs, beta_final] = fractioner{}(
      y, x, mt, size(model), n_points_per_decade, stops_at, includes_zero);
  auto beta_init =
      by_beta<double>(beta_final[0].rend() - 2, beta_final[0].rend());
  auto current = init_cuevi_mcmc(n_walkers, beta_init, mts, model, ys, xs);
  auto mcmc_run = checks_convergence(std::move(a), current);
  std::size_t iter = 0;
  report_title(rep, current);

  while (size(ys) < size(current.beta) || !mcmc_run.second) {
    while (!mcmc_run.second) {
      step_stretch_cuevi_mcmc(current, rep, mts, model, ys, xs);
      ++iter;
      thermo_cuevi_jump_mcmc(iter, current, rep, mt, mts, model, ys, xs,
                             thermo_jumps_every);
      report(iter, rep, current);
      mcmc_run = checks_convergence(std::move(mcmc_run.first), current);
    }
    if (size(ys) < size(current.beta)) {
      auto is_current =
          push_back_new_fraction(current, mts, beta_final, model, ys, xs);
      while (!(is_current)) {
        step_stretch_cuevi_mcmc(current, rep, mts, model, ys, xs);
        ++iter;
        thermo_cuevi_jump_mcmc(iter, current, rep, mt, mts, model, ys, xs,
                               thermo_jumps_every);
        report(iter, rep, current);
        is_current =
            push_back_new_fraction(current, mts, beta_final, model, ys, xs);
      }
      current = std::move(is_current.value());
      std::cerr << "\n  beta_run=" << current.beta.back() << "\n";
      mcmc_run = checks_convergence(std::move(mcmc_run.first), current);
    }
  }

  return std::pair(mcmc_run, current);
}

template <class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
auto cuevi_convergence(Model model, const DataType &y, const Variables &x,
                       std::string path, std::string filename,
                       std::size_t num_scouts_per_ensemble,
                       std::size_t thermo_jumps_every, std::size_t max_iter,
                       double n_points_per_decade, double stops_at,
                       bool includes_zero, std::size_t initseed) {
  return cuevi_impl(
      checks_derivative_var_ratio<cuevi_mcmc>(max_iter * model.size()), model,
      y, x, fractioner{},
      save_mcmc<save_likelihood, save_Parameter>(path, filename, 1ul, 1ul),
      num_scouts_per_ensemble, thermo_jumps_every, n_points_per_decade,
      stops_at, includes_zero, initseed);
}

#endif // CUEVI_H
