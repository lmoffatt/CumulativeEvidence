#ifndef PARALLEL_TEMPERING_H
#define PARALLEL_TEMPERING_H
#include "bayesian_linear_regression.h"
#include "mcmc.h"
#include "multivariate_normal_distribution.h"
#include <algorithm>
#include <fstream>

struct observer {
  observer() {}
  auto &operator[](std::size_t) const { return *this; }
};

template <class T> using ensemble = std::vector<T>;
template <class T> using by_beta = std::vector<T>;
template <class T> using by_iteration = std::vector<T>;

auto stretch_move(const Parameters &Xk, const Parameters &Xj, double z) {
  assert((Xj.size() == Xk.size()) && "sum of vector fields of different sizes");
  auto out = Xj;
  for (std::size_t i = 0; i < Xj.size(); ++i)
    out[i] += z * (Xk[i] - Xj[i]);
  return out;
}

auto init_mts(std::mt19937_64 &mt, std::size_t n) {
  std::uniform_int_distribution<typename std::mt19937_64::result_type> useed;
  std::vector<std::mt19937_64> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
    out.emplace_back(useed(mt));
  return out;
}

auto get_beta_list(double n_points_per_decade, double stops_at,
                   bool includes_zero) {
  std::size_t num_beta =
      std::ceil(-std::log10(stops_at) * n_points_per_decade) + 1;

  auto beta_size = num_beta;
  if (includes_zero)
    beta_size = beta_size + 1;

  auto out = std::vector<double>(beta_size, 0.0);
  for (std::size_t i = 0; i < num_beta; ++i)
    out[beta_size - 1 - i] = std::pow(10, -1.0 * i / n_points_per_decade);
  return out;
}

template <class Parameters> struct thermo_mcmc {
  by_beta<double> beta;
  ensemble<by_beta<mcmc<Parameters>>> walkers;
  ensemble<by_beta<std::size_t>> i_walkers;
};

template <template <class> class Thermo_mcmc, class Parameters>
std::size_t num_betas(Thermo_mcmc<Parameters> const &x) {
  return x.beta.size();
}

template <class Parameters>
std::size_t num_Parameters(thermo_mcmc<Parameters> const &x) {
  return x.walkers[0][0].parameter.size();
}

template <class Parameters>
std::size_t num_walkers(thermo_mcmc<Parameters> const &x) {
  return x.walkers.size();
}

template <class Parameters>
std::size_t num_samples(by_iteration<thermo_mcmc<Parameters>> const &series) {
  return series.size();
}

template <class Parameters>
auto mean_logL(thermo_mcmc<Parameters> const &mcmc) {
  auto out = by_beta<double>(num_betas(mcmc), 0);
  auto n_walkers = num_walkers(mcmc);
  for (std::size_t iwalker = 0; iwalker < num_walkers(mcmc); ++iwalker)
    for (std::size_t ibeta = 0; ibeta < num_betas(mcmc); ++ibeta)
      out[ibeta] += mcmc.walkers[iwalker][ibeta].logL / n_walkers;
  return out;
}
template <class Parameters>
auto mean_logP(thermo_mcmc<Parameters> const &mcmc) {
  auto out = by_beta<double>(num_betas(mcmc), 0);
  auto n_walkers = num_walkers(mcmc);
  for (std::size_t iwalker = 0; iwalker < num_walkers(mcmc); ++iwalker)
    for (std::size_t ibeta = 0; ibeta < num_betas(mcmc); ++ibeta)
      out[ibeta] += mcmc.walkers[iwalker][ibeta].logP / n_walkers;
  return out;
}

template <class Parameters>
auto var_logL(thermo_mcmc<Parameters> const &mcmc,
              by_beta<double> const &mean) {
  auto out = by_beta<double>(num_betas(mcmc), 0);
  auto n_walkers = num_walkers(mcmc);
  for (std::size_t iwalker = 0; iwalker < num_walkers(mcmc); ++iwalker)
    for (std::size_t ibeta = 0; ibeta < num_betas(mcmc); ++ibeta)
      out[ibeta] +=
          std::pow(mcmc.walkers[iwalker][ibeta].logL - mean[ibeta], 2) /
          n_walkers;
  return out;
}

template <class Parameters>
auto mean_logL(by_iteration<thermo_mcmc<Parameters>> const &series) {
  auto out = by_beta<double>(num_betas(series[0]), 0);
  auto n_walkers = num_walkers(series[0]);
  auto n_iters = num_samples(series);
  for (std::size_t i = 0; i < num_samples(series); ++i)
    for (std::size_t iwalker = 0; iwalker < num_walkers(series[0]); ++iwalker)
      for (std::size_t ibeta = 0; ibeta < num_betas(series[0]); ++ibeta)
        out[ibeta] +=
            series[i].walkers[iwalker][ibeta].logL / n_iters / n_walkers;
  return out;
}

template <class Parameters>
auto mean_logP(by_iteration<thermo_mcmc<Parameters>> const &series) {
  auto out = by_beta<double>(num_betas(series[0]), 0);
  auto n_walkers = num_walkers(series[0]);
  auto n_iters = num_samples(series);
  for (std::size_t i = 0; i < num_samples(series); ++i)
    for (std::size_t iwalker = 0; iwalker < num_walkers(series[0]); ++iwalker)
      for (std::size_t ibeta = 0; ibeta < num_betas(series[0]); ++ibeta)
        out[ibeta] +=
            series[i].walkers[iwalker][ibeta].logP / n_iters / n_walkers;
  return out;
}

template <class Parameters>
auto var_logL(by_iteration<thermo_mcmc<Parameters>> const &series,
              by_beta<double> const &mean) {
  auto out = by_beta<double>(num_betas(series[0]), 0);
  auto n_walkers = num_walkers(series[0]);
  auto n_iters = num_samples(series);
  for (std::size_t i = 0; i < num_samples(series); ++i)
    for (std::size_t iwalker = 0; iwalker < num_walkers(series[0]); ++iwalker)
      for (std::size_t ibeta = 0; ibeta < num_betas(series[0]); ++ibeta)
        out[ibeta] +=
            std::pow(series[i].walkers[iwalker][ibeta].logL - mean[ibeta], 2) /
            n_iters / n_walkers;
  return out;
}

auto derivative_var_ratio_beta(by_beta<double> const &mean,
                               by_beta<double> const &var,
                               by_beta<double> const &beta) {
  by_beta<double> out(mean.size() - 1);
  for (std::size_t i = 0; i < mean.size() - 1; ++i) {
    auto dL = (mean[i + 1] - mean[i]) / (beta[i + 1] - beta[i]);
    auto dL0 = var[i + 1];
    auto dL1 = var[i];
    out[i] = (var[i + 1] + var[i]) / (2 * (mean[i + 1] - mean[i])) *
             (beta[i + 1] - beta[i]);
  }
  return out;
}

template <class Parameters>
auto derivative_var_ratio(by_beta<double> const &mean,
                          by_beta<double> const &var,
                          thermo_mcmc<Parameters> const &current) {
  return derivative_var_ratio_beta(mean, var, current.beta);
}

template <class Parameters>
auto mean_logL_walker(by_iteration<thermo_mcmc<Parameters>> const &series) {
  auto out = ensemble<by_beta<double>>(
      nun_walkers(series[0]), by_beta<double>(num_betas(series[0]), 0));
  for (std::size_t i = 0; i < num_samples(series); ++i)
    for (std::size_t iwalker = 0; iwalker < num_walkers(series[0]); ++iwalker)

      for (std::size_t ibeta = 0; ibeta < num_beta(series[0]); ++ibeta)
        out[iwalker][ibeta] +=
            series[i].walkers[iwalker][ibeta].logL / num_samples(series);
  return out;
}
double calcEvidence(double b1, double b2, double L1, double L2) {
  return 0.5 * (L2 + L1) * (b2 - b1);
}
double calcEvidence_(double b1, double b2, double L1, double L2) {
  if (b1 == 0)
    return calcEvidence_(b1, b2, L1, L2);
  auto db = b2 - b1;
  auto dL = L2 - L1;

  return L1 * db + dL / (std::log(b2) - std::log(b1)) *
                       (b2 * std::log(b2) - b1 * std::log(b1) - b2 + b1 -
                        std::log(b1) * db);
}

double calcEvidence(double b1, double b2, double L1, double L2, double dL1,
                    double dL2) {
  auto dL = dL2 - dL1;
  auto db = b2 - b1;
  return L1 * b2 + L2 * b1 + dL * b2 * b1 +
         0.5 * (b2 * b2 - b1 * b1) / db * (L2 - L1 + dL * (b1 + b2)) +
         1.0 / 6.0 * (std::pow(b2, 3) - std::pow(b1, 3)) * dL / db;
}

double calcEvidence(double b0, double L0, double dL0) {
  return L0 * b0 - 0.5 * dL0 * b0 * b0;
}
double calcEvidence(double b0, double L0) { return L0 * b0; }

double calculate_Evidence(by_beta<double> const &beta,
                          by_beta<double> const &meanLik) {
  auto nb = beta.size();

  if (beta[0] > beta[nb - 1]) {
    double sum = calcEvidence(beta[nb - 1], meanLik[nb - 1]);
    for (std::size_t i = 1; i < beta.size(); ++i)
      sum += calcEvidence(beta[i], beta[i - 1], meanLik[i], meanLik[i - 1]);
    return sum;
  } else {
    double sum = calcEvidence(beta[0], meanLik[0]);
    for (std::size_t i = 1; i < beta.size(); ++i)
      sum += calcEvidence(beta[i - 1], beta[i], meanLik[i - 1], meanLik[i]);
    return sum;
  }
}

double calculate_Evidence(by_beta<double> const &beta,
                          by_beta<double> const &meanLik,
                          by_beta<double> const &varLik) {
  auto nb = beta.size();
  if (beta[0] > beta[nb - 1]) {
    double sum = calcEvidence(beta[nb - 1], meanLik[nb - 1], varLik[nb - 1]);
    for (std::size_t i = 1; i < beta.size(); ++i)
      sum += calcEvidence(beta[i], beta[i - 1], meanLik[i], meanLik[i - 1],
                          varLik[i], varLik[i - 1]);
    return sum;
  } else {
    double sum = calcEvidence(beta[0], meanLik[0], varLik[0]);
    for (std::size_t i = 1; i < beta.size(); ++i)
      sum += calcEvidence(beta[i-1], beta[i], meanLik[i-1], meanLik[i],
                          varLik[i-1], varLik[i]);
    return sum;
  }
}

template <class Parameters>
auto var_logL_walker(by_iteration<thermo_mcmc<Parameters>> const &series,
                     ensemble<by_beta<double>> const &mean_logL_walker) {
  auto out = ensemble<by_beta<double>>(
      nun_walkers(series[0]), by_beta<double>(num_betas(series[0]), 0));
  for (std::size_t i = 0; i < num_samples(series); ++i)
    for (std::size_t iwalker = 0; iwalker < num_walkers(series[0]); ++iwalker)
      for (std::size_t ibeta = 0; ibeta < num_beta(series[0]); ++ibeta)
        out[iwalker][ibeta] += std::pow(series[i].walkers[iwalker][ibeta].logL -
                                        mean_logL_walker[iwalker][ibeta]) /
                               num_samples(series);
  return out;
}
template <class Parameters>
auto var_mean_logL_walker(ensemble<by_beta<double>> const &mean_logL_walker,
                          by_beta<double> const &mean) {
  auto n_beta = mean_logL_walker[0].size();
  auto n_walkers = mean_logL_walker.size();
  auto out = by_beta<double>(n_beta, 0.0);
  for (std::size_t i = 0; i < n_beta; ++i)
    for (std::size_t j = 0; j < n_walkers; ++j)
      out[i] += std::pow(mean_logL_walker[j][i] - mean[i], 2) / n_walkers;
  return out;
}

template <class Parameters>
auto mean_var_logL_walker(ensemble<by_beta<double>> const &var_logL_walker) {
  auto n_beta = var_logL_walker[0].size();
  auto n_walkers = var_logL_walker.size();
  auto out = by_beta<double>(n_beta, 0.0);
  for (std::size_t i = 0; i < n_beta; ++i)
    for (std::size_t j = 0; j < n_walkers; ++j)
      out[i] += var_logL_walker[j][i] / n_walkers;
  return out;
}

template <class Parameters>
auto mixing_var_ratio(by_beta<double> const &mean_var,
                      by_beta<double> const &var_mean) {
  by_beta<double> out(mean_var.size());
  for (std::size_t i = 0; i < mean_var.size(); ++i)
    out[i] = var_mean[i] / mean_var[i];
  return out;
}

template <class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
auto init_thermo_mcmc(std::size_t n_walkers, by_beta<double> const &beta,
                      ensemble<std::mt19937_64> &mt, Model &model,
                      const DataType &y, const Variables &x) {

  ensemble<by_beta<std::size_t>> i_walker(n_walkers,
                                          by_beta<std::size_t>(beta.size()));
  ensemble<by_beta<mcmc<Parameters>>> walker(
      n_walkers, by_beta<mcmc<Parameters>>(beta.size()));

  for (std::size_t half = 0; half < 2; ++half)
#pragma omp parallel for
    for (std::size_t iiw = 0; iiw < n_walkers / 2; ++iiw) {
      auto iw = iiw + half * n_walkers / 2;
      for (std::size_t i = 0; i < beta.size(); ++i) {
        i_walker[iw][i] = iw + i * n_walkers;
        walker[iw][i] = init_mcmc(mt[iiw], model, y, x);
      }
    }
  return thermo_mcmc{beta, walker, i_walker};
}

template <class Parameters>
std::pair<std::pair<std::size_t, std::size_t>, bool>
check_iterations(std::pair<std::size_t, std::size_t> current_max,
                 const thermo_mcmc<Parameters> &) {
  if (current_max.first >= current_max.second)
    return std::pair(std::pair(0ul, current_max.second), true);
  else
    return std::pair(std::pair(current_max.first + 1, current_max.second),
                     false);
};

template <class Algorithm, template <class> class Thermo_mcmc>
concept is_Algorithm_conditions = requires(Algorithm &&a) {
  {
    checks_convergence(std::move(a),
                       std::declval<const Thermo_mcmc<Parameters> &>())
  } -> std::convertible_to<std::pair<Algorithm, bool>>;
};

class less_than_max_iteration {
  std::size_t current_iteration_;
  std::size_t max_iter_;
  std::size_t num_betas_;

public:
  less_than_max_iteration(std::size_t max_iter)
      : current_iteration_{0ul}, max_iter_{max_iter}, num_betas_{1ul} {}

  less_than_max_iteration &operator++() {
    ++current_iteration_;
    return *this;
  }
  std::size_t current_iteration() const { return current_iteration_; }
  std::size_t max_iteration() const { return max_iter_; }
  std::size_t betas_number() const { return num_betas_; }

  bool stop() const { return current_iteration() >= max_iteration(); }
  template <class Anything>
  friend auto checks_convergence(less_than_max_iteration &&c,
                                 const Anything &mcmc) {

    if (c.betas_number() < num_betas(mcmc)) {
      c.current_iteration_ = 0;
      c.num_betas_ = num_betas(mcmc);
    }
    if (c.stop()) {
      return std::pair(std::move(c), true);

    } else {
      ++c;
      return std::pair(std::move(c), false);
    }
  }
};
static_assert(is_Algorithm_conditions<less_than_max_iteration, thermo_mcmc>);

template <class Beta, class Var_ratio>
bool compare_to_max_ratio(Beta const &beta, Var_ratio const &mean_logL,
                          Var_ratio const &var_ratio, double max_ratio);

bool compare_to_max_ratio(by_beta<double> const &beta,
                          by_beta<double> const &mean_logL,
                          by_beta<double> const &var_ratio, double max_ratio) {
  for (std::size_t i = 0; i < var_ratio.size(); ++i) {
    std::cerr << "(" << beta[i] << "[~" << mean_logL[i] << "]=> "
              << var_ratio[i] << ")  ";
    if (var_ratio[i] > max_ratio) {
      std::cerr << "  FALSE \n";
      return false;
    }
  }
  std::cerr << " TRUE\n";
  return true;
}

template <template <class> class Thermo_mcmc>
class checks_derivative_var_ratio {
  std::size_t current_iteration_;
  double max_ratio_;
  by_iteration<Thermo_mcmc<Parameters>> curr_samples_;

public:
  checks_derivative_var_ratio(std::size_t sample_size, double max_ratio = 2)
      : current_iteration_(0ul), max_ratio_{max_ratio},
        curr_samples_{sample_size} {}

  checks_derivative_var_ratio &add(Thermo_mcmc<Parameters> const &x) {
    curr_samples_[current_iteration_ % curr_samples_.size()] = x;
    ++current_iteration_;
    return *this;
  }
  auto &current_samples() const { return curr_samples_; }

  auto get_derivative_var_ratio() const {
    auto m = mean_logL(curr_samples_);
    auto var = var_logL(current_samples(), m);
    return std::tuple(m, derivative_var_ratio(m, var, curr_samples_[0]));
  }

  bool converges() const {
    if (current_iteration_ % current_samples().size() == 0) {
      auto [meanL, var_ratio] = get_derivative_var_ratio();
      return compare_to_max_ratio(curr_samples_[0].beta, meanL, var_ratio,
                                  max_ratio_);
    } else {
      return false;
    }
  }
  friend auto checks_convergence(checks_derivative_var_ratio &&c,
                                 const Thermo_mcmc<Parameters> &mcmc) {
    if ((c.current_iteration_ > 0) &&
        (num_betas(c.current_samples()[0]) < num_betas(mcmc))) {
      c.current_iteration_ = 0;
    }
    c.add(mcmc);
    if (c.converges()) {
      return std::pair(std::move(c), true);
    } else {
      return std::pair(std::move(c), false);
    }
  }
};
static_assert(is_Algorithm_conditions<checks_derivative_var_ratio<thermo_mcmc>,
                                      thermo_mcmc>);

template <class Observer, class Parameters>
void observe_step_stretch_thermo_mcmc(
    Observer &obs, std::size_t j, double z, double r, const Parameters &current,
    const Parameters &candidate, double currlogP,
    Maybe_error<double> const calogP, double culogL,
    Maybe_error<double> const &calogL, bool doesChange) {}

template <class Observer, class Parameters>
void observe_thermo_jump_mcmc(Observer &obs, std::size_t jlanding,
                              const Parameters &current,
                              const Parameters &candidate, double culogL,
                              double calogL, double deltabeta, double logA,
                              double pJump, double r, bool doesChange) {}

template <class Observer, class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
void step_stretch_thermo_mcmc(std::size_t &iter,
                              thermo_mcmc<Parameters> &current, Observer &obs,
                              const by_beta<double> &beta,
                              ensemble<std::mt19937_64> &mt, Model const &model,
                              const DataType &y, const Variables &x,
                              double alpha_stretch = 2) {
  assert(beta.size() == num_betas(current));
  auto n_walkers = num_walkers(current);
  auto n_beta = beta.size();
  auto n_par = current.walkers[0][0].parameter.size();

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
      for (std::size_t ib = 0; ib < n_beta; ++ib) {
        // we can try in the outer loop

        auto z = std::pow(rdist[i](mt[i]) + 1, 2) / 2.0;
        auto r = rdist[i](mt[i]);

        // candidate[ib].walkers[iw].
        auto ca_par = stretch_move(current.walkers[iw][ib].parameter,
                                   current.walkers[jw][ib].parameter, z);
        auto ca_logP = logPrior(model, ca_par);
        auto ca_logL = logLikelihood(model, ca_par, y, x);

        if ((ca_logP) && (ca_logL)) {
          auto dthLogL =
              ca_logP.value() - current.walkers[iw][ib].logP +
              beta[ib] * (ca_logL.value() - current.walkers[iw][ib].logL);
          auto pJump =
              std::min(1.0, std::pow(z, n_par - 1) * std::exp(dthLogL));
          observe_step_stretch_thermo_mcmc(
              obs[iw][ib], jw, z, r, current.walkers[iw][ib].parameter,
              current.walkers[jw][ib].parameter, current.walkers[iw][ib].logP,
              ca_logP, current.walkers[iw][ib].logL, ca_logL, pJump >= r);
          if (pJump >= r) {
            current.walkers[iw][ib].parameter = std::move(ca_par);
            current.walkers[iw][ib].logP = ca_logP.value();
            current.walkers[iw][ib].logL = ca_logL.value();
          }
        }
      }
    }
  ++iter;
}

double calc_logA(double betai, double betaj, double logLi, double logLj) {
  return -(betai - betaj) * (logLi - logLj);
}

template <class Parameters, class Observer>
void thermo_jump_mcmc(std::size_t iter, thermo_mcmc<Parameters> &current,
                      Observer &obs, const by_beta<double> &beta,
                      std::mt19937_64 &mt, ensemble<std::mt19937_64> &mts,
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
      for (std::size_t ib = 0; ib < n_beta - 1; ++ib) {

        auto r = rdist[i](mts[i]);
        double logA =
            calc_logA(beta[ib], beta[ib + 1], current.walkers[iw][ib].logL,
                      current.walkers[jw][ib + 1].logL);
        auto pJump = std::min(1.0, std::exp(logA));
        observe_thermo_jump_mcmc(
            obs[iw][ib], jw, current.walkers[iw][ib].parameter,
            current.walkers[jw][ib + 1].parameter, current.walkers[iw][ib].logL,
            current.walkers[jw][ib + 1].logL, -(beta[ib] - beta[ib + 1]), logA,
            pJump, r, pJump > r);
        if (pJump > r) {
          std::swap(current.walkers[iw][ib], current.walkers[jw][ib + 1]);
          std::swap(current.i_walkers[iw][ib], current.i_walkers[jw][ib + 1]);
        }
      }
    }
  }
}

template <class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
auto push_back_new_beta(std::size_t &iter, thermo_mcmc<Parameters> &current,
                        ensemble<std::mt19937_64> &mts,
                        by_beta<double> const &new_beta, Model &model,
                        const DataType &y, const Variables &x) {
  auto n_walkers = current.walkers.size();
  auto n_beta_old = current.walkers[0].size();
  for (std::size_t half = 0; half < 2; ++half)
    for (std::size_t i = 0; i < n_walkers / 2; ++i) {
      auto iw = i + half * n_walkers / 2;
      current.walkers[iw].push_back(init_mcmc(mts[i], model, y, x));
      current.i_walkers[iw].push_back(n_beta_old * n_walkers + iw);
    }
  iter = 0;
  current.beta = new_beta;
  return current;
}

class save_likelihood {

public:
  std::string sep = ",";
  std::ofstream f;
  std::size_t save_every = 1;
  save_likelihood(std::string const &path, std::size_t interval)
      : f{std::ofstream(path + "__i_beta__i_walker.csv")},
        save_every{interval} {}

  friend void report_title(save_likelihood &s,
                           thermo_mcmc<Parameters> const &) {

    s.f << "n_betas" << s.sep << "iter" << s.sep << "beta" << s.sep
        << "i_walker" << s.sep << "id_walker" << s.sep << "logP" << s.sep
        << "logLik"
        << "\n";
  }
  template <class Model, class Variables, class DataType>
  friend void report_model(save_likelihood &,Model &, const DataType &,
                           const Variables &, by_beta<double> const &) {

   }
  friend void report(std::size_t iter, save_likelihood &s,
                     thermo_mcmc<Parameters> const &data) {
    if (iter % s.save_every == 0)
      for (std::size_t i_beta = 0; i_beta < num_betas(data); ++i_beta)
        for (std::size_t i_walker = 0; i_walker < num_walkers(data); ++i_walker)

          s.f << num_betas(data) << s.sep << iter << s.sep << data.beta[i_beta]
              << s.sep << i_walker << s.sep << data.i_walkers[i_walker][i_beta]
              << s.sep << data.walkers[i_walker][i_beta].logP << s.sep
              << data.walkers[i_walker][i_beta].logL << "\n";
  }
};




class save_Evidence {

public:
  std::string sep = ",";
  std::ofstream f;
  std::size_t save_every = 1;
  save_Evidence(std::string const &path, std::size_t interval)
      : f{std::ofstream(path + "__i_iter.csv")}, save_every{interval} {}

  friend void report_title(save_Evidence &s, thermo_mcmc<Parameters> const &) {

    s.f << "n_betas" << s.sep << "iter" << s.sep << "beta" << s.sep
        << "meanPrior" << s.sep << "meanLik" << s.sep << "varLik" << s.sep
        << "Evidence_mean" << s.sep << "Evidence_var"
        << "\n";
  }

  friend void report(std::size_t iter, save_Evidence &s,
                     thermo_mcmc<Parameters> const &data) {
    if (iter % s.save_every == 0) {

      auto meanLik = mean_logL(data);
      auto meanPrior = mean_logP(data);

      auto varLik = var_logL(data, meanLik);
      if (data.beta[0] == 1) {
        auto Evidence2 = calculate_Evidence(data.beta, meanLik, varLik);
        auto Evidence1 = calculate_Evidence(data.beta, meanLik);
        for (std::size_t i_beta = 0; i_beta < num_betas(data); ++i_beta)
          s.f << num_betas(data) << s.sep << iter << s.sep << data.beta[i_beta]
              << s.sep << meanPrior[i_beta] << s.sep << meanLik[i_beta] << s.sep
              << varLik[i_beta] << s.sep << Evidence1 << s.sep << Evidence2
              << "\n";
      }
    }
  }

  template <class Model, class Variables, class DataType>
  friend void report_model(save_Evidence &s,Model &model, const DataType &y,
                           const Variables &x, by_beta<double> const &beta0) {

    auto expected_Evidence= bayesian_linear_regression_calculate_Evidence(model,y,x);

    auto meanLik = bayesian_linear_regression_calculate_mean_logLik(model,y,x,beta0);
    if (meanLik&& expected_Evidence)
    for (std::size_t i_beta = 0; i_beta < size(beta0); ++i_beta)
      s.f<< 0 << s.sep << 0 << s.sep  << beta0[i_beta]<< s.sep
            << 0 << s.sep << meanLik.value()[i_beta] << s.sep << 0 << s.sep
            << expected_Evidence << s.sep << expected_Evidence<< "\n";
  }

};

class save_Parameter {

public:
  std::string sep = ",";
  std::ofstream f;
  std::size_t save_every;
  save_Parameter(std::string const &path, std::size_t interval)
      : f{std::ofstream(path + "__i_beta__i_walker__i_par.csv")},
        save_every{interval} {}

  friend void report_title(save_Parameter &s, thermo_mcmc<Parameters> const &) {

    s.f << "n_betas" << s.sep << "iter" << s.sep << "beta" << s.sep
        << "i_walker" << s.sep << "id_walker" << s.sep << "i_par" << s.sep
        << "par_value"
        << "\n";
  }
  template <class Model, class Variables, class DataType>
  friend void report_model(save_Parameter &,Model &, const DataType &,
                    const Variables &, by_beta<double> const &) {
  }

  friend void report(std::size_t iter, save_Parameter &s,
                     thermo_mcmc<Parameters> const &data) {
    if (iter % s.save_every == 0)
      for (std::size_t i_beta = 0; i_beta < num_betas(data); ++i_beta)
        for (std::size_t i_walker = 0; i_walker < num_walkers(data); ++i_walker)
          for (std::size_t i_par = 0; i_par < num_Parameters(data); ++i_par)

            s.f << num_betas(data) << s.sep << iter << s.sep
                << data.beta[i_beta] << s.sep << i_walker << s.sep
                << data.i_walkers[i_walker][i_beta] << s.sep << i_par << s.sep
                << data.walkers[i_walker][i_beta].parameter[i_par] << "\n";
  }
};

template <class... saving> class save_mcmc : public observer, public saving... {

  std::string directory_;
  std::string filename_prefix_;

public:
  template <typename... Size>
    requires((std::integral<Size> && ...) &&
             (sizeof...(saving) == sizeof...(Size)))
  save_mcmc(std::string dir, std::string filename_prefix,
            Size... sampling_intervals)
      : saving{dir + filename_prefix, sampling_intervals}..., directory_{dir},
        filename_prefix_{filename_prefix} {}

  save_mcmc(std::string dir, std::string filename_prefix)
      : saving{dir + filename_prefix, 1ul}..., directory_{dir},
        filename_prefix_{filename_prefix} {}

  friend void report(std::size_t iter, save_mcmc &f,
                     thermo_mcmc<Parameters> const &data) {
    (report(iter, static_cast<saving &>(f), data), ..., 1);
  }
  template <class Model, class Variables, class DataType>
  friend void report_model(save_mcmc &s,Model &model, const DataType &y,
                     const Variables &x, by_beta<double> const &beta0) {
    (report_model(static_cast<saving &>(s), model,y,x,beta0), ..., 1);
  }
  friend void report_title(save_mcmc &f, thermo_mcmc<Parameters> const &data) {
    (report_title(static_cast<saving &>(f), data), ..., 1);
  }
};

template <class Algorithm, class Model, class Variables, class DataType,
          class Reporter,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_Algorithm_conditions<Algorithm, thermo_mcmc> &&
           is_model<Model, Parameters, Variables, DataType>)

auto thermo_impl(const Algorithm &alg, Model &model, const DataType &y,
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
  std::size_t iter = 0;
  report_title(rep, current);
  report_model(rep,model,y,x, beta);

  while (beta_run.size() < beta.size() || !mcmc_run.second) {
    while (!mcmc_run.second) {
      step_stretch_thermo_mcmc(iter, current, rep, beta_run, mts, model, y, x);
      thermo_jump_mcmc(iter, current, rep, beta_run, mt, mts,
                       thermo_jumps_every);
      report(iter, rep, current);
      mcmc_run = checks_convergence(std::move(mcmc_run.first), current);
    }
    if (beta_run.size() < beta.size()) {
      beta_run.insert(beta_run.begin(), beta[beta_run.size()]);
      current = push_back_new_beta(iter, current, mts, beta_run, model, y, x);
      std::cerr << "\n  beta_run=" << beta_run[0] << "\n";
      mcmc_run = checks_convergence(std::move(mcmc_run.first), current);
    }
  }

  return std::pair(mcmc_run, current);
}

template <class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
auto thermo_max_iter(Model model, const DataType &y, const Variables &x,
                     std::string path, std::string filename,
                     std::size_t num_scouts_per_ensemble,
                     std::size_t thermo_jumps_every, std::size_t max_iter,
                     double n_points_per_decade, double stops_at,
                     bool includes_zero, std::size_t initseed) {
  return thermo_impl(less_than_max_iteration(max_iter), model, y, x,
                     save_mcmc<save_likelihood, save_Parameter, save_Evidence>(
                         path, filename, 10ul, 10ul, 10ul),
                     num_scouts_per_ensemble, thermo_jumps_every,
                     n_points_per_decade, stops_at, includes_zero, initseed);
}

template <class Model, class Variables, class DataType,
          class Parameters = std::decay_t<decltype(sample(
              std::declval<std::mt19937_64 &>(), std::declval<Model &>()))>>
  requires(is_model<Model, Parameters, Variables, DataType>)
auto thermo_convergence(Model model, const DataType &y, const Variables &x,
                        std::string path, std::string filename,
                        std::size_t num_scouts_per_ensemble,
                        std::size_t thermo_jumps_every, std::size_t max_iter,
                        double n_points_per_decade, double stops_at,
                        bool includes_zero, std::size_t initseed) {
  return thermo_impl(
      checks_derivative_var_ratio<thermo_mcmc>(max_iter * model.size()), model,
      y, x,
      save_mcmc<save_likelihood, save_Parameter, save_Evidence>(path, filename, 10ul, 100ul,10ul),
      num_scouts_per_ensemble, thermo_jumps_every, n_points_per_decade,
      stops_at, includes_zero, initseed);
}

#endif // PARALLEL_TEMPERING_H
