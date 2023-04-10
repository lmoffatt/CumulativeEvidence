#ifndef CUEVI_H
#define CUEVI_H
#include "list.h"
#include "range.h"
#include "vector_field.h"
#include <algorithm>
#include <random>
#include <vector>

typedef std::vector<double> Parameters;
typedef std::vector<double> Data;

template <class T>
concept Variables = requires(T a) {
  { a[int{}] } -> std::regular;
  { a.size() } -> std::convertible_to<std::size_t>;
};

typedef std::vector<std::size_t> Indexes;

using IndexedData = std::pair<Indexes, Data>;

template <class T>
concept sample_Parameters = requires(T a) {
  { a(std::declval<std::mt19937_64 &>()) } -> std::convertible_to<Parameters>;
};

template <class T, class Variable>
concept samples_Data = requires(T a) {
  {
    a(std::declval<std::mt19937_64 &>(), Parameters{}, Variable{})
  } -> std::convertible_to<IndexedData>;
  { Variable{} } -> Variables;
};

template <class T>
concept calculates_PriorProb = requires(T a) {
  { a(Parameters{}) } -> std::convertible_to<double>;
};

template <class T, class Variable>
concept calculates_Likelihood = requires(T a) {
  { a(Parameters{}, IndexedData{}, Variable{}) } -> std::convertible_to<double>;
  { Variable{} } -> Variables;
};

template <class T, class Variable>
concept BayesianModel =
    sample_Parameters<T> && samples_Data<T, Variable> &&
    calculates_PriorProb<T> && calculates_Likelihood<T, Variable> &&
    Variables<Variable> && has_size<T>;

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


template <class L>
requires includes_this<L, m<initseed, std::size_t>>
auto calc_seed(L&& li) {

  if (li[initseed{}]() == 0) {
    std::random_device rd;
    li[initseed{}]() = rd();
  }
  return li;
}

struct mt {
  constexpr static std::string name() { return "mt"; }
};

template <class L>
requires includes_this<L,  m<initseed, std::size_t>>&&
    std::is_same_v<L,std::decay_t<L>>
auto init_mt(L&& li) {
  li=calc_seed(std::move(li));
  return std::move(li)&m(mt{},std::mt19937_64(li[initseed{}]()));
}
struct mts {
  constexpr static std::string name() { return "mts"; }
};


auto init_mt_vector(std::mt19937_64 &mt, std::size_t n) {
  std::uniform_int_distribution<typename std::mt19937_64::result_type> useed;
  std::vector<std::mt19937_64> out;
  for (std::size_t i = 0; i < n; ++i)
    out.emplace_back(useed(mt));
  return out;
}

struct jump_factor {
  constexpr static std::string name() { return "jump_factor"; }
};
struct stops_at {
  constexpr static std::string name() { return "stops_at"; }
};

struct includes_zero {
  constexpr static std::string name() { return "includes_zero"; }
};
struct beta {
  constexpr static std::string name() { return "beta"; }
};

template <class L>
  requires includes_this<L, m<jump_factor, double>> &&
           includes_this<L, m<stops_at, double>> &&
           includes_this<L, m<includes_zero, bool>>
auto get_beta_list(const L &li) {
  std::size_t num_beta =
      std::ceil(std::log(li[stops_at{}]()) / std::log(li[jump_factor{}]())) + 1;

  auto beta_size = num_beta;
  if (li[includes_zero{}]())
    beta_size = beta_size + 1;

  auto out = range(beta_size) |
             [li](std::size_t i) { return std::pow(li[jump_factor{}](), i); };
  return out;
}

template <class T> using ensemble = std::vector<T>;
template <class T> using by_fraction = std::vector<T>;
template <class T> using by_beta = std::vector<T>;

template <class L, Variables Variable>
  requires includes_this<L, m<beta, by_beta<double>>>
auto init_parameters(L &&li, BayesianModel<Variable> auto const &model,
                     by_beta<ensemble<std::mt19937_64>> &mts) {

  auto out = by_beta<ensemble<Parameters>>(mts.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = ensemble<Parameters>(mts[i].size());
    for (std::size_t j = 0; j < out[i].size(); ++j) {
      out[i][j] = model(mts[i][j]);
    }
  }
  return out;
}

struct parameters {
  constexpr static std::string name() { return "parameters"; }
};

struct logPrior {
  constexpr static std::string name() { return "logPrior"; }
};

struct logLik {
  constexpr static std::string name() { return "logLik"; }
};

struct thermo_mcmc {
  by_beta<double> beta;
  by_beta<ensemble<Parameters>> parameter;
  by_beta<ensemble<double>> logP;
  by_beta<ensemble<double>> logL;
};

struct cuevi_mcmc {

  by_fraction<Indexes> ind;

  by_fraction<ensemble<Parameters>> parameter;
  by_beta<ensemble<double>> logP;
  by_beta<ensemble<double>> logL_0;
  by_beta<ensemble<double>> logL_1;
};

template <Variables Variable>
auto init_zero_mcmc(BayesianModel<Variable> auto const &model,
                    const Variable &v, const Data &y,
                    std::size_t number_of_jumps_per_decade,
                    std::size_t number_sub_jumps_per_sample_group,
                    double min_sample, bool use_beta_zero,
                    std::size_t initseed) {}

template <Variables Variable, class L>
  requires includes_this<L, m<jump_factor, double>> &&
           includes_this<L, m<stops_at, double>> &&
           includes_this<L, m<includes_zero, bool>> &&
           includes_this<L, m<initseed, std::size_t>>

auto cuevi_impl(BayesianModel<Variable> auto const &model, const Variable &v,
                const Data &y, const L &li,
                std::size_t number_of_jumps_per_decade,
                std::size_t number_sub_jumps_per_sample_group,
                double min_sample, bool use_beta_zero, std::size_t initseed) {
  auto mt = init_mt(li);

  auto beta = get_beta_list()

      auto indexesList = generate_Indexes(mt, y.size(), model.size(),
                                          number_of_jumps_per_decade)
}

#endif // CUEVI_H
