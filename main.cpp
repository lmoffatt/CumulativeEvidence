#include "bayesian_linear_regression.h"
#include "cuevi.h"
#include "distributions.h"
#include "lapack_headers.h"
#include "matrix.h"
#include "matrix_random.h"
#include "maybe_error.h"
#include "mcmc.h"
#include <iostream>
// using namespace std;

int main() {
  auto initseed = 0;
  auto myseed = calc_seed(initseed);
  myseed = 9762841416869310605ul;
  std::cerr << "myseed=\n" << myseed << "\n";
  auto mt = init_mt(myseed);

  auto npar = 4ul;
  auto nsamples = 500ul;
  auto log10_std_par = 1.0;

  auto mean_mean_par = 0.0;
  auto std_mean_par = 10.0;

  auto mean_b = 0.0;
  auto std_b = 10.0;

  auto prior_error_factor = 1;

  auto par_stds = apply([](auto &x) { return std::pow(10, x); },
                        random_matrix_normal(mt, 1, npar, 0, log10_std_par));
  auto cov_par = random_covariance(mt, npar, par_stds);

  auto mean_par =
      random_matrix_normal(mt, 1, npar, mean_mean_par, std_mean_par);

  auto b = random_matrix_normal(mt, 1, npar, mean_b, std_b);

  std::cerr << "b proposed" << b;

  auto par_dist_ = make_multivariate_normal_distribution(std::move(mean_par),
                                                         std::move(cov_par));

  auto par_dist = std::move(par_dist_.value());
  auto X = par_dist(mt, nsamples);
  double prior_eps_df = 1.0;
  double prior_eps_variance = 1.0;
  auto a_0 = prior_eps_df / 2.0;
  auto b_0 = prior_eps_df * prior_eps_variance / 2.0;

  auto prior_error_distribution = log_inverse_gamma_distribution(a_0, b_0);

  auto logs2 = prior_error_distribution(mt);
  auto s2 = std::exp(logs2);
  auto s = std::sqrt(s2);
  std::cerr << "proposed s = " << s << "\n";
  std::cerr << "proposed s2 = " << s2 << "\n";
  std::cerr << "proposed logs2 = " << logs2 << "\n";

  std::cerr << "beta real = " << logs2 << "\t" << b;

  auto eps = random_matrix_normal(mt, nsamples, 1, 0, s);

  auto y = X * tr(b) + eps;
  std::cerr << "\ny\n" << y;

  auto prior_b = make_multivariate_normal_distribution(
      Matrix<double>(1ul, npar, mean_b),
      IdM<double>(npar) * std_b * std_b * prior_error_factor);

  auto prior = distributions(prior_error_distribution, prior_b.value());
  auto sam = prior(mt);
  //  std::cerr << "prior sample\n" << sam;

  //  std::cerr<<"logLik="<<logLikelihood(linear_model{},sam,y,X);

  static_assert(Multivariate<decltype(prior_b.value())>);

  //  std::cerr << "prior logP =" << prior.logP(sam)<<"\n";

  if (prior_b.valid()) {
    auto reg = bayesian_linear_regression(prior_b.value(), prior_eps_df,
                                          prior_eps_variance, y, X);
    std::cout << "Evidence" << reg << "\n";
  }

  auto linear_model =
      make_bayesian_linear_model(
          prior_eps_df, prior_eps_variance, Matrix<double>(1ul, npar, mean_b),
          IdM<double>(npar) * std_b * std_b * prior_error_factor)
          .value();

  auto p = sample(mt, linear_model);
  static_assert(is_model<decltype(linear_model), Matrix<double>, Matrix<double>,
                         Matrix<double>>);

  std::size_t num_scouts_per_ensemble = 32;
  double n_points_per_decade = 12;
  double n_points_per_decade_fraction=3;
  double stops_at = 1e-6;
  bool includes_zero = true;
  std::size_t max_iter = 5000;
  std::string path = "";
  std::string filename = "A";

  std::size_t thermo_jumps_every = linear_model.size()*1e0;

  std::size_t checks_derivative_every_model_size = 1000;
  double max_ratio=8;

  double min_fraction=10;
  auto beta = get_beta_list(n_points_per_decade, stops_at, includes_zero);

  auto mean_logLik = bayesian_linear_regression_calculate_mean_logLik(
      prior_b.value(), prior_eps_df, prior_eps_variance, y, X, beta);

  std::cerr << "\n\nbeta and mean_logLik!!\n";
  for (std::size_t i = 0; i < beta.size(); ++i)
    std::cerr << beta[i] << "\t" << mean_logLik[i] << "\n";

  if (false)
    auto opt =
        thermo_max_iter(linear_model, y, X, path, "Iteri",
                        num_scouts_per_ensemble, thermo_jumps_every, max_iter,
                        n_points_per_decade, stops_at, includes_zero, initseed);

  if (false)
    auto opt2 = thermo_convergence(
        linear_model, y, X, path, "thermo_6", num_scouts_per_ensemble,
        thermo_jumps_every, checks_derivative_every_model_size,
        n_points_per_decade, stops_at, includes_zero, initseed);

  // std::cout<<y;
  if (true)
  auto opt3 = cuevi_convergence(
      linear_model, y, X, path, "cuevi_10_7_6", num_scouts_per_ensemble,min_fraction,
      thermo_jumps_every, checks_derivative_every_model_size,max_ratio,
      n_points_per_decade,n_points_per_decade_fraction, stops_at, includes_zero, initseed);

  return 0;
}
