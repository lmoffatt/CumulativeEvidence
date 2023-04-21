#include "bayesian_linear_regression.h"
#include "distributions.h"
#include "lapack_headers.h"
#include "matrix.h"
#include "matrix_random.h"
#include "mcmc.h"
#include <iostream>
// using namespace std;

int main() {
  auto initseed = 0;
  auto mt = init_mt(initseed);

  auto npar = 2ul;
  auto nsamples = 10000ul;
  auto log10_std_par = 1.0;

  auto mean_mean_par = 0.0;
  auto std_mean_par = 10.0;

  auto mean_b = 0.0;
  auto std_b = 10.0;

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

  auto prior_error_distribution = inverse_gamma_distribution(a_0, b_0);

  auto s = std::sqrt(prior_error_distribution(mt));
  std::cerr << "proposed s = " << s << "\n";
  auto eps = random_matrix_normal(mt, nsamples, 1, 0, s);

  auto y = X * tr(b) + eps;

  auto prior_b = make_multivariate_normal_distribution(
      Matrix<double>(1ul, npar, mean_b), IdM<double>(npar));

  auto prior = distributions(prior_error_distribution, prior_b.value());
  auto sam = prior(mt);
  std::cerr << "prior sample\n" << sam;


  static_assert(Multivariate<decltype(prior_b.value())>);

  std::cerr << "prior logP =" << prior.logP(sam);

  if (prior_b.valid()) {
    auto reg = bayesian_linear_regression(prior_b.value(), prior_eps_df,
                                          prior_eps_variance, y, X);
    std::cout << reg;
  }

  // std::cout<<y;

  return 0;
}
