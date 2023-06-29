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
  auto myseed = 9762841416869310605ul;
  std::cerr << "myseed=\n" << myseed << "\n";

  auto npar = 40ul;
  auto nsamples = 5000ul;
  auto log10_std_par = 1.0;

  auto mean_mean_par = 0.0;
  auto std_mean_par = 10.0;
  auto mean_b = 0.0;
  auto std_b = 10.0;
  auto prior_error_factor = 1;

  double prior_eps_df = 1.0;
  double prior_eps_variance = 1.0;

  myseed=calc_seed(myseed);
  auto mt = init_mt(myseed);
  auto a_0 = prior_eps_df / 2.0;
  auto b_0 = prior_eps_df * prior_eps_variance / 2.0;

  std::size_t num_scouts_per_ensemble = 32;
  double n_points_per_decade = 3;
  double n_points_per_decade_fraction=3;
  double stops_at = 1e-7;
  bool includes_zero = true;
  std::size_t max_iter = 5000;
  std::string path = "";
  std::string filename = "A";


  std::size_t checks_derivative_every_model_size = 1000;
  double max_ratio=8;

  double min_fraction=2;

  auto my_linear_model =
      make_bayesian_linear_model(
          prior_eps_df, prior_eps_variance,  npar, mean_b,std_b, prior_error_factor)
          .value();
  auto [X,mean_par,cov_par] = independent_variable_model(npar,log10_std_par,mean_mean_par,std_mean_par)(mt, nsamples);
//  auto y = X * tr(b) + eps;

  auto b= sample(mt,my_linear_model);

  auto y= simulate(mt,my_linear_model,b,X);


  auto beta = get_beta_list(n_points_per_decade, stops_at, includes_zero);
  std::size_t thermo_jumps_every = my_linear_model.size()*1e0;



  if (false)
  {
      auto tmi=thermo_by_max_iter( path, "Iteri",
                                    num_scouts_per_ensemble, thermo_jumps_every, max_iter,
                                    n_points_per_decade, stops_at, includes_zero, myseed);
      auto opt =
          evidence(std::move(tmi),my_linear_model.prior(),my_linear_model.likelihood(), y, X);

  }
  if (false)
  {
    auto opt2 = thermo_convergence(
        my_linear_model.prior(),my_linear_model.likelihood(), y, X, path, "exp_thermo", num_scouts_per_ensemble,
        thermo_jumps_every, checks_derivative_every_model_size,
        n_points_per_decade, stops_at, includes_zero, myseed);

  // std::cout<<y;
  }
  if (true)
  {
    auto cbc=cuevi_by_convergence(path, "exp_cuevi_40", num_scouts_per_ensemble,min_fraction,
        thermo_jumps_every, checks_derivative_every_model_size,max_ratio,
                                    n_points_per_decade,n_points_per_decade_fraction, stops_at, includes_zero, myseed);
        auto opt3 = evidence(std::move(cbc),
      my_linear_model.prior(),my_linear_model.likelihood(), y, X);
  }
  return 0;
}
