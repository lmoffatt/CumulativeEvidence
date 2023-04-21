#ifndef BAYESIAN_LINEAR_REGRESSION_H
#define BAYESIAN_LINEAR_REGRESSION_H

#include "distributions.h"
#include "matrix.h"
#include "matrix_random.h"
#include "multivariate_normal_distribution.h"
#include "parallel_tempering.h"
#include <cmath>

struct linear_model {};


auto logLikelihood(linear_model, const Matrix<double> &beta,
                   const Matrix<double> &y, const Matrix<double> &X) {
  assert(beta.ncols() - 1 == X.ncols() && "beta has the right number");
  assert(y.nrows() == X.nrows() && "right number of rows in X");

  double var = beta[0];
  auto b = Matrix<double>(1, X.ncols(), false);
  for (std::size_t i = 0; i < b.ncols(); ++i)
    b[i] = beta[i + 1];
  auto yfit = X * tr(b);
  auto ydiff = y - yfit;
  double n = y.nrows();
  double SS = xtx(ydiff);
  double chi2 = SS / n / var;
  return -0.5 * (n * log(std::numbers::pi) + n * std::log(var) + chi2);
}

auto simulate(std::mt19937_64& mt,linear_model,
                     const Matrix<double>& beta,const Matrix<double>& X)
{
  assert(beta.ncols() - 1 == X.ncols() && "beta has the right number");

  double var = beta[0];
  auto b = Matrix<double>(1, X.ncols(), false);
  for (std::size_t i = 0; i < b.ncols(); ++i)
    b[i] = beta[i + 1];
  auto yfit = X * tr(b);
  auto noise= random_matrix_normal(mt,yfit.nrows(),yfit.ncols(),0,std::sqrt(var));

  return yfit+noise;
}


template<class Cov>
class bayesian_linear_model: public linear_model, distributions<inverse_gamma_distribution,multivariate_normal_distribution<double, Cov>>
{
public:

    using dist_type=distributions<inverse_gamma_distribution,multivariate_normal_distribution<double, Cov>>;

    using dist_type::operator();
    using dist_type::logP;

    bayesian_linear_model(dist_type&& d):linear_model{},dist_type{std::move(d)}{}
 };

template<class Cov>
 requires(Covariance<double,Cov>)
Maybe_error<bayesian_linear_model<Cov>> make_bayesian_linear_model(double prior_eps_df, double prior_eps_variance, Matrix<double>&& mean,Cov&& cov  )
 {
    auto a = prior_eps_df / 2.0;
    auto b = prior_eps_df * prior_eps_variance / 2.0;
    auto prior=make_multivariate_normal_distribution(std::move(mean), std::move(cov));
    if (prior)
    return bayesian_linear_model<Cov>(distributions(inverse_gamma_distribution(a,b),std::move(prior.value())));
    else
        return prior.error()+"\n in make_bayesian_linear_model";
 }





template <class Cova>
  requires Covariance<double, Cova>
auto bayesian_linear_regression(
    const multivariate_normal_distribution<double, Cova> &prior,
    double prior_eps_df, double prior_eps_variance, const Matrix<double> &y,
    const Matrix<double> &X) {
  auto L_0 = prior.cov_inv() * prior_eps_variance;
  auto SSx = XTX(X);
  auto n = y.nrows();
  auto a_0 = prior_eps_df / 2.0;
  auto b_0 = prior_eps_df * prior_eps_variance / 2.0;
  auto beta_0 = prior.mean();
  // auto beta_ml=inv(SSx)*tr(X)*y;
  auto L_n = SSx + L_0;
  auto beta_n = tr(inv(L_n) * (tr(X) * y + (L_0 * tr(prior.mean()))));

  auto posterior =
      make_multivariate_normal_distribution_from_precision(beta_n, L_n);
  auto a_n = a_0 + n / 2;
  auto b_n = b_0 + 0.5 * (xtx(y) + xAxt(beta_0, L_0) - xAxt(beta_n, L_n));

  // auto
  // E_n=std::pow(2*std::numbers::pi,-n/2)*std::sqrt(det(L_0)/det(L_n))*std::pow(b_0,a_0)/std::pow(b_n,a_n)*std::tgamma(a_n)/std::tgamma(a_0);
  auto logE_n = 0.5 * (-std::log(std::numbers::pi) * n + logdet(L_0) -
                       logdet(L_n) + a_0 * log(b_0) - a_n * log(b_n) +
                       std::lgamma(a_n) - std::lgamma(a_0));
  return std::tuple(logE_n, a_n, b_n, sqrt(b_n / (a_n - 1)), posterior,
                    -std::log(std::numbers::pi) * n, logdet(L_0), -logdet(L_n),
                    +a_0 * log(b_0), -a_n * log(b_n), std::lgamma(a_n),
                    -std::lgamma(a_0), logE_n, a_n, b_n, sqrt(b_n / (a_n - 1)));
}

#endif // BAYESIAN_LINEAR_REGRESSION_H
