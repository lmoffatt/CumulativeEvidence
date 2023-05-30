#ifndef BAYESIAN_LINEAR_REGRESSION_H
#define BAYESIAN_LINEAR_REGRESSION_H

#include "distributions.h"
#include "matrix.h"
#include "matrix_random.h"
#include "multivariate_normal_distribution.h"
#include "parallel_tempering.h"
#include <cmath>

inline double digamma(double x) {
  if (x > 2)

    return std::log(x) - 0.5 / x - (1. / 12.) * std::pow(x, -2) +
           (1. / 120.) * std::pow(x, -4) - (1. / 252.) * std::pow(x, -6) +
           (1. / 240.) * std::pow(x, -8) - (5. / 660.) * std::pow(x, -10) +
           (691. / 32760.) * std::pow(x, -12) - (1. / 12.) * std::pow(x, -14);
  else {
    double eps = 1e-6;
    double xp = x * (1 + eps);
    return (std::lgamma(xp) - std::lgamma(x)) / (xp - x);
  }
}

struct linear_model {};

Maybe_error<double> logLikelihood(linear_model, const Matrix<double> &beta,
                                  const Matrix<double> &y,
                                  const Matrix<double> &X) {
  assert(beta.ncols() - 1 == X.ncols() && "beta has the right number");
  assert(y.nrows() == X.nrows() && "right number of rows in X");

  double logvar = beta[0];
  double var = std::exp(logvar);
  auto b = Matrix<double>(1, X.ncols(), false);
  for (std::size_t i = 0; i < b.ncols(); ++i)
    b[i] = beta[i + 1];
  auto yfit = X * tr(b);
  auto ydiff = y - yfit;
  double n = y.nrows();
  double SS = xtx(ydiff);
  double chi2 = SS / var;
  double out = -0.5 * (n * log(2 * std::numbers::pi) + n * logvar + chi2);
  if (std::isfinite(out))
    return out;
  else {
    std::cerr << std::string("likelihood error: ") + std::to_string(out)
              << "\n";
    return std::string("likelihood error: ") + std::to_string(out);
  }
}

auto simulate(std::mt19937_64 &mt, linear_model, const Matrix<double> &beta,
              const Matrix<double> &X) {
  assert(beta.ncols() - 1 == X.ncols() && "beta has the right number");

  double logvar = beta[0];
  double var = std::exp(logvar);
  auto b = Matrix<double>(1, X.ncols(), false);
  for (std::size_t i = 0; i < b.ncols(); ++i)
    b[i] = beta[i + 1];
  auto yfit = X * tr(b);
  auto noise =
      random_matrix_normal(mt, yfit.nrows(), yfit.ncols(), 0, std::sqrt(var));

  return yfit + noise;
}

template <class Cov>
class bayesian_linear_model_old
    : public linear_model,
      distributions<log_inverse_gamma_distribution,
                    multivariate_normal_distribution<double, Cov>> {
public:
  using dist_type =
      distributions<log_inverse_gamma_distribution,
                    multivariate_normal_distribution<double, Cov>>;

  using dist_type::operator();
  using dist_type::logP;
  using dist_type::size;
  bayesian_linear_model_old(dist_type &&d)
      : linear_model{}, dist_type{std::move(d)} {}
};
template <class Cov>
class bayesian_linear_model
    : public linear_model,
     public multivariate_gamma_normal_distribution<double, Cov> {
  public:
  using dist_type =
      multivariate_gamma_normal_distribution<double, Cov>;

  using dist_type::operator();
  using dist_type::logP;
  using dist_type::size;
  bayesian_linear_model(dist_type &&d)
      : linear_model{}, dist_type{std::move(d)} {}
};

template <class Cov>
  requires(Covariance<double, Cov>)
Maybe_error<bayesian_linear_model<Cov>>
make_bayesian_linear_model(double prior_eps_df, double prior_eps_variance,
                           Matrix<double> &&mean, Cov &&cov) {
  auto a = prior_eps_df / 2.0;
  auto b = prior_eps_df * prior_eps_variance / 2.0;
  auto prior =
      make_multivariate_normal_distribution(std::move(mean), std::move(cov));
  if (prior)
    return bayesian_linear_model<Cov>(multivariate_gamma_normal_distribution(
        log_inverse_gamma_distribution(a, b), std::move(prior.value())));
  else
    return prior.error() + "\n in make_bayesian_linear_model";
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

  auto beta_ = inv(SSx) * tr(X) * y;
  auto beta_nn = tr(inv(L_n) * (SSx * beta_ + (L_0 * tr(prior.mean()))));
  auto beta_n = tr(inv(L_n) * (tr(X) * y + (L_0 * tr(prior.mean()))));

  std::cerr << "beta_n<<beta_nn<<beta_n-beta_nn" << beta_n << beta_nn
            << beta_n - beta_nn;
  auto yfit = X * tr(beta_n);
  auto ydiff = y - yfit;
  auto SS = xtx(ydiff.value());

  auto posterior =
      make_multivariate_normal_distribution_from_precision(beta_n, L_n);
  auto a_n = a_0 + n / 2.0;
  auto b_n = b_0 + 0.5 * (xtx(y) + xAxt(beta_0, L_0) - xAxt(beta_n, L_n));

  auto b_n2 = b_0 + 0.5 * SS + 0.5 * xAxt(beta_0 - beta_n, L_0);
  // auto
  // E_n=std::pow(2*std::numbers::pi,-n/2)*std::sqrt(det(L_0)/det(L_n))*std::pow(b_0,a_0)/std::pow(b_n,a_n)*std::tgamma(a_n)/std::tgamma(a_0);

  std::cerr << "logdet(L_n)" << logdet(L_n);

  auto logE_n = -0.5 * n * std::log(2 * std::numbers::pi) +
                0.5 * (logdet(L_0) - logdet(L_n)) + a_0 * log(b_0) -
                a_n * log(b_n) + std::lgamma(a_n) - std::lgamma(a_0);
  return std::tuple(
      logE_n, "beta_n", beta_n, "b_n", b_n, "b_n2", b_n2,

      "xtx(y)", xtx(y),
      "0.5*( logdet(L_0)) + a_0 * log(b_0) - std::lgamma(a_0)",
      0.5 * (logdet(L_0)) + a_0 * log(b_0) - std::lgamma(a_0),
      "0.5*(-logdet(L_n)) - a_n * log(b_n) + std::lgamma(a_n)",
      0.5 * (-logdet(L_n)) - a_n * log(b_n) + std::lgamma(a_n),
      "-0.5* n*std::log(2*std::numbers::pi)",
      -0.5 * n * std::log(2 * std::numbers::pi),
      "0.5*( logdet(L_0) -logdet(L_n))", 0.5 * (logdet(L_0) - logdet(L_n)),
      "a_0 * log(b_0) - a_n * log(b_n)", a_0 * log(b_0) - a_n * log(b_n),
      "std::lgamma(a_n) - std::lgamma(a_0)",
      std::lgamma(a_n) - std::lgamma(a_0),

      "sqrt(b_n / (a_n - 1))", sqrt(b_n / (a_n - 1)), "b_n / (a_n - 1)",
      b_n / (a_n - 1));
}


template <class Cova>
    requires Covariance<double, Cova>
auto bayesian_linear_regression_calculate_Evidence(
    const multivariate_normal_distribution<double, Cova> &prior,
    double prior_eps_df, double prior_eps_variance, const Matrix<double> &y,
    const Matrix<double> &X) {
  auto L_0 = prior.cov_inv() * prior_eps_variance;
  auto SSx = XTX(X);
  auto n = y.nrows();
  auto a_0 = prior_eps_df / 2.0;
  auto b_0 = prior_eps_df * prior_eps_variance / 2.0;
  auto beta_0 = prior.mean();
  auto L_n = SSx + L_0;
  auto beta_n = tr(inv(L_n) * (tr(X) * y + (L_0 * tr(prior.mean()))));

  auto yfit = X * tr(beta_n);
  auto ydiff = y - yfit;
  auto SS = xtx(ydiff.value());
  auto a_n = a_0 + n / 2.0;
  auto b_n = b_0 + 0.5 * SS + 0.5 * xAxt(beta_0 - beta_n, L_0);

  auto logE_n = -0.5 * n * std::log(2 * std::numbers::pi) +
                0.5 * (logdet(L_0) - logdet(L_n)) + a_0 * log(b_0) -
                a_n * log(b_n) + std::lgamma(a_n) - std::lgamma(a_0);
  return logE_n;
}

template <class Cova>
    requires Covariance<double, Cova>
auto bayesian_linear_regression_calculate_Evidence(
    const multivariate_gamma_normal_distribution<double, Cova> &prior,
    const Matrix<double> &y,
    const Matrix<double> &X) {
  auto a_0 = prior.alpha(); ;
  auto prior_eps_df= 2.0 *a_0;
  auto b_0= prior.beta();
  auto  prior_eps_variance = 2.0* b_0/prior_eps_df;

  auto L_0 = prior.Gamma();


  auto SSx = XTX(X);
  auto n = y.nrows();
  auto beta_0 = prior.mean();
  auto L_n = SSx + L_0;
  auto beta_n = tr(inv(L_n) * (tr(X) * y + (L_0 * tr(prior.mean()))));

  auto yfit = X * tr(beta_n);
  auto ydiff = y - yfit;
  auto SS = xtx(ydiff.value());
  auto a_n = a_0 + n / 2.0;
  auto b_n = b_0 + 0.5 * SS + 0.5 * xAxt(beta_0 - beta_n, L_0);

  auto logE_n = -0.5 * n * std::log(2 * std::numbers::pi) +
                0.5 * (logdet(L_0) - logdet(L_n)) + a_0 * log(b_0) -
                a_n * log(b_n) + std::lgamma(a_n) - std::lgamma(a_0);
  return logE_n;
}



template <class Cova>
    requires Covariance<double, Cova>
auto bayesian_linear_regression_calculate_posterior(
    const multivariate_normal_distribution<double, Cova> &prior,
    double prior_eps_df, double prior_eps_variance, const Matrix<double> &y,
    const Matrix<double> &X, double beta0) {
  auto L_0 = prior.cov_inv() * prior_eps_variance;
  auto SSx = XTX(X);
  auto n = y.nrows();
  auto a_0 = prior_eps_df / 2.0;
  auto b_0 = prior_eps_df * prior_eps_variance / 2.0;
  auto beta_0 = prior.mean();
  auto L_n =   L_0 + beta0 * SSx;

  auto beta_n =
      tr(inv(L_n) * (beta0*(tr(X) * y) + (L_0 * tr(prior.mean()))));

  auto yfit = X * tr(beta_n);
  auto ydiff = y - yfit;
  auto SS = beta0 * xtx(ydiff.value());

  auto a_n = a_0 + beta0 * n / 2.0;
  auto b_n = b_0 + 0.5 *  SS + 0.5 * xAxt(beta_0 - beta_n, L_0);
  return std::tuple(std::log(b_n.value()/a_n),beta_n.value());
}



template <class Cova>
    requires Covariance<double, Cova>
auto bayesian_linear_regression_calculate_posterior(
    const multivariate_gamma_normal_distribution<double, Cova> &prior,
    const Matrix<double> &y,
    const Matrix<double> &X) {
  auto a_0 = prior.alpha(); ;
  auto prior_eps_df= 2.0 *a_0;
  auto b_0= prior.beta();
  auto  prior_eps_variance = 2.0* b_0/prior_eps_df;

  auto L_0 = prior.Gamma();
    auto SSx = XTX(X);
  auto n = y.nrows();
  auto beta_0 = prior.mean();
  auto L_n =   L_0 +  SSx;

  auto beta_n =
      tr(inv(L_n) * ((tr(X) * y) + (L_0 * tr(prior.mean()))));

  auto yfit = X * tr(beta_n);
  auto ydiff = y - yfit;
  auto SS = xtx(ydiff.value());

  auto a_n = a_0 +  n / 2.0;
  auto b_n = b_0 + 0.5 *  SS + 0.5 * xAxt(beta_0 - beta_n, L_0);

  auto posterior_Normal= make_multivariate_normal_distribution_from_precision(std::move(beta_n), std::move(L_n));


  return multivariate_gamma_normal_distribution<double, SymPosDefMatrix<double>>(
      log_inverse_gamma_distribution(a_n,b_n.value()),
                                                              std::move(posterior_Normal.value()));
}

template <class Cova>
    requires Covariance<double, Cova>
auto bayesian_linear_regression_calculate_mean_logLik(
    const multivariate_gamma_normal_distribution<double, Cova> &prior,
    const Matrix<double> &y,
    const Matrix<double> &X, double beta0) {
  auto a_0 = prior.alpha(); ;
  auto b_0= prior.beta();
  auto L_0 = prior.Gamma();
  auto SSx = XTX(X);
  auto n = y.nrows();
  auto beta_0 = prior.mean();
  auto L_n =   L_0 + beta0 * SSx;
  auto beta_n =
      tr(inv(L_n) * (beta0*(tr(X) * y) + (L_0 * tr(prior.mean()))));
  auto yfit = X * tr(beta_n);
  auto ydiff = y - yfit;
  auto SS = beta0 * xtx(ydiff.value());

  auto a_n = a_0 + beta0 * n / 2.0;
  auto b_n = b_0 + 0.5 *  SS + 0.5 * xAxt(beta_0 - beta_n, L_0);
  double d_a_n = 1.0 * n / 2.0;
  auto d_b_n = 0.5 * xtx(ydiff.value());
  auto mean_logLi = -0.5 * n * std::log(2 * std::numbers::pi) -
                    0.5 * Trace(inv(L_n) * SSx) - a_n / b_n * d_b_n +
                    (digamma(a_n) - log(b_n)) * d_a_n;
  return mean_logLi;
}

template <class Cova>
    requires Covariance<double, Cova>
Maybe_error<by_beta<double>>  bayesian_linear_regression_calculate_mean_logLik(
    const multivariate_gamma_normal_distribution<double, Cova> &prior,
    const Matrix<double> &y,
    const Matrix<double> &X, by_beta<double> const& beta0) {

  by_beta<double> out(size(beta0));
  for (std::size_t i=0; i<size(out); ++i)
  {
    auto meanLogLiki=bayesian_linear_regression_calculate_mean_logLik(prior,y,X,beta0[i]);
    if (!meanLogLiki)
        return "bayesian_linear_regression_calculate_mean_logLik error for beta =" +std::to_string(beta0[i])+":  "+ meanLogLiki.error();
    else
        out[i]=meanLogLiki.value();
  }
  return out;
}



template <class Cova>
    requires Covariance<double, Cova>
auto bayesian_linear_regression_calculate_mean_logLik(
    const multivariate_normal_distribution<double, Cova> &prior,
    double prior_eps_df, double prior_eps_variance, const Matrix<double> &y,
    const Matrix<double> &X, double beta0) {
  auto L_0 = prior.cov_inv() * prior_eps_variance;
  auto SSx = XTX(X);
  auto n = y.nrows();
  auto a_0 = prior_eps_df / 2.0;
  auto b_0 = prior_eps_df * prior_eps_variance / 2.0;
  auto beta_0 = prior.mean();
  auto L_n =   L_0 + beta0 * SSx;
  auto beta_n =
      tr(inv(L_n) * (beta0*(tr(X) * y) + (L_0 * tr(prior.mean()))));
  auto yfit = X * tr(beta_n);
  auto ydiff = y - yfit;
  auto SS = beta0 * xtx(ydiff.value());

  auto a_n = a_0 + beta0 * n / 2.0;
  auto b_n = b_0 + 0.5 *  SS + 0.5 * xAxt(beta_0 - beta_n, L_0);
  double d_a_n = 1.0 * n / 2.0;
  auto d_b_n = 0.5 * xtx(ydiff.value());
  auto mean_logLi = -0.5 * n * std::log(2 * std::numbers::pi) -
                    0.5 * Trace(inv(L_n) * SSx) - a_n / b_n * d_b_n +
                    (digamma(a_n) - log(b_n)) * d_a_n;
  return mean_logLi;
}




template <class Cova>
    requires Covariance<double, Cova>
auto bayesian_linear_regression_calculate_mean_logLik_posterior(
    const multivariate_normal_distribution<double, Cova> &prior,
    double prior_eps_df, double prior_eps_variance, const Matrix<double> &y,
    const Matrix<double> &X, double beta0) {

  auto L_0 = prior.cov_inv() * prior_eps_variance;
  auto SSx = XTX(X);
  auto n = y.nrows();
  auto a_0 = prior_eps_df / 2.0;
  auto b_0 = prior_eps_df * prior_eps_variance / 2.0;
  auto beta_0 = prior.mean();
  std::tuple<Maybe_error<double>,double,Parameters> mean_logLik;


    auto L_n =   L_0 + beta0 * SSx;

    auto beta_n =
        tr(inv(L_n) * (beta0*(tr(X) * y) + (L_0 * tr(prior.mean()))));

    auto yfit = X * tr(beta_n);
    auto ydiff = y - yfit;
    auto SS = beta0 * xtx(ydiff.value());

    auto a_n = a_0 + beta0 * n / 2.0;
    auto b_n = b_0 + 0.5 *  SS + 0.5 * xAxt(beta_0 - beta_n, L_0);


    auto logE_n = -0.5 * beta0 * n * std::log(2 * std::numbers::pi) +
                  0.5 * (logdet(L_0) - logdet(L_n)) + a_0 * log(b_0) -
                  a_n * log(b_n) + std::lgamma(a_n) - std::lgamma(a_0);

    double d_a_n = 1.0 * n / 2.0;
    auto d_b_n = 0.5 * xtx(ydiff.value());

    auto mean_logLi = -0.5 * n * std::log(2 * std::numbers::pi) -
                      0.5 * Trace(inv(L_n) * SSx) - a_n / b_n * d_b_n +
                      (digamma(a_n) - log(b_n)) * d_a_n;





    return std::tuple(mean_logLi,std::log(b_n.value()/a_n),beta_n.value());
 }





template <class Cova>
  requires Covariance<double, Cova>
auto bayesian_linear_regression_calculate_mean_logLik(
    const multivariate_normal_distribution<double, Cova> &prior,
    double prior_eps_df, double prior_eps_variance, const Matrix<double> &y,
    const Matrix<double> &X, by_beta<double> const &beta0) {

  auto beta = beta0;
  auto L_0 = prior.cov_inv() * prior_eps_variance;
  auto SSx = XTX(X);
  auto n = y.nrows();
  auto a_0 = prior_eps_df / 2.0;
  auto b_0 = prior_eps_df * prior_eps_variance / 2.0;
  auto beta_0 = prior.mean();
  by_beta<std::tuple<Maybe_error<double>,double,Parameters>> mean_logLik;
  mean_logLik.reserve(beta.size());


  //  by_beta<Maybe_error<double>> mean_Ev;
//  mean_Ev.reserve(beta.size());
//  by_beta<Maybe_error<double>> mean_logLik_diff;
//  mean_logLik_diff.reserve(beta.size());

//  by_beta<Maybe_error<double>> diff_logdetLn;
//  diff_logdetLn.reserve(beta.size());

//  by_beta<Maybe_error<double>> diff_alogb;
//  diff_alogb.reserve(beta.size());

//  by_beta<Maybe_error<double>> diff_lgamma;
//  diff_lgamma.reserve(beta.size());


  for (std::size_t i = 0; i < beta.size(); ++i) {
    auto L_n =   L_0 + beta[i] * SSx;

    auto beta_n =
        tr(inv(L_n) * (beta[i]*(tr(X) * y) + (L_0 * tr(prior.mean()))));

    auto yfit = X * tr(beta_n);
    auto ydiff = y - yfit;
    auto SS = beta[i] * xtx(ydiff.value());
    std::cerr<<"SS\n"<<SS<<"\n";

    auto a_n = a_0 + beta[i] * n / 2.0;
    auto b_n = b_0 + 0.5 *  SS + 0.5 * xAxt(beta_0 - beta_n, L_0);


    auto logE_n = -0.5 * beta[i] * n * std::log(2 * std::numbers::pi) +
                  0.5 * (logdet(L_0) - logdet(L_n)) + a_0 * log(b_0) -
                  a_n * log(b_n) + std::lgamma(a_n) - std::lgamma(a_0);
//    std::cerr<<beta[i]<<"\t"<<"Ev"<<logE_n<<"\n";
//    std::cerr<<beta[i]<<"\tL_0\t"<<L_0 <<"\n";
//    std::cerr<<beta[i]<<"\tL_n\t"<<L_n<<"\n";
//    std::cerr<<beta[i]<<"\tlogdet(L_0) - logdet(L_n)\t"<<logdet(L_0) - logdet(L_n)<<"\n";
//    std::cerr<<beta[i]<<"\t+ a_0 * log(b_0) -a_n * log(b_n)\t"<<+ a_0 * log(b_0) -a_n * log(b_n)<<"\n";
//    std::cerr<<beta[i]<<"\t+ std::lgamma(a_n) - std::lgamma(a_0)\t"<<+ std::lgamma(a_n) - std::lgamma(a_0)<<"\n";


    double d_a_n = 1.0 * n / 2.0;
    auto d_b_n = 0.5 * xtx(ydiff.value());

    auto mean_logLi = -0.5 * n * std::log(2 * std::numbers::pi) -
                      0.5 * Trace(inv(L_n) * SSx) - a_n / b_n * d_b_n +
                      (digamma(a_n) - log(b_n)) * d_a_n;





    mean_logLik.push_back(std::tuple(mean_logLi,std::log(b_n.value()/a_n),beta_n.value()));
//    mean_Ev.push_back(logE_n);
//    diff_logdetLn.push_back(logdet(L_n));
//    diff_alogb.push_back(a_n * log(b_n));
//    diff_lgamma.push_back(std::lgamma(a_n));
  }
/*
  for (std::size_t i = 0; i < beta.size(); ++i) {
    beta[i]=std::max(beta[i]*(1+1e-6),beta[i]+1e-9);
    auto L_n = beta[i] * SSx + L_0;

    auto beta_n =
        tr(inv(L_n) * (beta[i] * (tr(X) * y) + (L_0 * tr(prior.mean()))));

    auto yfit = X * tr(beta_n);
    auto ydiff = y - yfit;
    auto SS = beta[i] * xtx(ydiff.value());
    std::cerr<<"SS\n"<<SS<<"\n";

    auto a_n = a_0 + beta[i] * n / 2.0;
    auto b_n = b_0 + 0.5 * SS + 0.5 * xAxt(beta_0 - beta_n, L_0);


    auto logE_n = -0.5 * beta[i] * n * std::log(2 * std::numbers::pi) +
                  0.5 * (logdet(L_0) - logdet(L_n)) +
                  a_0 * log(b_0) - a_n * log(b_n) +
                  std::lgamma(a_n) - std::lgamma(a_0);
    double d_a_n = 1.0 * n / 2.0;
    auto d_b_n = 0.5 * xtx(ydiff.value());
    std::cerr<<beta[i]<<"\t"<<"Ev"<<logE_n<<"\n\n";
    std::cerr<<beta[i]<<"\t\t diff logdet(L_n))\t"<<-(diff_logdetLn[i]-logdet(L_n))/(beta[i]-beta0[i])<<"\n";
    std::cerr<<std::setprecision(12)<<"L_n\n"<<L_n<<"\n";
    std::cerr<<std::setprecision(12)<<"SSx\n"<<SSx<<"\n";
    std::cerr<<beta[i]<<"\t\t inv(L_n) * SSx\t"<<inv(L_n) * SSx<<"\n\n";
    std::cerr<<beta[i]<<"\t\t inv(L_n) \t"<<inv(L_n) <<"\n\n";

    std::cerr<<beta[i]<<"\t\tTrace(inv(L_n) * SSx)\t"<<Trace(inv(L_n) * SSx)<<"\n\n";
    std::cerr<<beta[i]<<"\t\tTrace(inv(static_cast<Matrix<double>const&>(L_n)) * SSx)\t"<<Trace(inv(static_cast<Matrix<double>const&>(L_n)) * SSx)<<"\n\n";

    std::cerr<<beta[i]<<"\t\tdiff_alogb[i]-a_n * log(b_n))\t"<<-(diff_alogb[i]-a_n * log(b_n))/(beta[i]-beta0[i])<<"\n";
    std::cerr<<beta[i]<<"\t\t(a_n / b_n * d_b_n +log(b_n) * d_a_n)\t"<<(a_n / b_n * d_b_n +log(b_n) * d_a_n)<<"\n\n";


    std::cerr<<beta[i]<<"\t\t-(diff_lgamma[i]-std::lgamma(a_n))\t"<<-(diff_lgamma[i]-std::lgamma(a_n))/(beta[i]-beta0[i])<<"\n";
    std::cerr<<beta[i]<<"\t\tdigamma(a_n)  * d_a_n\t"<<+digamma(a_n)  * d_a_n<<"\n\n";



    auto mean_logLi = -0.5 * n * std::log(2 * std::numbers::pi) -
                      0.5 * Trace(inv(L_n) * SSx) -
                      (a_n / b_n * d_b_n +log(b_n) * d_a_n)+
                      digamma(a_n)  * d_a_n;
    mean_logLik_diff.push_back((logE_n-mean_Ev[i])/(beta[i]-beta0[i]));

  }

  */

//  for (std::size_t i=0; i<beta0.size(); ++i)
//    std::cerr<<"beta= "<<beta0[i]<<"\tmean_logLi"<<mean_logLik[i]<<"\tdiff: "<<mean_logLik_diff[i]<<"\n";

  return mean_logLik;
}

#endif // BAYESIAN_LINEAR_REGRESSION_H
