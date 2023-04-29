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
    double eps = 1e-5;
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

  double var = beta[0];
  auto b = Matrix<double>(1, X.ncols(), false);
  for (std::size_t i = 0; i < b.ncols(); ++i)
    b[i] = beta[i + 1];
  auto yfit = X * tr(b);
  auto noise =
      random_matrix_normal(mt, yfit.nrows(), yfit.ncols(), 0, std::sqrt(var));

  return yfit + noise;
}

template <class Cov>
class bayesian_linear_model
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
    return bayesian_linear_model<Cov>(distributions(
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
  auto a_n = a_0 + n / 2;
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
auto bayesian_linear_regression(
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

  by_beta<double> mean_logLik(beta.size());
  by_beta<double> mean_logLik_diff(beta.size());
  by_beta<double> Ev_b(beta.size());
  by_beta<double> Ev_b2(beta.size());

  for (std::size_t i = 0; i < beta.size(); ++i) {
    // auto beta_ml=inv(SSx)*tr(X)*y;
    auto L_n = beta[i] * SSx + L_0;

    auto beta_n =
        tr(inv(L_n) * (beta[i] * (tr(X) * y) + (L_0 * tr(prior.mean()))));

    auto yfit = X * tr(beta_n);
    auto ydiff = y - yfit;
    auto SS = beta[i] * xtx(ydiff.value());

    auto a_n = a_0 + beta[i] * n / 2;
    auto b_n2 =
        b_0 + 0.5 * (beta[i] * xtx(y) + xAxt(beta_0, L_0) - xAxt(beta_n, L_n));

    auto b_n = b_0 + 0.5 * SS + 0.5 * xAxt(beta_0 - beta_n, L_0);

    // std::cerr<<"\n\nb_n b_n2 "<<b_n<<"\t"<<b_n2<<"\n\n";
    //  auto
    //  E_n=std::pow(2*std::numbers::pi,-n/2)*std::sqrt(det(L_0)/det(L_n))*std::pow(b_0,a_0)/std::pow(b_n,a_n)*std::tgamma(a_n)/std::tgamma(a_0);

    // std::cerr<<"logdet(L_n)"<<logdet(L_n);

    auto logE_n = -0.5 * beta[i] * n * std::log(2 * std::numbers::pi) +
                  0.5 * (logdet(L_0) - logdet(L_n)) + a_0 * log(b_0) -
                  a_n * log(b_n) + std::lgamma(a_n) - std::lgamma(a_0);

    Ev_b[i] = logE_n.value();
    // std::cerr<<"logE_n\n"<<logE_n;

    double d_a_n = 1.0 * n / 2.0;
    auto d_b_n = 0.5 * xtx(ydiff.value());
    std::cerr << "\nd_b_n\n" << d_b_n << "\n";

    auto mean_logLi = -0.5 * n * std::log(2 * std::numbers::pi) -
                      0.5 * Trace(inv(L_n).value() * SSx) - a_n / b_n * d_b_n +
                      (digamma(a_n) - log(b_n)) * d_a_n;
    mean_logLik[i] = mean_logLi.value();

    std::cerr << "beta\n" << beta[i] << "\n";
    std::cerr << "mean_logLik\n" << mean_logLik[i] << "\n";
    std::cerr << "-0.5*Trace(inv(L_n).value()*SSx)\n"
              << -0.5 * Trace(inv(L_n).value() * SSx) << "\n";
    std::cerr << "-a_n/b_n*d_b_n\n" << -a_n / b_n * d_b_n << "\n";
    std::cerr << "(digamma(a_n)-log(b_n))*d_a_n\n"
              << (digamma(a_n) - log(b_n)) * d_a_n << "\n";
    std::cerr << "-a_n/b_n*d_b_n-log(b_n)*d_a_n\n"
              << -a_n / b_n * d_b_n - log(b_n) * d_a_n << "\n";
    std::cerr << "a_n\n" << a_n << "\n";
    std::cerr << "digamma(a_n)*d_a_n\n" << digamma(a_n) * d_a_n << "\n";
  }

  for (std::size_t i = 0; i < beta.size(); ++i) {
    auto beta_old = beta[i];
    beta[i] = beta[i] * (1 + 1e-5);
    // auto beta_ml=inv(SSx)*tr(X)*y;
    auto L_n = beta[i] * SSx + L_0;

    auto beta_n =
        tr(inv(L_n) * (beta[i] * (tr(X) * y) + (L_0 * tr(prior.mean()))));

    auto yfit = X * tr(beta_n);
    auto ydiff = y - yfit;
    auto SS = beta[i] * xtx(ydiff.value());

    auto a_n = a_0 + beta[i] * n / 2;
    auto b_n2 =
        b_0 + 0.5 * (beta[i] * xtx(y) + xAxt(beta_0, L_0) - xAxt(beta_n, L_n));

    auto b_n = b_0 + 0.5 * SS + 0.5 * xAxt(beta_0 - beta_n, L_0);

    // std::cerr<<"\n\nb_n b_n2 "<<b_n<<"\t"<<b_n2<<"\n\n";
    //  auto
    //  E_n=std::pow(2*std::numbers::pi,-n/2)*std::sqrt(det(L_0)/det(L_n))*std::pow(b_0,a_0)/std::pow(b_n,a_n)*std::tgamma(a_n)/std::tgamma(a_0);

    // std::cerr<<"logdet(L_n)"<<logdet(L_n);

    auto logE_n = -0.5 * beta[i] * n * std::log(2 * std::numbers::pi) +
                  0.5 * (logdet(L_0) - logdet(L_n)) + a_0 * log(b_0) -
                  a_n * log(b_n) + std::lgamma(a_n) - std::lgamma(a_0);

    Ev_b2[i] = logE_n.value();
    mean_logLik_diff[i] = (Ev_b2[i] - Ev_b[i]) / (beta[i] - beta_old);

    // std::cerr<<"logE_n\n"<<logE_n;

    double d_a_n = 1.0 * n / 2.0;
    auto d_b_n = 0.5 * xtx(ydiff.value());
    std::cerr << "\nd_b_n\n" << d_b_n << "\n";

    auto mean_logLi = -0.5 * Trace(inv(L_n).value() * SSx) - a_n / b_n * d_b_n +
                      (digamma(a_n) - log(b_n)) * d_a_n;
    // mean_logLik[i]=mean_logLi.value();
  }
  return mean_logLik;
}

#endif // BAYESIAN_LINEAR_REGRESSION_H
