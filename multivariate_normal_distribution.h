#ifndef MULTIVARIATE_NORMAL_DISTRIBUTION_H
#define MULTIVARIATE_NORMAL_DISTRIBUTION_H

#include "matrix.h"
#include <random>

template <typename T>
class Covariance
    : public std::variant<DiagPosDetMatrix<T>, SymPosDefMatrix<T>> {
public:
  using base_type = std::variant<DiagPosDetMatrix<T>, SymPosDefMatrix<T>>;
  using cholesky_type = std::variant<DiagPosDetMatrix<T>, DownTrianMatrix<T>>;
  Covariance(DiagPosDetMatrix<T> &&diag_cov) : base_type(std::move(diag_cov)) {}
  Covariance(SymPosDefMatrix<T> &&symm_cov) : base_type(std::move(symm_cov)) {}
  Covariance(DiagPosDetMatrix<T> const &diag_cov) : base_type{diag_cov} {}
  Covariance(SymPosDefMatrix<T> const &symm_cov) : base_type{symm_cov} {}

  friend auto operator*(const Covariance &cov, double a) {
    return std::visit([&a](auto &x) { return Covariance(x * a); }, cov);
  }
  friend auto operator+(const SymPosDefMatrix<T> &SSx, const Covariance &cov) {
    return std::visit([&SSx](auto &x) { return SSx + x; }, cov);
  }
  friend auto operator*(const Covariance &cov, const Matrix<T> &a) {
    return std::visit([&a](auto &x) { return x * a; }, cov);
  }

  friend auto xtAx(const Matrix<T> &x, const Covariance &A) {
    return std::visit([&x](auto &cov) { return xtAx(x, cov); }, A);
  }

  friend auto logdet(const Covariance &A) {
    return std::visit([](auto &cov) { return logdet(cov); }, A);
  }
};

template <typename T>
Maybe_error<typename Covariance<T>::cholesky_type>
cholesky(const Covariance<T> &cov) {
  return make_Maybe_variant<DiagPosDetMatrix<T>, DownTrianMatrix<T>>(
      std::visit([](auto const &var) { return ::cholesky(var); }, cov));
}

template <typename T>
Maybe_error<Covariance<T>>
inv_from_chol(const typename Covariance<T>::cholesky_type &cho) {
  return Maybe_error<Covariance<T>>(std::visit(
      overloaded{
          [](const DiagPosDetMatrix<T> &s) {
            return apply([](auto x) { return std::pow(x, -2); }, s);
          },
          [](const DownTrianMatrix<T> &v) { return ::inv_from_chol(v); }},
      cho));
}

template <typename T> class multivariate_normal_distribution {
private:
  std::normal_distribution<T> n_;
  Matrix<T> mean_;
  Covariance<T> cov_;
  using cholesky_type = Covariance<T>::cholesky_type;
  cholesky_type cho_;
  Covariance<T> cov_inv_;

  static double calc_logdet(const cholesky_type &cho) {
    return std::visit([](auto const &m) { return 2 * logdet(m); }, cho);
  }
  multivariate_normal_distribution(Matrix<T> &&mean, Covariance<T> &&cov,
                                   cholesky_type &&chol,
                                   Covariance<T> &&cov_inv)
      : n_{}, mean_{std::move(mean)}, cov_{std::move(cov)},
        cho_{std::move(chol)}, cov_inv_{std::move(cov_inv)} {}

public:
  template <class Mat, class Cov>
    requires(contains_value<Mat &&, Matrix<double>> &&
             contains_value<Cov &&, Covariance<double>>)
  friend Maybe_error<multivariate_normal_distribution<double>>
  make_multivariate_normal_distribution(Mat &&mean, Cov &&cov_or_inv,
                                        bool from_precision);
  auto &mean() const { return mean_; }

  auto &cov() const { return cov_; }

  auto &cov_inv() const { return cov_inv_; }
  auto &cholesky() const { return cho_; }

  auto operator()(std::mt19937_64 &mt) {
    auto z = random_matrix_normal(mt, mean().nrows(), mean().ncols);
    return cholesky() * z;
  }
};

template <class Mat, class Cov>
  requires(contains_value<Mat &&, Matrix<double>> &&
           contains_value<Cov &&, Covariance<double>>)
Maybe_error<multivariate_normal_distribution<double>>
make_multivariate_normal_distribution(Mat &&mean, Cov &&cov_or_inv,
                                      bool from_precision) {
  return_error<multivariate_normal_distribution<double>> Error{
      "make_multivariate_normal_distribution"};
  if (!is_valid(mean))
    return Error(get_error(mean));
  else if (!is_valid(cov_or_inv))
    return Error(get_error(cov_or_inv));
  else if (!from_precision) {
    auto &cov = get_value(cov_or_inv);
    auto chol = cholesky(cov);

    if (chol) {
      auto inv = inv_from_chol(chol.value());
      if (inv)
      {
          auto meanbeta=get_value(mean);
        return multivariate_normal_distribution<double>(
              std::move(meanbeta), std::move(cov), std::move(chol.value()),
            std::move(inv.value()));
      }
      else
        return Error(inv.error() + " covariance cannot be inverted");
    } else
      return Error(chol.error() +
                   " cholesky fails to build a normal distribution");
  } else {
    auto &cov_inv = cov_or_inv;
    auto chol_inv = cholesky(cov_inv);

    if (chol_inv) {
      auto chol = inv(chol_inv.value());
      if (chol) {
        auto cov = XXT(chol.value());
        auto beta_mean=get_value(mean);
        return multivariate_normal_distribution<double>(
            std::move(beta_mean), std::move(cov), std::move(chol.value()),
            std::move(cov_inv));
      } else
        return Error(chol.error() + " cholesky cannot be inverted");
    } else
      return Error(chol_inv.error() +
                   " cholesky fails to be built from precision");
  }
}

#endif // MULTIVARIATE_NORMAL_DISTRIBUTION_H
