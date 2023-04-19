#ifndef MULTIVARIATE_NORMAL_DISTRIBUTION_H
#define MULTIVARIATE_NORMAL_DISTRIBUTION_H

#include "matrix.h"
#include <random>


template <typename T, typename C>
concept Covariance = std::convertible_to<C, DiagPosDetMatrix<T>> ||
                     std::convertible_to<C, SymPosDefMatrix<T>>;

template <typename T, class Cova>
  requires Covariance<T, Cova>
class multivariate_normal_distribution {
private:
  std::normal_distribution<T> n_;
  Matrix<T> mean_;
  Cova cov_;
  using cholesky_type = std::decay_t<decltype(get_value(::cholesky(cov_)))>;
  cholesky_type cho_;
  Cova cov_inv_;

  static double calc_logdet(const cholesky_type &cho) {
    return std::visit([](auto const &m) { return 2 * logdet(m); }, cho);
  }

  multivariate_normal_distribution(Matrix<T> &&mean, Cova &&cov,
                                   cholesky_type &&chol, Cova &&cov_inv)
      : n_{}, mean_{std::move(mean)}, cov_{std::move(cov)},
        cho_{std::move(chol)}, cov_inv_{std::move(cov_inv)} {}

public:
  template <
      class Mat, class Cov,
      typename Te ,//= std::decay_t<decltype(get_value(std::declval<Mat>())(0, 0))>,
      class Covae> // = std::decay_t<decltype(get_value(std::declval<Cov>()))>>
    requires Covariance<Te, Covae> && contains_value<Mat &&, Matrix<Te>> &&
             contains_value<Cov &&, Covae>
  friend Maybe_error<multivariate_normal_distribution<Te, Covae>>
  make_multivariate_normal_distribution(Mat &&mean, Cov &&cov);

    template <
        class Mat, class Cov,
        typename Te ,//= std::decay_t<decltype(get_value(std::declval<Mat>())(0, 0))>,
        class Covae> // = std::decay_t<decltype(get_value(std::declval<Cov>()))>>
    requires Covariance<Te, Covae> && contains_value<Mat &&, Matrix<Te>> &&
        contains_value<Cov &&, Covae>
        friend Maybe_error<multivariate_normal_distribution<Te, Covae>>
        make_multivariate_normal_distribution_from_precision(Mat &&mean, Cov &&cov_inv);

  auto &mean() const { return mean_; }

  auto &cov() const { return cov_; }

  auto &cov_inv() const { return cov_inv_; }
  auto &cholesky() const { return cho_; }

  auto operator()(std::mt19937_64 &mt) {
    auto z = random_matrix_normal(mt, mean().nrows(), mean().ncols());
    return cholesky() * z;
  }
  auto operator()(std::mt19937_64 &mt, std::size_t n) {
    auto out = Matrix<T>(n, mean_.ncols());
    for (std::size_t i = 0; i < n; ++i) {
      auto z = random_matrix_normal(mt, mean().nrows(), mean().ncols());
      z = cholesky() * z;
      for (std::size_t j = 0; j < out.ncols(); ++j)
        out(i, j) = z[j];
    }
    return out;
  }

  friend std::ostream& operator<<(std::ostream& os, const multivariate_normal_distribution& m)
  {
    os<<"mean "<<m.mean();
    os<<"cov "<<m.cov();
    return os;

  }
};

template <
    class Mat, class Cov,
    typename T = std::decay_t<decltype(get_value(std::declval<Mat>())(0, 0))>,
    class Cova = std::decay_t<decltype(get_value(std::declval<Cov>()))>>
  requires Covariance<T, Cova> && contains_value<Mat &&, Matrix<T>> &&
           contains_value<Cov &&, Cova>
Maybe_error<multivariate_normal_distribution<T, Cova>>
make_multivariate_normal_distribution(Mat &&mean, Cov &&cov) {
  return_error<multivariate_normal_distribution<T, Cova>> Error{
      "make_multivariate_normal_distribution"};
  if (!is_valid(mean))
    return Error(get_error(mean));
  else if (!is_valid(cov))
    return Error(get_error(cov));
  else  {
    auto beta_cov = get_value(std::forward<Cov>(cov));
    auto chol = cholesky(beta_cov);

    if (chol) {
      auto inv = inv_from_chol(chol.value());
      if (inv) {
        auto meanbeta = get_value(std::forward<Mat>(mean));
        return multivariate_normal_distribution<T, Cova>(
            std::move(meanbeta), std::move(cov), std::move(chol.value()),
            std::move(inv.value()));
      } else
        return Error(inv.error() + " covariance cannot be inverted");
    } else
      return Error(chol.error() +
                   " cholesky fails to build a normal distribution");
  }
}


template <
    class Mat, class Cov,
    typename T = std::decay_t<decltype(get_value(std::declval<Mat>())(0, 0))>,
    class Cova = std::decay_t<decltype(get_value(std::declval<Cov>()))>>
requires Covariance<T, Cova> && contains_value<Mat &&, Matrix<T>> &&
    contains_value<Cov &&, Cova>
        Maybe_error<multivariate_normal_distribution<T, Cova>>
        make_multivariate_normal_distribution_from_precision(Mat &&mean, Cov &&cov_inv) {
  return_error<multivariate_normal_distribution<T, Cova>> Error{
                                                                "make_multivariate_normal_distribution"};
  if (!is_valid(mean))
    return Error(get_error(mean));
  else if (!is_valid(cov_inv))
    return Error(get_error(cov_inv));
  else  {
    auto chol_inv = cholesky(cov_inv);

    if (chol_inv) {
      auto chol = inv(chol_inv.value());
      if (chol) {
        auto cov = XTX(chol.value());
        auto beta_mean = get_value(std::forward<Mat>(mean));
        auto beta_cov_inv=get_value(std::forward<Cov>(cov_inv));
        return multivariate_normal_distribution<T, Cova>(
            std::move(beta_mean), std::move(cov), std::move(chol.value()),
            std::move(beta_cov_inv));
      } else
        return Error(chol.error() + " cholesky cannot be inverted");
    } else
      return Error(chol_inv.error() +
                   " cholesky fails to be built from precision");
  }
}




#endif // MULTIVARIATE_NORMAL_DISTRIBUTION_H
