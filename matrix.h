#ifndef MATRIX_H
#define MATRIX_H

#include "maybe_error.h"
#include <cassert>
#include <cstddef>
#include <memory>

template <class> class Matrix;

namespace lapack {
Matrix<double> Lapack_Full_Product(const Matrix<double> &x,
                                   const Matrix<double> &y, bool transpose_x,
                                   bool transpose_y, double alpha = 1.0,
                                   double beta = 0.0);

std::pair<Matrix<double>, Matrix<double>> Lapack_QR(const Matrix<double> &x);
Maybe_error<Matrix<double>> Lapack_Full_inv(const Matrix<double> &a);

} // namespace lapack

template <class T> class DiagonalMatrix;

template <class T> class Matrix {
private:
  std::size_t size_ = 0;
  std::size_t nrows_ = 0;
  std::size_t ncols_ = 0;
  T *x_ = nullptr;

public:
  explicit Matrix(std::size_t _nrows, std::size_t _ncols,
                  bool initialize = true)
      : size_{_nrows * _ncols}, nrows_{_nrows}, ncols_{_ncols},
        x_{initialize ? new T[size_]() : new T[size_]} {}
  explicit Matrix(std::size_t _nrows, std::size_t _ncols, T value)
      : size_{_nrows * _ncols}, nrows_{_nrows}, ncols_{_ncols},
        x_{new T[size_]} {
    for (std::size_t i = 0; i < size_; ++i)
      x_[i] = value;
  }

  Matrix(const Matrix &x)
      : size_{x.size()}, nrows_{x.nrows()}, ncols_{x.ncols()},
        x_{new T[nrows_ * ncols_]} {
    for (std::size_t i = 0; i < size_; ++i)
      x_[i] = x[i];
  }
  Matrix(Matrix &&x)
      : size_{x.size()}, nrows_{x.nrows()}, ncols_{x.ncols()}, x_{x.x_} {
    x.x_ = nullptr;
  }

  friend class DiagonalMatrix<T>;

  Matrix &operator=(const Matrix &x) {
    if (size() != x.size()) {
      size_ = x.size();
      nrows_ = x.nrows();
      ncols_ = x.ncols();
      delete[] x_;
      x_ = new T[x.size()];
    }

    for (std::size_t i = 0; i < size_; ++i)
      x_[i] = x[i];
    return *this;
  }

  auto &operator[](std::size_t i) { return x_[i]; }
  auto &operator[](std::size_t i) const { return x_[i]; }
  auto &operator()(std::size_t i, std::size_t j) { return x_[i * ncols_ + j]; }
  auto &operator()(std::size_t i, std::size_t j) const {
    return x_[i * ncols_ + j];
  }
  auto ncols() const { return ncols_; }
  auto nrows() const { return nrows_; }
  auto size() const { return size_; }
  ~Matrix() { delete[] x_; }

  friend auto operator*(const Matrix &a, const Matrix &b) {
    return lapack::Lapack_Full_Product(a, b, false, false);
  }

  friend auto inv(const Matrix &a) { return lapack::Lapack_Full_inv(a); }

  friend auto tr(const Matrix &a) {
    Matrix out(a.ncols(), a.nrows(), false);
    for (std::size_t i = 0; i < out.nrows(); ++i)
      for (std::size_t j = 0; j < out.ncols(); ++j)
        out(i, j) = a(j, i);
    return out;
  }

  template <class F> friend auto apply(F &&f, Matrix &&x) {
    for (std::size_t i = 0; i < x.size(); ++i)
      x[i] = f(x[i]);
    return x;
  }

  template <class F> friend auto reduce(F &&f, const Matrix &x) {
    auto cum = x[0];
    for (std::size_t i = 1; i < x.size(); ++i)
      cum = f(cum, x[i]);
    return cum;
  }

  template <class F> friend auto reduce_ij(F &&f, const Matrix &x, T init) {
    auto cum = init;
    for (std::size_t i = 0; i < x.nrows(); ++i)
      for (std::size_t j = 0; j < x.ncols(); ++j)
        cum = f(cum, i,j,x(i,j));
    return cum;
  }


  friend auto &operator<<(std::ostream &os, const Matrix &x) {
    os << "\n";
    for (std::size_t i = 0; i < x.nrows(); ++i) {
      for (std::size_t j = 0; j < x.ncols(); ++j)
        os << x(i, j) << "\t";
      os << "\n";
    }
    return os;
  }
};

template <class T> class DiagonalMatrix {
private:
  std::size_t size_ = 0;
  std::size_t nrows_ = 0;
  std::size_t ncols_ = 0;
  T *x_ = nullptr;

public:
  explicit DiagonalMatrix(std::size_t _nrows, std::size_t _ncols,
                          bool initialize = true)
      : size_{std::min(_nrows, _ncols)}, nrows_{_nrows}, ncols_{_ncols},
        x_{initialize ? new T[size_]() : new T[size_]} {}
  explicit DiagonalMatrix(std::size_t _nrows, std::size_t _ncols, T value)
      : size_{std::min(_nrows, _ncols)}, nrows_{_nrows}, ncols_{_ncols},
        x_{new T[size_]} {
    for (std::size_t i = 0; i < size_; ++i)
      x_[i] = value;
  }
  explicit DiagonalMatrix(std::initializer_list<T>&& a):
      size_{a.size()}, nrows_{a.size()}, ncols_{a.size()},
      x_{ new T[size_]} {
    std::copy(a.begin(),a.end(),x_);
  }

  explicit DiagonalMatrix(const Matrix<T> &a)
      : size_{(a.ncols() == 1 || a.nrows() == 1)
                  ? a.size()
                  : std::min(a.nrows(), a.ncols())},
        nrows_{(a.ncols() == 1 || a.nrows() == 1) ? a.size() : a.nrows()},
        ncols_{(a.ncols() == 1 || a.nrows() == 1) ? a.size() : a.ncols()},
        x_{new T[size_]} {
    if (a.ncols() == 1 || a.nrows() == 1) {
      for (std::size_t i = 0; i < size(); ++i)
        (*this)[i] = a[i];
    } else {
      for (std::size_t i = 0; i < size(); ++i)
        (*this)[i] = a(i, i);
    }
  }

  explicit DiagonalMatrix(Matrix<T> &&a)
      : size_{(a.ncols() == 1 || a.nrows() == 1)
                  ? a.size()
                  : std::min(a.nrows(), a.ncols())},
        nrows_{(a.ncols() == 1 || a.nrows() == 1) ? a.size() : a.nrows()},
        ncols_{(a.ncols() == 1 || a.nrows() == 1) ? a.size() : a.ncols()},
        x_{(a.ncols() == 1 || a.nrows() == 1) ? nullptr : new T[size_]} {
    if (a.ncols() == 1 || a.nrows() == 1) {
      std::swap(x_, a.x_);

    } else {
      for (std::size_t i = 0; i < size(); ++i)
        (*this)[i] = a(i, i);
    }
  }

  DiagonalMatrix(const DiagonalMatrix &x)
      : size_{x.size()}, nrows_{x.nrows()}, ncols_{x.ncols()},
        x_{new T[x.size()]} {
    for (std::size_t i = 0; i < size_; ++i)
      x_[i] = x[i];
  }
  DiagonalMatrix(DiagonalMatrix &&x)
      : size_{x.size()}, nrows_{x.nrows()}, ncols_{x.ncols()}, x_{x.x_} {
    x.x_ = nullptr;
  }

  DiagonalMatrix &operator=(const DiagonalMatrix &x) {
    if (size() != x.size()) {
      size_ = x.size();
      nrows_ = x.nrows();
      ncols_ = x.ncols();
      delete[] x_;
      x.x_ = new T[x.size()];
    }

    for (std::size_t i = 0; i < size_; ++i)
      x_[i] = x[i];
    return *this;
  }

  DiagonalMatrix &operator=(DiagonalMatrix &&x) {
    size_ = x.size();
    nrows_ = x.nrows();
    ncols_ = x.ncols();
    std::swap(x_, x.x_);
    return *this;
  }

  auto &operator[](std::size_t i) { return x_[i]; }
  auto &operator[](std::size_t i) const { return x_[i]; }
  auto operator()(std::size_t i, std::size_t j) const {
    if (i == j)
      return x_[i];
    else
      return T{};
  }

  auto ncols() const { return ncols_; }
  auto nrows() const { return nrows_; }
  auto size() const { return size_; }
  ~DiagonalMatrix() { delete[] x_; }

  friend auto operator*(const DiagonalMatrix &a, const Matrix<double> &b) {
    assert(a.ncols() == b.nrows() && "matrix product dimensions mismatch");
    Matrix<double> out(a.nrows(), b.ncols(), false);
    for (std::size_t i = 0; i < a.size(); ++i)
      for (std::size_t j = 0; j < out.ncols(); ++j)
        out(i, j) = a[i] * b(i, j);
    for (std::size_t i = a.size(); i < out.nrows(); ++i)
      for (std::size_t j = 0; j < out.ncols(); ++j)
        out(i, j) = 0;
    return out;
  }

  friend auto operator*(const Matrix<double> &a, const DiagonalMatrix &b) {
    assert(a.ncols() == b.nrows() && "matrix product dimensions mismatch");
    Matrix<double> out(a.nrows(), b.ncols(), false);
    for (std::size_t i = 0; i < out.nrows(); ++i) {
      for (std::size_t j = 0; j < b.size(); ++j)
        out(i, j) = a(i, j) * b[j];
      for (std::size_t j = b.size(); j < out.ncols(); ++j)
        out(i, j) = 0;
    }
    return out;
  }

  friend auto operator*(const DiagonalMatrix &a, const DiagonalMatrix &b) {
    assert(a.ncols() == b.nrows() && "matrix product dimensions mismatch");
    auto out = DiagonalMatrix(a.nrows(), b.ncols());
    for (std::size_t i = 0; i < out.size(); ++i)
      out[i] = a[i] * b[i];
    return out;
  }

  friend auto tr(const DiagonalMatrix &a) { return a; }

  friend auto diag(const DiagonalMatrix &a) {
    Matrix out(a.size(), 1, false);
    for (std::size_t i = 0; i < out.size(); ++i)
      out[i] = a[i];
    return out;
  }

  template <class F> friend auto apply(F &&f, DiagonalMatrix &&x) {
    for (std::size_t i = 0; i < x.size(); ++i)
      x[i] = f(x[i]);
    return x;
  }

  template <class F> friend auto reduce(F &&f, const DiagonalMatrix &x) {
    auto cum = x[0];
    for (std::size_t i = 1; i < x.size(); ++i)
      cum = f(cum, x[i]);
    return cum;
  }

  friend auto &operator<<(std::ostream &os, const DiagonalMatrix &x) {
    os << "Diagonal matrix "
       << "nrows: " << x.nrows() << "ncols: " << x.ncols() << "\n";
    for (std::size_t i = 0; i < x.size(); ++i)
      os << x[i] << "\t";
    os << "\n";

    return os;
  }
};

template <class T> auto diag(const Matrix<T> &a) {
  return DiagonalMatrix<T>(a);
}
template <class T> auto diag(Matrix<T> &&a) {
  return DiagonalMatrix<T>(std::move(a));
}

template <class T> auto diag(std::initializer_list<T> &&a) {
  return DiagonalMatrix<T>(std::move(a));
}



template <class T> auto qr(const Matrix<T> &a) { return lapack::Lapack_QR(a); }

#endif // MATRIX_H
