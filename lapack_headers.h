#ifndef LAPACK_HEADERS_H
#define LAPACK_HEADERS_H

#include "matrix.h"
#include "maybe_error.h"

namespace lapack {

extern "C" void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
                       double *ALPHA, double *A, int *LDA, double *B, int *LDB,
                       double *BETA, double *C, int *LDC);

extern "C" void dgeqrf_(int *M, int *N, double *A, int *LDA, double *TAU,
                        double *WORK, int *LWORK, int *INFO);

extern "C" void dorgqr_(int *M, int *N, int *K, double *A, int *LDA,
                        double *TAU, double *WORK, int *LWORK, int *INFO);

auto Lapack_UT(const Matrix<double> &x) {
  Matrix<double> out(x.nrows(), x.ncols(), false);
  for (std::size_t i = 0; i < out.ncols(); ++i)
    for (std::size_t j = 0; j < out.nrows(); ++j)
      out(j, i) = j >= i ? x(j, i) : 0;
  return out;
}

auto Lapack_LT(const Matrix<double> &x) {
  Matrix<double> out(x.nrows(), x.ncols(), false);
  for (std::size_t i = 0; i < x.ncols(); ++i)
    for (std::size_t j = 0; j < x.nrows(); ++j)
      out(j, i) = i >= j ? x(j, i) : 0;
  return out;
}

std::pair<Matrix<double>, Matrix<double>> Lapack_QR(const Matrix<double> &x) {
  int M = x.nrows();
  /*    [in]	M

                  M is INTEGER
                      The number of rows of the matrix A.  M >= 0.
  */

  int N = x.ncols();
  /*    [in]	N

                N is INTEGER
                The number of columns of the matrix A.  N >= 0.

  */
  auto a = tr(x);
  double &A = a[0];

  /*
  [in,out]	A

            A is DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).
*/

  int LDA = M;
  /*

  [in]	LDA

            LDA is INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

*/
  Matrix<double> tau(std::min(M, N), 1, false);
  double &TAU = tau[0];

  /*
  [out]	TAU

            TAU is DOUBLE PRECISION array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

*/

  Matrix<double> work(N * M, 1, false);
  double &WORK = work[0];
  int LWORK = N * M;
  /*
  [out]	WORK

            WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

*/
  /*
  [in]	LWORK

            LWORK is INTEGER
            The dimension of the array WORK.
            LWORK >= 1, if MIN(M,N) = 0, and LWORK >= N, otherwise.
            For optimum performance LWORK >= N*NB, where NB is
            the optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.
*/

  int INFO;
  /*

  [out]	INFO

            INFO is INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

*/

  dgeqrf_(&M, &N, &A, &LDA, &TAU, &WORK, &LWORK, &INFO);

  /*

Purpose:

     DGEQRF computes a QR factorization of a real M-by-N matrix A:

        A = Q * ( R ),
                ( 0 )

     where:

        Q is a M-by-M orthogonal matrix;
        R is an upper-triangular N-by-N matrix;
        0 is a (M-N)-by-N zero matrix, if M > N.

*/

  auto r = Lapack_UT(a);

  int K = std::min(N, M);

  dorgqr_(&M, &N, &K, &A, &LDA, &TAU, &WORK, &LWORK, &INFO);

  auto q = std::move(a);
  return std::pair(std::move(q), std::move(r));
}

Matrix<double> &Lapack_Full_Product(const Matrix<double> &x,
                                    const Matrix<double> &y, Matrix<double> &z,
                                    bool transpose_x, bool transpose_y,
                                    double alpha = 1.0, double beta = 0.0) {

  std::size_t cols_i, rows_e, cols_e;
  if (transpose_x) {
    rows_e = x.ncols();
    cols_i = x.nrows();
  } else {
    rows_e = x.nrows();
    cols_i = x.ncols();
  }

  if (transpose_y) {
    //  rows_i=y.ncols();
    cols_e = y.nrows();
  } else {
    //  rows_i=y.nrows();
    cols_e = y.ncols();
  }

  // assert(rows_i==cols_i);
  assert(((transpose_y ? y.ncols() : y.nrows()) == cols_i));
  // First it has to find out if the last dimension of x matches the
  // first of y
  // now we build the M_Matrix result
  if (z.nrows() != rows_e || z.ncols() != cols_e)
    z = Matrix<double>(rows_e, cols_e);

  /***  as fortran uses the reverse order for matrices and we want to
      avoid a copying operation, we calculate
          Transpose(Z)=Transpose(y)*Transpose(x)

          Transpose(matrix)=just plain matrix in C++ format


      */
  char TRANSA;
  char TRANSB;

  if (transpose_y)
    TRANSA = 'T';
  else
    TRANSA = 'N';

  if (transpose_x)
    TRANSB = 'T';
  else
    TRANSB = 'N';

  int M = cols_e;
  int N = rows_e;
  int K = cols_i;

  double ALPHA = alpha;
  double *A = const_cast<double *>(&y[0]);
  int LDA;
  if (transpose_y)
    LDA = K;
  else
    LDA = M;

  double *B = const_cast<double *>(&x[0]);

  int LDB;
  if (transpose_x)
    LDB = N;
  else
    LDB = K;

  double BETA = beta;

  double *C = &z[0];

  int LDC = M;

  try {
    dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C,
           &LDC);
  } catch (...) {
    assert(false);
  }
  return z;
}

Matrix<double> Lapack_Full_Product(const Matrix<double> &x,
                                   const Matrix<double> &y, bool transpose_x,
                                   bool transpose_y, double alpha,
                                   double beta) {
  using lapack::dgemm_;

  std::size_t cols_i, rows_e, cols_e;
  if (transpose_x) {
    rows_e = x.ncols();
    cols_i = x.nrows();
  } else {
    rows_e = x.nrows();
    cols_i = x.ncols();
  }

  if (transpose_y) {
    //  rows_i=y.ncols();
    cols_e = y.nrows();
  } else {
    //  rows_i=y.nrows();
    cols_e = y.ncols();
  }

  // assert(rows_i==cols_i);
  assert(((transpose_y ? y.ncols() : y.nrows()) == cols_i));
  // First it has to find out if the last dimension of x matches the
  // first of y
  // now we build the M_Matrix result
  Matrix<double> z(rows_e, cols_e);

  /***  as fortran uses the reverse order for matrices and we want to
      avoid a copying operation, we calculate
          Transpose(Z)=Transpose(y)*Transpose(x)

          Transpose(matrix)=just plain matrix in C++ format


      */
  char TRANSA;
  char TRANSB;

  if (transpose_y)
    TRANSA = 'T';
  else
    TRANSA = 'N';

  if (transpose_x)
    TRANSB = 'T';
  else
    TRANSB = 'N';

  int M = cols_e;
  int N = rows_e;
  int K = cols_i;

  double ALPHA = alpha;
  double *A = const_cast<double *>(&y[0]);
  int LDA;
  if (transpose_y)
    LDA = K;
  else
    LDA = M;

  double *B = const_cast<double *>(&x[0]);

  int LDB;
  if (transpose_x)
    LDB = N;
  else
    LDB = K;

  double BETA = beta;

  double *C = &z[0];

  int LDC = M;

  try {
    dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C,
           &LDC);
  } catch (...) {
    assert(false);
  }
  return z;
}


extern "C" void dgecon_(char *NORM, int *N, double *A, int *LDA, double *ANORM,
                        double *RCOND, double *WORK, int *IWORK, int *INFO);
extern "C" void dgetrf_(int *M, int *N, double *A, int *LDA, int *IPIV,
                        int *INFO);
extern "C" void dgetri_(int *n, double *B, int *dla, int *ipiv, double *work1,
                        int *lwork, int *info);
extern "C" double dlange_(char *NORM, int *M, int *N, double *A, int *LDA,
                          double *WORK);

Maybe_error<Matrix<double>> Lapack_Full_inv(const Matrix<double> &a);
}
template<> constexpr std::string function_name<&lapack::Lapack_Full_inv>(){return "Lapack_Full_inv";}
namespace lapack {


Maybe_error<Matrix<double>> Lapack_Full_inv(const Matrix<double> &a)

{
  return_error<Lapack_Full_inv,Matrix<double>> Error;
  const double min_inv_condition_number = 1e-12;
  // using Matrix_Unary_Transformations::Transpose;
  using lapack::dgecon_;
  using lapack::dgetrf_;
  using lapack::dgetri_;
  using lapack::dlange_;
  if (a.size() == 0)
    return Error("EMPTY MATRIX");
  else {
    assert(a.ncols() == a.nrows());

    char NORM = '1';
    int N = a.ncols();
    int M = N;

    int INFO = 0;
    //  char msg[101];
    auto IPIV = std::make_unique<int[]>(N);
    int LWORK;
    //        M_Matrix<double> B=Transpose(a);
    Matrix<double> B = a;
    int LDA = N;
    // A=new double[n*n];
    double *A = &B[0]; // more efficient code
    auto WORK_lange = std::make_unique<double[]>(N);

    double ANORM = dlange_(&NORM, &M, &N, A, &LDA, WORK_lange.get());

    dgetrf_(&N, &M, A, &LDA, IPIV.get(), &INFO);

    double RCOND;
    auto WORK_cond = std::make_unique<double[]>(N * 4);
    auto IWORK = std::make_unique<int[]>(N);
    int INFO_con;

    dgecon_(&NORM, &N, A, &LDA, &ANORM, &RCOND, WORK_cond.get(), IWORK.get(),
            &INFO_con);

    LWORK = N * N;
    Matrix<double> W(N, N);
    double *WORK = &W[0];

    dgetri_(&N, A, &LDA, IPIV.get(), WORK, &LWORK, &INFO);

    if (RCOND < min_inv_condition_number)
      return Error("bad condition number RCOND=" +std::to_string(RCOND));
    if (INFO == 0)
      return B;
    //  return Op({B,RCOND});
    else
      return Error("Singular Matrix on i=" + std::to_string(INFO));
    ;
  }
}

} // namespace lapack

#endif // LAPACK_HEADERS_H
