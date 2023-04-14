#ifndef QR_FACTORIZATION_H
#define QR_FACTORIZATION_H
#include "matrix.h"






Matrix<double> Lapack_Full_Product(const Matrix<double> &x,
                                   const Matrix<double> &y,
                                   bool transpose_x, bool transpose_y,
                                   double alpha , double beta) {

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





#endif // QR_FACTORIZATION_H
