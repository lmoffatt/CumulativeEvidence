#ifndef BAYESIAN_LINEAR_REGRESSION_H
#define BAYESIAN_LINEAR_REGRESSION_H



#include "matrix.h"
#include "multivariate_normal_distribution.h"
#include <cmath>
auto bayesian_linear_regression(const multivariate_normal_distribution<double>& prior,
                                double prior_eps_df, double prior_eps_variance,
                                const Matrix<double>& y, const Matrix<double>& X)
{
    auto L_0=prior.cov_inv()*   prior_eps_variance;
    auto SSx= XTX(X);
    auto n=y.nrows();
    auto a_0=prior_eps_df/2.0;
    auto b_0=prior_eps_df*prior_eps_variance/2.0;
    auto beta_0=prior.mean();
   // auto beta_ml=inv(SSx)*tr(X)*y;
    auto L_n=SSx+L_0;
    auto beta_n=inv(L_n)*(tr(X)*y+(L_0*prior.mean()));

    auto posterior=make_multivariate_normal_distribution(beta_n,L_n,true);
    auto a_n= a_0+n/2;
    auto b_n=b_0+0.5*(xtx(y)+xtAx(beta_0,L_0)-xtAx(beta_n,L_n));
    //auto E_n=std::pow(2*std::numbers::pi,-n/2)*std::sqrt(det(L_0)/det(L_n))*std::pow(b_0,a_0)/std::pow(b_n,a_n)*std::tgamma(a_n)/std::tgamma(a_0);
    auto logE_n=0.5*(-n*std::log(std::numbers::pi)+logdet(L_0)-logdet(L_n)+a_0*log(b_0)-a_n*log(b_n)+std::lgamma(a_n)-std::lgamma(a_0));


}



#endif // BAYESIAN_LINEAR_REGRESSION_H
