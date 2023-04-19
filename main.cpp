#include "matrix.h"
#include "matrix_random.h"
#include "mcmc.h"
#include "lapack_headers.h"
#include <iostream>
#include "bayesian_linear_regression.h"
//using namespace std;

int main()
{
    auto initseed=0;
    auto mt=init_mt(initseed);

    auto npar= 5ul;
    auto nsamples=1000ul;
    auto log10_std_par =2.0;

    auto mean_mean_par=0.0;
    auto std_mean_par=10.0;

    auto mean_b=0.0;
    auto std_b=10.0;

    auto par_stds=apply([] (auto& x) {return std::pow(10,x);},random_matrix_normal(mt,1,npar,0,log10_std_par));
    auto cov_par=random_covariance(mt,npar,par_stds);

    auto mean_par=random_matrix_normal(mt,1,npar,mean_mean_par,std_mean_par);

    auto b=random_matrix_normal(mt,npar,1,mean_b,std_b);

    std::cerr<<"b proposed"<<b;


    auto par_dist_=make_multivariate_normal_distribution(std::move(mean_par),std::move(cov_par));

    auto par_dist=std::move(par_dist_.value());
    auto X=par_dist(mt,nsamples);
    auto y=X*b;

    auto prior=make_multivariate_normal_distribution(Matrix<double>(1ul,npar,mean_b),IdM<double>(npar));

    std::cout<<"mean"<<Matrix<double>(1ul,npar,mean_b);
    std::cout<<"cov"<<IdM<double>(npar);


    std::cout<<"prior"<<prior;
    double prior_eps_df =1.0;
    double prior_eps_variance =1.0;
    if (prior.valid())
    {
        auto reg=bayesian_linear_regression(prior.value(),prior_eps_df,prior_eps_variance,y,X);
        std::cout<<"bayes\n"<<reg;
    }
   // std::cout<<y;

    return 0;
}
