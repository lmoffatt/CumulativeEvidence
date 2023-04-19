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
    auto cov=random_covariance(mt,5,{200,1,1,1,0.005});

    auto X=random_matrix_normal(mt,5,5);
    std::cout<<"X"<<X;
    auto [Q,R]=qr(X);
    std::cout<<"Q"<<Q;
    std::cout<<"R"<<R;





    std::cout << "Hello World!" << std::endl;
    std::cout<<"cov"<<cov;

    auto covInv=inv(cov);
    std::cout<<"covInv"<<covInv.value();
    std::cout<<"cov*covInv=\n"<<cov*covInv.value();

    auto cho=cholesky(cov);
    std::cout<<"cholesky"<<cho.value();
    std::cout<<"cholesky tr"<<tr(cho.value());

    std::cout<<"cholesky squared"<<tr(cho.value())*cho.value();
    std::cout<<"cholesky test"<<tr(cho.value())*cho.value()-cov;




    if (covInv)
        std::cout<<"cov inv="<<covInv.value();
    else
        std::cout<<"cov inv="<<covInv.error();



    auto corr=correlation_matrix(cov);
    std::cout<<"corr"<<corr;
    auto corrInv=correlation_matrix(covInv.value());
    std::cout<<"corrInv"<<corrInv;

    std::cout<<"max correlation    \t"<<reduce_ij([](auto one, auto i, auto j, auto two){return ((i!=j)&(std::abs(two)>std::abs(one)))?two:one; },corr,0)<<"\n";

    std::cout<<"max correlation Inv\t"<<reduce_ij([](auto one, auto i, auto j, auto two){return ((i!=j)&(std::abs(two)>std::abs(one)))?two:one; },corrInv,0)<<"\n";

    return 0;
}
