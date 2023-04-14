#include "matrix_random.h"
#include "mcmc.h"
#include "lapack_headers.h"
#include <iostream>

//using namespace std;

int main()
{
    auto initseed=0;
    auto mt=init_mt(initseed);
    auto cov=random_covariance(mt,5,{200,1,1,1,1});




    std::cout << "Hello World!" << std::endl;
    std::cout<<cov;
    auto covInv=inv(cov);
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
