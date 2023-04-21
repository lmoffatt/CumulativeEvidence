#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include "matrix.h"
#include <concepts>
#include <functional>
#include <random>
#include <utility>

template <class Distribution>
concept is_Distribution = requires(Distribution &m, Distribution const& m_const) {
    {
        m(std::declval<std::mt19937_64 &>())
    } ;

    {
        m_const.logP(m(std::declval<std::mt19937_64 &>()))
    }->std::convertible_to<double>;

};

template <class Distribution, class T>
concept is_Distribution_of = requires(Distribution &m, Distribution const& m_const) {
    {
        m(std::declval<std::mt19937_64 &>())
    } ->std::convertible_to<T>;

    {
        m_const.logP(m(std::declval<std::mt19937_64 &>()))
    }->std::convertible_to<double>;
};

template<class T>
concept has_size= requires(const T& a)
{
    {a.size()}->std::convertible_to<std::size_t>;
};

template<class T>
concept index_accesible=requires(T a)
{
    {a[std::size_t{}]}->std::same_as<double&>;
};

template<class T>
requires (index_accesible<T>)
auto& get_at(T& v, std::size_t i)
{
    return v[i];
}

template<class T>
auto get_at(T& v, std::size_t i)->std::decay_t<decltype(v[i])>
{
    return v[i];
}
double get_at(double const& x,std::size_t i){
    assert(i==0);
    return x;
}


double& get_at(double& x,std::size_t i){
    assert(i==0);
    return x;
}

template<class T>
requires (has_size<T>)
std::size_t size(const T& x) {return x.size();}

constexpr std::size_t size(double){return 1;}


template<class F, class T>
auto operator|(F&& f, T&& x )->std::invoke_result_t<F,T>
{
    return std::invoke(std::forward<F>(f),std::forward<T>(x));
}


auto& append_to(Matrix<double>& m, std::size_t )
{
    return m;
}

template<class X,class ...Xs>
auto& append_to(Matrix<double>& m, std::size_t i, X&& x, Xs&&...xs)
{
    for (std::size_t n=0; n<size(x); ++n)
        m[i+n]=get_at(x,n);
    return append_to(m,i+size(x),std::forward<Xs>(xs)...);
}


template<class ...Xs>
auto concatenate_to_columns(Xs&&...xs)
{
    auto n=(size(xs)+...);
    auto out=Matrix<double>(1,n);
    std::size_t i=0;
    out=append_to(out,i,std::forward<Xs>(xs)...);
    return out;
}


template<class Dist>
concept Multivariate=requires (Dist& d)
{
    {d(std::declval<std::mt19937_64&>())}->std::convertible_to<Matrix<double>>;
};

double logP_impl(const Matrix<double>& , std::size_t, double partial_logP)
{
    return partial_logP;
}
template<class Dist, class... Ds>
requires(Multivariate<Dist>)
double logP_impl(const Matrix<double>& x, std::size_t ipos, double partial_logP, const Dist& d, const Ds&...ds)
{
    auto n=d.size();
    auto out=Matrix<double>(1,size(d));
    for (std::size_t i=0; i<n; ++i)
        out[i]=x[ipos+i];
    auto logPi=d.logP(out);
    return logP_impl(x,ipos+n,partial_logP+logPi,ds...);
}
template<class Dist, class... Ds>
requires(!Multivariate<Dist>)
 double logP_impl(const Matrix<double>& x, std::size_t ipos, double partial_logP, const Dist& d, const Ds&...ds)
{
    auto logPi=d.logP(x[ipos]);
    return logP_impl(x,ipos+1,partial_logP+logPi,ds...);
}






template<class... ds>
requires(is_Distribution<ds>&&...)
class distributions: public ds...
{
public:

    auto operator()(std::mt19937_64& mt)
    {
        return concatenate_to_columns(ds::operator()(mt)...);
    }

    double logP(const Matrix<double>& x)const
    {
        return logP_impl(x,0,0.0,static_cast<ds const&>(*this)...);
    }

    explicit distributions(ds&&... d): ds{std::move(d)}...{}
    explicit distributions(ds const &... d): ds{d}...{}
};

#endif // DISTRIBUTIONS_H
