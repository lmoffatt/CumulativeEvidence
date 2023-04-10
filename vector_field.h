#ifndef VECTOR_FIELD_H
#define VECTOR_FIELD_H
#include <algorithm>
#include <random>
#include <vector>
#include<cassert>

template <class T>
concept has_size = requires(T a) {
    { a.size() } -> std::convertible_to<std::size_t>;

};

template<class V, class T>
concept indexed_access= requires (V a)
{
    { a[std::size_t{}] } -> std::convertible_to<T>;
};

template<class V, class T>
concept indexed_multiplication= requires (V a)
{
    { a[std::size_t{}] * double{} } -> std::convertible_to<T>;
    { double{} * a[std::size_t{}]} -> std::convertible_to<T>;
};

template<class V, class T>
concept indexed_sum= requires (V a)
{
    { a[std::size_t{}] + a[std::size_t{}] } -> std::convertible_to<T>;
};


template<class V, class T>
concept vector_field= has_size<T>&&indexed_access<V,T>&&indexed_sum<V,T>&&indexed_multiplication<V,T>;


template<class T>
auto operator+(vector_field<T>  auto const& x,vector_field<T>  auto const& y)
{
    assert((x.size()==y.size())&& "sum of vector fields of different sizes");
    auto out=x;
    for (std::size_t i=0; i<x.size(); ++i)
        out[i]=x[i]+y[i];
    return out;
}

template<class T>
auto operator*(vector_field<T>  auto const& x,double a)
{
    auto out=x;
    for (std::size_t i=0; i<x.size(); ++i)
        out[i]=x[i]*a;
    return out;
}

template<class T>
auto operator*(double a,vector_field<T>  auto const& x)
{
    auto out=x;
    for (std::size_t i=0; i<x.size(); ++i)
        out[i]=a*x[i];
    return out;
}



#endif // VECTOR_FIELD_H
