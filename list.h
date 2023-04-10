#ifndef LIST_H
#define LIST_H


#include <type_traits>
#include <utility>

template <typename C,template<typename...>class V> struct is_of_this_template_type : std::false_type {};
template <template<typename...>class V, typename... Ts> struct is_of_this_template_type< V<Ts...>,V> : std::true_type {};
template <typename C,template<typename...>class V> inline constexpr bool is_of_this_template_type_v = is_of_this_template_type<C,V>::value;


template<class Id>
concept Identifier= std::is_empty_v<Id>;


template<class Id>
concept Object= !std::is_empty_v<Id>;



template<Identifier IdT, class T>
requires Object<T>
struct m
{
    T value;
    using Id=IdT;
    m(Id,T v):value{v}{}
    auto& operator[](Id){return *this;}
    auto& operator[](Id)const {return *this;}
    auto& operator()(){return value;}
    auto& operator()()const{return value;}
};





template<class M>
concept is_Member=is_of_this_template_type_v<M,m>;


template<class... ms>
requires (is_Member<ms>&&...)
struct l:public ms...
{
    using ms::operator[]...;
    l(ms...m):ms{m}...{}

    template<class... ns>
    requires (is_Member<ns>&&...)
    friend l<ms...,ns...> operator &(l&& x, l<ns...>&& y)
    {
        return {std::move(x[typename ms::Id{}])...,std::move(x[typename ns::Id{}])...};
    }

    template<class... ns>
    requires (is_Member<ns>&&...)
    friend l<ms...,ns...> c(l&& x, l<ns...>&& y)
    {
        return {std::move(x[typename ms::Id{}])...,std::move(x[typename ns::Id{}])...};
    }

    template<class Id, class T>
    friend l<ms...,m<Id,T>> c(l&& x, m<Id,T>&& y)
    {
        return {std::move(x[typename ms::Id{}])...,std::move(y)};
    }

};



template<class L>
concept is_List=is_of_this_template_type_v<L,l>;

template<class L, class field>
concept includes_this=std::is_base_of_v<field,L>;




#endif // LIST_H
