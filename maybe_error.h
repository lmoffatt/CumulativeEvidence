#ifndef MAYBE_ERROR_H
#define MAYBE_ERROR_H

#include <string>
#include <variant>

template<auto F>
std::string function_name();

template<class T>
class Maybe_error: private std::variant<T,std::string>
{
public:
    using base_type=std::variant<T,std::string>;
    Maybe_error(T&& x):base_type(std::move(x)){}
    Maybe_error(const T& x):base_type(x){}

    Maybe_error(std::string&& x):base_type{(std::move(x))}{}
    Maybe_error(const std::string& x):base_type{x}{}
    constexpr explicit operator bool() const noexcept
    {
        return this->index()==0;
    }
    constexpr const T& value() const& noexcept
    {
        return std::get<0>(*this);
    }
    constexpr T& value() & noexcept
    {
        return std::get<0>(*this);
    }
    constexpr const T&& value() const&& noexcept
    {
        return std::get<0>(std::move(*this));
    }
    constexpr T&& value() && noexcept
    {
        return std::get<0>(std::move(*this));

    }
    auto&  error()const {return std::get<1>(*this);}
    auto&  error() {return std::get<1>(*this);}



};


template<auto F, class T>
struct return_error{
    Maybe_error<T> operator()(std::string&& error){ return Maybe_error<T>(function_name<F>()+": "+std::move(error));}
};



#endif // MAYBE_ERROR_H
