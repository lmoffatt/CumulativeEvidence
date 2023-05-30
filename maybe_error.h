#ifndef MAYBE_ERROR_H
#define MAYBE_ERROR_H

#include <cmath>
#include <string>
#include <tuple>
#include <variant>
#include <vector>
template <typename C, template <typename...> class V>
struct is_of_this_template_type : std::false_type {};
template <template <typename...> class V, typename... Ts>
struct is_of_this_template_type<V<Ts...>, V> : std::true_type {};
template <typename C, template <typename...> class V>
inline constexpr bool is_of_this_template_type_v =
    is_of_this_template_type<C, V>::value;






template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;




template <auto F> std::string function_name();

template <class T> class Maybe_error;

template <class T>
concept is_Maybe_error = is_of_this_template_type_v<T, Maybe_error>;





template <class T>
struct make_Maybe_error{
    using type=std::conditional_t<is_Maybe_error<T>,T,Maybe_error<T>>;
};

template<class T>
using Maybe_error_t=typename make_Maybe_error<T>::type;

template <class T> constexpr bool is_valid(const Maybe_error<T> &v) {
    return v.valid();
}

template<class T>
requires(!is_Maybe_error<T>)
 constexpr bool is_valid(const T&) { return true; }


template <class T>
  requires(!is_Maybe_error<T>)
auto &get_value(const T &x) {
  return x;
}

template <class T> auto &get_value(const Maybe_error<T> &x) {
  return x.value();
}

template <class T> T get_value( Maybe_error<T> &&x) {
  return std::move(x.value());
}


template<class C, class T>
concept contains_value= std::convertible_to<decltype(get_value(std::declval<C>())),T>;




template <class T>
  requires(!is_Maybe_error<T>)
auto get_error(const T &) {
  return std::string("");
}

template <class T> auto get_error(const Maybe_error<T> &x) { return x.error(); }

template <class T> class Maybe_error : private std::variant<T, std::string> {
public:
  using base_type = std::variant<T, std::string>;
  Maybe_error(T &&x) : base_type(std::move(x)) {}
  Maybe_error(const T &x) : base_type(x) {}
  Maybe_error()=default;
  Maybe_error(char* error): base_type{error}{}
  Maybe_error(std::string &&x) : base_type{(std::move(x))} {}
  Maybe_error(const std::string &x) : base_type{x} {}
  constexpr explicit operator bool() const noexcept {
    return this->index() == 0;
  }
  constexpr  bool valid() const noexcept {
    return this->index() == 0;
  }

  constexpr const T &value() const & noexcept { return std::get<0>(*this); }
  constexpr T &value() & noexcept { return std::get<0>(*this); }
  constexpr const T &&value() const && noexcept {
    return std::get<0>(std::move(*this));
  }
  constexpr T &&value() && noexcept { return std::get<0>(std::move(*this)); }
  std::string error() const {
    if (*this)
      return "";
    else
      return std::get<1>(*this);
  }


  friend std::ostream& operator<<(std::ostream &os, const Maybe_error& x)
  {
    if (x)
      os<<x.value();
    else
      os<<x.error();
    return os;
  }
  friend std::ostream& operator<<(std::ostream &os,  Maybe_error&& x)
  {
    if (x)
      os<<x.value();
    else
      os<<x.error();
    return os;
  }
};

template <class T, class S>
  requires(is_Maybe_error<T> || is_Maybe_error<S>)
auto operator*(const T &x, const S &y) {
  if (is_valid(x) && is_valid(y))
    return Maybe_error(get_value(x) * get_value(y));
  else
    return Maybe_error<std::decay_t<decltype(get_value(x) * get_value(y))>>(
        get_error(x) + " multiplies " + get_error(y));
}

template<class T>
Maybe_error<std::vector<T>> promote_Maybe_error(std::vector<Maybe_error<T>>const & x)
{
  std::vector<T> out(size(x));

  for (std::size_t i=0; i<size(out); ++i)
    if (x[i])
      out[i]=x[i].value();
    else return "error at "+std::to_string(i)+x[i].error();
  return out;

}

template <class T, class S>
requires(is_Maybe_error<T> || is_Maybe_error<S>)
    auto operator+(const T &x, const S &y) {
  if (is_valid(x) && is_valid(y))
    return Maybe_error(get_value(x) + get_value(y));
  else
    return Maybe_error<std::decay_t<decltype(get_value(x) + get_value(y))>>(
        get_error(x) + " multiplies " + get_error(y));
}

template <class T, class S>
requires(is_Maybe_error<T> || is_Maybe_error<S>)
    auto operator-(const T &x, const S &y) {
  if (is_valid(x) && is_valid(y))
    return Maybe_error(get_value(x) - get_value(y));
  else
    return Maybe_error<std::decay_t<decltype(get_value(x) - get_value(y))>>(
        get_error(x) + " multiplies " + get_error(y));
}

template <class T, class S>
requires(is_Maybe_error<T> || is_Maybe_error<S>)
    auto operator/(const T &x, const S &y) {
  if (is_valid(x) && is_valid(y))
    return Maybe_error(get_value(x) / get_value(y));
  else
    return Maybe_error<std::decay_t<decltype(get_value(x) / get_value(y))>>(
        get_error(x) + " divides " + get_error(y));
}


template <class T, class S>
requires(is_Maybe_error<T> || is_Maybe_error<S>)
auto xtAx(const T &x, const S &y) {
  if (is_valid(x) && is_valid(y))
    return Maybe_error(xtAx(get_value(x),get_value(y)));
  else
    return Maybe_error<std::decay_t<decltype(xtAx(get_value(x),get_value(y)))>>(
        get_error(x) + " xtAx " + get_error(y));
}

template <class T, class S>
requires(is_Maybe_error<T> || is_Maybe_error<S>)
    auto xAxt(const T &x, const S &y) {
  if (is_valid(x) && is_valid(y))
    return Maybe_error(xAxt(get_value(x),get_value(y)));
  else
    return Maybe_error<std::decay_t<decltype(xtAx(get_value(x),get_value(y)))>>(
        get_error(x) + " xAxt " + get_error(y));
}



template< class T,class... Ts>
requires(std::is_same_v<T,Ts>||...)
Maybe_error<std::variant<Ts...>> make_Maybe_variant(Maybe_error<T>&& x)
{
  if (x)
    return std::move(x);
  else
    return x.error();
}




template <class T>
requires(is_Maybe_error<T>)
auto inv(const T &x) {
  using return_type=Maybe_error_t<std::decay_t<decltype(inv(x.value()))>>;
  if (x)
    return return_type(inv(x.value()));
  else
    return return_type(x.error()+"\n inverse");
}
template <class T>
requires(is_Maybe_error<T>)
    auto diag(const T &x) {
  using return_type=Maybe_error_t<std::decay_t<decltype(diag(x.value()))>>;
  if (x)
    return return_type(diag(x.value()));
  else
    return return_type(x.error()+"\n diag");
}
template <class T>
requires(is_Maybe_error<T>)
    auto XXT(const T &x) {
  using return_type=Maybe_error_t<std::decay_t<decltype(XXT(x.value()))>>;
  if (x)
    return return_type(XXT(x.value()));
  else
    return return_type(x.error()+"\n diag");
}

template <class T>
requires(is_Maybe_error<T>)
    auto operator-(const T &x) {
  using return_type=Maybe_error_t<std::decay_t<decltype(-(x.value()))>>;
  if (x)
    return return_type(-(x.value()));
  else
    return return_type(x.error()+"\n minus");
}


template <class T>
requires(is_Maybe_error<T>)
    auto logdet(const T &x) {
  using return_type=Maybe_error_t<std::decay_t<decltype(logdet(x.value()))>>;
  if (x)
    return return_type(logdet(x.value()));
  else
    return return_type(x.error()+"\n logdet");
}

using std::log;
template <class T>
requires(is_Maybe_error<T>)
    auto log(const T &x) {
  using return_type=Maybe_error_t<std::decay_t<decltype(log(x.value()))>>;
  if (x)
    return return_type(log(x.value()));
  else
    return return_type(x.error()+"\n log");
}


template <class T>
requires(is_Maybe_error<T>)
 auto cholesky(const T &x) {
  using return_type=Maybe_error_t<std::decay_t<decltype(cholesky(x.value()))>>;
  if (x)
    return return_type(cholesky(x.value()));
  else
    return return_type(x.error()+"\n log");
}
using std::sqrt;
template <class T>
requires(is_Maybe_error<T>)
    auto sqrt(const T &x) {
  using return_type=Maybe_error_t<std::decay_t<decltype(sqrt(x.value()))>>;
  if (x)
    return return_type(sqrt(x.value()));
  else
    return return_type(x.error()+"\n sqrt");
}

template <class T>
requires(is_Maybe_error<T>)
    auto tr(const T &x) {
  using return_type=Maybe_error_t<std::decay_t<decltype(tr(x.value()))>>;
  if (x)
    return return_type(tr(x.value()));
  else
    return return_type(x.error()+"\n transpose");
}

template <class T>
    requires(is_Maybe_error<T>)
auto Trace(const T &x) {
  using return_type=Maybe_error_t<std::decay_t<decltype(Trace(x.value()))>>;
  if (x)
    return return_type(Trace(x.value()));
  else
    return return_type(x.error()+"\n Trace");
}

template <class T, auto ...F> struct return_error {
  std::string function_label;
  Maybe_error<T> operator()(std::string &&error) {
    return Maybe_error<T>(function_label+(function_name<F>() +... +(": " + std::move(error))));
  }




};


template<class... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& tu)
{
  return std::apply([&os](auto&... x)->std::ostream& {((os<<x<<"\n"),...);return os; },tu);
}

template<class Ts>
std::ostream& operator<<(std::ostream& os, const std::vector<Ts>& v)
{
  for (std::size_t i=0; i<v.size(); ++i)
    os<<v[i]<<"\n";
//  os<<"\n";
  return os;
}



#endif // MAYBE_ERROR_H
