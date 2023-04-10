#ifndef RANGE_H
#define RANGE_H
#include "list.h"
#include <functional>
#include <vector>

class range
{
private:

    std::size_t start_;
    std::size_t end_;
    std::size_t step_;
public:
    range(std::size_t end):start_{0},end_{end},step_{1}{}
    range(std::size_t start,std::size_t end):start_{start},end_{end},step_{1}{}
    range(std::size_t start,std::size_t end,std::size_t step):start_{start},end_{end},step_{step}{}
    std::size_t size()const {
        if (end_>start_)
            return (end_-start_)/step_;
        else
            return 0;
    }

    template<class F>
    friend auto operator|(range&& x, F&& f)
    {
        auto out =std::vector<std::invoke_result_t<F,std::size_t>>(x.size());
            for (std::size_t i=x.start_; i<x.end_; i+=x.step_)
            out[i]=std::invoke(std::forward<F>(f),i);
        return out;
    }

};






#endif // RANGE_H
