#ifndef INDEXES_H
#define INDEXES_H

#include "list.h"
#include <functional>
#include <vector>







template<Identifier Id>
class Index
{
private:
    std::size_t size_;
public:
};

template<Identifier Id, Object T>
class IndexedVector
{
private:
    std::vector<T> data_;
public:
    template<class F>
    requires std::convertible_to<std::invoke_result_t<F,std::size_t>,T>
    friend IndexedVector operator|(const Index<Id>& id,F&&f)
    {
        std::vector<T> out;
        out.reserve(size(id));
        for (std::size_t i=0;  i<size(id); ++i)
        {
            out.emplace(std::invoke(std::forward<F>(f),i));
        }
        return out;
    }

};







#endif // INDEXES_H
