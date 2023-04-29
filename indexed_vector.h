#ifndef INDEXED_VECTOR_H
#define INDEXED_VECTOR_H



template<class vector, class... Ids>
class indexed_vector: public vector
{
public:
    using base_type=vector;
    using base_type::operator[];
};










#endif // INDEXED_VECTOR_H
