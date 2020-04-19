#ifndef en605_617_matrix_hpp
#define en605_617_matrix_hpp
#include <vector>
template <class T>
class Matrix
{
public:
    Matrix( const int& m, const int& n, const std::vector<T>& matrix_values )
        : m( m ), n( n ), matrix_values( matrix_values ){}
    int m_size() const {return m;}
    int n_size() const {return n;}
    int size() const {return m * n;}
    std::vector<T> matrix(){ return matrix_values; }

private:
    int m;
    int n;
    std::vector<T> matrix_values;
};

#endif //en605_617_matrix_hpp
