#ifndef en605_617_matrix_driver_hpp
#define en605_617_matrix_driver_hpp
#include "matrix/Matrix.hpp"

#include <vector>
#include <iostream>
template<class T>
class MatrixDriver
{
public:
    MatrixDriver(){}
    MatrixDriver( const Matrix<T>& mat_a, const Matrix<T>& mat_b )
        : _mat_a( mat_a )
        , _mat_b( mat_b ){}

    virtual ~MatrixDriver(){}
    virtual void multiply_matrices() = 0;
protected:
    Matrix<T> _mat_a;
    Matrix<T> _mat_b;
};
#endif //en605_617_matrix_driver_hpp
