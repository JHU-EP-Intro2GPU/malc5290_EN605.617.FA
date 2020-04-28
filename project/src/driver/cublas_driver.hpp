#ifndef en605_617_cublas_driver_hpp
#define en605_617_cublas_driver_hpp
#include <vector>
#include "driver/MatrixDriver.hpp"

template <class T>
class CublasDriver: public MatrixDriver<T>
{
public:
    CublasDriver(){}
    CublasDriver( const Matrix<T>& mat_a, const Matrix<T>& mat_b )
        : MatrixDriver<T>( mat_a, mat_b ){}
    virtual ~CublasDriver(){}
    virtual void multiply_matrices();
protected:
};
#endif //en605_617_cublas_driver_hpp
