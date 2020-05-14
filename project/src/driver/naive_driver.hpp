#ifndef en605_617_naive_driver_hpp
#define en605_617_naive_driver_hpp
#include <vector>
#include "driver/MatrixDriver.hpp"

template <class T>
class NaiveDriver: public MatrixDriver<T>
{
public:
    NaiveDriver(){}
    NaiveDriver( const Matrix<T>& mat_a, const Matrix<T>& mat_b )
        : MatrixDriver<T>( mat_a, mat_b ){}
    virtual ~NaiveDriver(){}
    virtual void multiply_matrices();
protected:
};
#endif //en605_617_naive_driver_hpp
