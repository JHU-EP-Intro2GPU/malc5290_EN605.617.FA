#ifndef en605_617_temp_driver_hpp
#define en605_617_temp_driver_hpp
#include <vector>
#include "driver/MatrixDriver.hpp"

template <class T>
class TempDriver: public MatrixDriver<T>
{
public:
    TempDriver(){}
    TempDriver( const Matrix<T>& mat_a, const Matrix<T>& mat_b )
        : MatrixDriver<T>( mat_a, mat_b ){}
    virtual ~TempDriver(){}
    virtual void multiply_matrices();
protected:
};
#endif //en605_617_temp_driver_hpp
