#ifndef en605_617_cutlass_driver_hpp
#define en605_617_cutlass_driver_hpp
#include <vector>
#include "driver/MatrixDriver.hpp"

template <class T>
class CutlassDriver: public MatrixDriver<T>
{
public:
    CutlassDriver(){}
    CutlassDriver( const Matrix<T>& mat_a, const Matrix<T>& mat_b )
        : MatrixDriver<T>( mat_a, mat_b ){}
    virtual ~CutlassDriver(){}
    virtual void multiply_matrices();
protected:
};
#endif //en605_617_cutlass_driver_hpp
