#include "driver/temp_driver.hpp"

#include <iostream>

template <class T>
void TempDriver<T>::multiply_matrices()
{
    std::cout << "CUBLAS DRIVER: MULTIPLY_MATRICES\n";
}

template class TempDriver<int>;
template class TempDriver<short>;
template class TempDriver<double>;
template class TempDriver<float>;
/*
void __register_types__()
{
    CublasDriver<int> intType;
    CublasDriver<float> floatType;
    CublasDriver<double> doubleType;
    CublasDriver<short> shortType;
}
*/
