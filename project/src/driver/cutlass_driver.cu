#include "driver/cutlass_driver.hpp"
#include "driver/cutlass_helper.h"
#include "driver/performance_helper.hpp"
#include "matrix/matrix_helper.hpp"

#include "cutlass/gemm/device/gemm.h"

#include <iostream>

using ColumnMajor = cutlass::layout::ColumnMajor;
using IntGemm =
    cutlass::gemm::device::Gemm< int, ColumnMajor, int, ColumnMajor, int, ColumnMajor>;
using ShortGemm =
    cutlass::gemm::device::Gemm< short, ColumnMajor, short, ColumnMajor, short, ColumnMajor>;
using DoubleGemm =
    cutlass::gemm::device::Gemm< double, ColumnMajor, double, ColumnMajor, double, ColumnMajor>;
using FloatGemm =
    cutlass::gemm::device::Gemm< float, ColumnMajor, float, ColumnMajor, float, ColumnMajor>;

template <class T>
void init_data(cutlass::Status& status, const Matrix<T>& mat_a, const Matrix<T>& mat_b, T*& a, T*& b, T*& results, T*& cutlass_a, T*& cutlass_b, T*& cutlass_results )
{    
    cudaError_t cuda_result;
    a = (T*)malloc(mat_a.size() * sizeof(T));
    b = (T*)malloc(mat_b.size() * sizeof(T));
    results = (T*)malloc(mat_a.m_size() * mat_b.n_size() * sizeof(T));
    MatrixHelper::copy_data<T>(mat_a, a);
    MatrixHelper::copy_data<T>(mat_b, b);
    int result_m = mat_a.m_size();
    int result_n = mat_b.n_size();

    cuda_result = cudaMalloc((void**)&cutlass_a, mat_a.size() * sizeof(T));
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to allocate matrix a: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    cuda_result = cudaMalloc((void**)&cutlass_b, mat_b.size() * sizeof(T));
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to allocate matrix b: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    cuda_result = cudaMalloc((void**)&cutlass_results, result_m * result_n * sizeof(T));
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to allocate matrix cutlass_cuda_results: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }

    cuda_result = cudaMemcpy( cutlass_a, a, mat_a.size() * sizeof(T), cudaMemcpyHostToDevice);
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to copy mat a: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }

    cuda_result = cudaMemcpy( cutlass_b, b, mat_b.size() * sizeof(T), cudaMemcpyHostToDevice); 
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to copy mat b: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }


//    std::cout << "MATRIX A: \n";
//    MatrixHelper::print_matrix<float>(Orientation::COLUMN_MAJOR, mat_a.m_size(), mat_a.n_size(), (float*)a);
//    std::cout << "MATRIX B: \n";
//    MatrixHelper::print_matrix<float>(Orientation::COLUMN_MAJOR, mat_b.m_size(), mat_b.n_size(), (float*)b);
}

template <class T>
void free_data(T*& a, T*& b, T*& results, T*& cutlass_a, T*& cutlass_b, T*& cutlass_results )
{    
    free(a);
    free(b);
    free(results);
    cudaFree(cutlass_a);
    cudaFree(cutlass_b);
    cudaFree(cutlass_results);
}

template <class T>
cutlass::Status  multiply(T* a, T* b, T* results, T* cutlass_a, T* cutlass_b, T* cutlass_results, const int& result_m, const int& result_n, const int& aNSize )
{
    return cutlass::Status::kInvalid;
}

template <>
cutlass::Status multiply<double>(double* a, double* b, double* results, double* cutlass_a, double* cutlass_b, double* cutlass_results, const int& result_m, const int& result_n, const int& aNSize )
{
    DoubleGemm gemm_operator; 
    
    DoubleGemm::Arguments args({ result_m, result_n, aNSize },
                                {cutlass_a, result_m},
                                {cutlass_b, aNSize},
                                {cutlass_results, result_m},
                                {cutlass_results, result_m},
                                {1, 0});
    return gemm_operator(args);
}

template <>
cutlass::Status multiply<float>(float* a, float* b, float* results, float* cutlass_a, float* cutlass_b, float* cutlass_results, const int& result_m, const int& result_n , const int& aNSize )
{
    FloatGemm gemm_operator; 
    
    FloatGemm::Arguments args({ result_m, result_n, aNSize },
                                {cutlass_a, result_m},
                                {cutlass_b, aNSize},
                                {cutlass_results, result_m},
                                {cutlass_results, result_m},
                                {1, 0});
    return gemm_operator(args);
}

template <>
cutlass::Status multiply<int>(int* a, int* b, int* results, int* cutlass_a, int* cutlass_b, int* cutlass_results, const int& result_m, const int& result_n, const int& aNSize  )
{
    IntGemm gemm_operator; 
    
    IntGemm::Arguments args({ result_m, result_n, aNSize },
                                {cutlass_a, result_m},
                                {cutlass_b, aNSize},
                                {cutlass_results, result_m},
                                {cutlass_results, result_m},
                                {1, 0});
    return gemm_operator(args);
}

template <>
cutlass::Status multiply<short>(short* a, short* b, short* results, short* cutlass_a, short* cutlass_b, short* cutlass_results, const int& result_m, const int& result_n, const int& aNSize  )
{
    ShortGemm gemm_operator; 
    
    ShortGemm::Arguments args({ result_m, result_n, aNSize },
                                {cutlass_a, result_m},
                                {cutlass_b, aNSize},
                                {cutlass_results, result_m},
                                {cutlass_results, result_m},
                                {1, 0});
    return gemm_operator(args);
}

template <class T>
void multiply( const Matrix<T> mat_a, const Matrix<T> mat_b )
{
    cutlass::Status status;

    T* a; 
    T* b;
    T* results;
    T* cutlass_a;
    T* cutlass_b;
    T* cutlass_results;

    int result_m = mat_a.m_size();
    int result_n = mat_b.n_size();
    auto start = get_clock_time();
    init_data<T>(
            status,                // Cublas Status
            mat_a,                 // Matrix A
            mat_b,                 // Matrix B
            a,                     // Pointer to matrix A values
            b,                     // Pointer to matrix B values
            results,               // Pointer to multiplication results
            cutlass_a,              // Device pointer to matrix A values 
            cutlass_b,              // Device pointer to matrix B values
            cutlass_results);       // Device pointer to multiplication results
    auto stop = get_clock_time();
    std::cout << get_duration_seconds(start, stop) << " ";

#ifdef DEBUG
    std::cout << "A: M" << mat_a.m_size() << "\n";
    std::cout << "A: N" << mat_a.n_size() << "\n";
    std::cout << "B: M" << mat_b.m_size() << "\n";
    std::cout << "B: N" << mat_b.n_size() << "\n";
#endif
    start = get_clock_time();
    status = multiply(a, b, results, cutlass_a, cutlass_b, cutlass_results, result_m, result_n, mat_a.n_size());
    cudaDeviceSynchronize();
    stop = get_clock_time();
    if (status != cutlass::Status::kSuccess) {
        free_data<T>(
            a,
            b,
            results,
            cutlass_a,
            cutlass_b,
            cutlass_results);   
        std::cerr << "FAILED TO RUN GEMM\n";
        exit(EXIT_FAILURE);
    }
    std::cout << get_duration_seconds(start, stop) << " ";

    start = get_clock_time();
    cudaError_t cuda_result = cudaMemcpy( results, cutlass_results, result_m * result_n * sizeof(T), cudaMemcpyDeviceToHost);
    stop = get_clock_time();
    std::cout << get_duration_seconds(start, stop) << "\n";
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to copy result post gemm: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }

    //MatrixHelper::print_matrix<short>(Orientation::ROW_MAJOR, result_m, result_n, results);
    //MatrixHelper::print_matrix<T>(Orientation::COLUMN_MAJOR, result_m, result_n, results);

    free_data<T>(
            a,
            b,
            results,
            cutlass_a,
            cutlass_b,
            cutlass_results);   

}

template <class T>
void CutlassDriver<T>::multiply_matrices()
{
//    std::cout << "CUTLASS DRIVER: MULTIPLY_MATRICES\n";
    if ( MatrixDriver<T>::_mat_a.orientation() == Orientation::ROW_MAJOR )
    {
        MatrixHelper::change_orientation(MatrixDriver<T>::_mat_a,Orientation::COLUMN_MAJOR); 
    }
    if ( MatrixDriver<T>::_mat_b.orientation() == Orientation::ROW_MAJOR )
    {
        MatrixHelper::change_orientation(MatrixDriver<T>::_mat_b,Orientation::COLUMN_MAJOR); 
    }
    //std::cout << "ORIENTATION CHANGED TO COLUMN MAJOR FOR CUTLASS\n";
    multiply<T>(MatrixDriver<T>::_mat_a, MatrixDriver<T>::_mat_b);
}

template class CutlassDriver<int>;
template class CutlassDriver<short>;
template class CutlassDriver<double>;
template class CutlassDriver<float>;
