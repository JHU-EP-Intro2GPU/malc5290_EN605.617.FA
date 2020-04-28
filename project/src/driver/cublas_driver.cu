#include "driver/cublas_driver.hpp"
#include "matrix/matrix_helper.hpp"
#include <iostream>
#include <cublas.h>

template <class T>
void copy_data( const Matrix<T>& data, T*& copy )
{
    int index = 0;
    for ( T val : data.matrix() )
    {
        copy[index] = val;
        index++;
#ifdef DEBUG
        std::cout << "COPYING: " << val << "\n";
#endif
    }
}

template <class T>
void init_data(cublasStatus& status, const Matrix<T>& mat_a, const Matrix<T>& mat_b, T*& a, T*& b, T*& results, T*& cublas_a, T*& cublas_b, T*& cublas_results )
{    
    a = (T*)malloc(mat_a.size() * sizeof(T));
    b = (T*)malloc(mat_b.size() * sizeof(T));
    results = (T*)malloc(mat_a.m_size() * mat_b.n_size() * sizeof(T));
    copy_data<T>(mat_a, a);
    copy_data<T>(mat_b, b);
    int result_m = mat_a.m_size();
    int result_n = mat_b.n_size();

    status = cublasAlloc(mat_a.size(), sizeof(T), (void**)&cublas_a);
    if (status != CUBLAS_STATUS_SUCCESS)
    {   
      exit(EXIT_FAILURE);
    }   

    status = cublasAlloc(mat_b.size(), sizeof(T), (void**)&cublas_b);
    if (status != CUBLAS_STATUS_SUCCESS)
    {   
      exit(EXIT_FAILURE);
    }   

    status = cublasAlloc(result_m * result_n, sizeof(T), (void**)&cublas_results);
    if (status != CUBLAS_STATUS_SUCCESS)
    {   
      exit(EXIT_FAILURE);
    }   


    status = cublasSetMatrix( mat_a.m_size(), mat_a.n_size(), sizeof(T), a, mat_a.m_size(), cublas_a, mat_a.m_size() );
    if (status != CUBLAS_STATUS_SUCCESS)
    {   
      exit(EXIT_FAILURE);
    }   

    status = cublasSetMatrix( mat_b.m_size(), mat_b.n_size(), sizeof(T), b, mat_b.m_size(), cublas_b, mat_b.m_size() );
    if (status != CUBLAS_STATUS_SUCCESS)
    {   
      exit(EXIT_FAILURE);
    }
}

template <class T>
void free_data(T*& a, T*& b, T*& results, T*& cublas_a, T*& cublas_b, T*& cublas_results )
{    
    free(a);
    free(b);
    free(results);
    cudaFree(cublas_a);
    cudaFree(cublas_b);
    cudaFree(cublas_results);

}

// Cublas does not support integer or float types for Gausean matrix multiplication
// So we need to specialize.
template <class T>
void multiply( const Matrix<T> mat_a, const Matrix<T> mat_b )
{
    std::cerr << "Unspported type for CuBlas\n";
    exit(1);
}

// Float specialization for multiply
template <>
void multiply<float>( const Matrix<float> mat_a, const Matrix<float> mat_b )
{
    cublasStatus status;

    status = cublasInit();
    float* a; 
    float* b;
    float* results;
    float* cublas_a;
    float* cublas_b;
    float* cublas_results;

    int result_m = mat_a.m_size();
    int result_n = mat_b.n_size();

    init_data<float>(
            status,                // Cublas Status
            mat_a,                 // Matrix A
            mat_b,                 // Matrix B
            a,                     // Pointer to matrix A values
            b,                     // Pointer to matrix B values
            results,               // Pointer to multiplication results
            cublas_a,              // Device pointer to matrix A values 
            cublas_b,              // Device pointer to matrix B values
            cublas_results);       // Device pointer to multiplication results
#ifdef DEBUG
    std::cout << "A: M" << mat_a.m_size() << "\n";
    std::cout << "A: N" << mat_a.n_size() << "\n";
    std::cout << "B: M" << mat_b.m_size() << "\n";
    std::cout << "B: N" << mat_b.n_size() << "\n";
#endif
    cublasSgemm(
            'n',                  // Normal
            'n',                  // Normal
            result_m,             // Row size of matrix A ( aka results m )
            result_n,             // Column size of matrix B ( aka results n )
            mat_a.n_size(),       // Column size of A and row Size of B
            1,                    // No scaling for multiplication
            cublas_a,             // Device pointer to matrix A
            mat_a.m_size(),       // Row size of matrix A ( Leading dimension )
            cublas_b,             // Device pointer to matrix B
            mat_b.m_size(),       // Row size of matrix B ( Leading dimension )
            1,                    // No scaling of result matrix
            cublas_results,       // Device pointer to multiplication results
            result_n);            // Row size of matrix C ( Leading dimension )
      
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
  
    cublasGetMatrix(result_m, result_n, sizeof(float), cublas_results, result_m, results, result_m);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
    
    std::cout << "Results!\n";
    MatrixHelper::print_matrix<float>(Orientation::COLUMN_MAJOR, result_m, result_n, results);
    free_data<float>(
            a,
            b,
            results,
            cublas_a,
            cublas_b,
            cublas_results);
}

// Double specialization for multiply
template <>
void multiply<double>( const Matrix<double> mat_a, const Matrix<double> mat_b )
{
    cublasStatus status;

    status = cublasInit();
    double* a; 
    double* b;
    double* results;
    double* cublas_a;
    double* cublas_b;
    double* cublas_results;

    int result_m = mat_a.m_size();
    int result_n = mat_b.n_size();

    init_data<double>(
            status,                // Cublas Status
            mat_a,                 // Matrix A
            mat_b,                 // Matrix B
            a,                     // Pointer to matrix A values
            b,                     // Pointer to matrix B values
            results,               // Pointer to multiplication results
            cublas_a,              // Device pointer to matrix A values 
            cublas_b,              // Device pointer to matrix B values
            cublas_results);       // Device pointer to multiplication results
    
#ifdef DEBUG
    std::cout << "A: M" << mat_a.m_size() << "\n";
    std::cout << "A: N" << mat_a.n_size() << "\n";
    std::cout << "B: M" << mat_b.m_size() << "\n";
    std::cout << "B: N" << mat_b.n_size() << "\n";
#endif

    cublasDgemm(
            'n',                  // Normal
            'n',                  // Normal
            result_m,             // Row size of matrix A ( aka results m )
            result_n,             // Column size of matrix B ( aka results n )
            mat_a.n_size(),       // Column size of A and row Size of B
            1,                    // No scaling for multiplication
            cublas_a,             // Device pointer to matrix A
            mat_a.m_size(),       // Row size of matrix A ( Leading dimension )
            cublas_b,             // Device pointer to matrix B
            mat_b.m_size(),       // Row size of matrix B ( Leading dimension )
            1,                    // No scaling of result matrix
            cublas_results,       // Device pointer to multiplication results
            result_n);            // Row size of matrix C ( Leading dimension )
      
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
  
    cublasGetMatrix(result_m, result_n, sizeof(double), cublas_results, result_m, results, result_m);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
    
    std::cout << "Results!\n";
    for ( int i = 0; i < result_m * result_n; i++ )
    {
        std::cout << results[i] << "\n";
    }

    free_data<double>(
            a,
            b,
            results,
            cublas_a,
            cublas_b,
            cublas_results);
}

template <class T>
void CublasDriver<T>::multiply_matrices()
{
    std::cout << "CUBLAS DRIVER: MULTIPLY_MATRICES\n";
    if ( MatrixDriver<T>::_mat_a.orientation() == Orientation::ROW_MAJOR )
    {
        MatrixHelper::change_orientation(MatrixDriver<T>::_mat_a,Orientation::COLUMN_MAJOR); 
    }
    if ( MatrixDriver<T>::_mat_b.orientation() == Orientation::ROW_MAJOR )
    {
        MatrixHelper::change_orientation(MatrixDriver<T>::_mat_b,Orientation::COLUMN_MAJOR); 
    }
    std::cout << "ORIENTATION CHANGED TO COLUMN MAJOR FOR CUBLAS\n";
    multiply<T>(MatrixDriver<T>::_mat_a, MatrixDriver<T>::_mat_b);
}

template class CublasDriver<int>;
template class CublasDriver<short>;
template class CublasDriver<double>;
template class CublasDriver<float>;
