#include "driver/naive_driver.hpp"
#include "driver/performance_helper.hpp"
#include "matrix/matrix_helper.hpp"

#include <iostream>
typedef struct
{
    int m;
    int n;
    int p;
    int lda;
    int ldb;
    int ldc;
} MultConstants;
template <class T>
void init_data(cudaError_t& status, const Matrix<T>& mat_a, const Matrix<T>& mat_b, T*& a, T*& b, T*& results, T*& d_a, T*& d_b, T*& d_results, MultConstants*& d_multinfo )
{    
    cudaError_t cuda_result;
    a = (T*)malloc(mat_a.size() * sizeof(T));
    b = (T*)malloc(mat_b.size() * sizeof(T));
    results = (T*)malloc(mat_a.m_size() * mat_b.n_size() * sizeof(T));
    MatrixHelper::copy_data<T>(mat_a, a);
    MatrixHelper::copy_data<T>(mat_b, b);
    int result_m = mat_a.m_size();
    int result_n = mat_b.n_size();
    MultConstants* multinfo = new MultConstants();

    multinfo->m = mat_a.m_size();
    multinfo->n = mat_a.n_size();
    multinfo->p = mat_b.n_size();
    multinfo->lda = mat_a.m_size();
    multinfo->ldb = mat_b.m_size();
    multinfo->ldc = mat_a.m_size();

    cuda_result = cudaMalloc((void**)&d_a, mat_a.size() * sizeof(T));
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to allocate matrix a: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    cuda_result = cudaMalloc((void**)&d_b, mat_b.size() * sizeof(T));
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to allocate matrix b: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    cuda_result = cudaMalloc((void**)&d_results, result_m * result_n * sizeof(T));
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to allocate matrix d_cuda_results: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    cuda_result = cudaMalloc((void**)&d_multinfo, sizeof(MultConstants));
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to allocate matrix d_multinfo: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }

    cuda_result = cudaMemcpy( d_a, a, mat_a.size() * sizeof(T), cudaMemcpyHostToDevice);
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to copy mat a: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }

    cuda_result = cudaMemcpy( d_b, b, mat_b.size() * sizeof(T), cudaMemcpyHostToDevice); 
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to copy mat b: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }

    cuda_result = cudaMemcpy( d_multinfo, multinfo, sizeof(MultConstants), cudaMemcpyHostToDevice); 
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to copy mult info: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    free(multinfo); 
}

template <class T>
void free_data(T*& a, T*& b, T*& results, T*& d_a, T*& d_b, T*& d_results, MultConstants*& d_multinfo )
{    
    free(a);
    free(b);
    free(results);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_results);
    cudaFree(d_multinfo);
}

__global__
void i_naive_multiply( const MultConstants* const d_multinfo,  const int * d_a, const int* d_b, int* d_results )
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (row < d_multinfo->m && col < d_multinfo->n) 
    {
        float accumulator = 0;
        for (int k = 0; k < d_multinfo->n; ++k) 
        {
            accumulator += d_a[row + k * d_multinfo->lda] * d_b[k + col * d_multinfo->ldb];
        }

        d_results[row + col * d_multinfo->ldc] = accumulator + d_results[row + col * d_multinfo->ldc];
    }

}

__global__
void s_naive_multiply( const MultConstants* const d_multinfo,  const short * d_a, const short* d_b, short* d_results )
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (row < d_multinfo->m && col < d_multinfo->n) 
    {
        float accumulator = 0;
        for (int k = 0; k < d_multinfo->n; ++k) 
        {
            accumulator += d_a[row + k * d_multinfo->lda] * d_b[k + col * d_multinfo->ldb];
        }

        d_results[row + col * d_multinfo->ldc] = accumulator + d_results[row + col * d_multinfo->ldc];
    }

}

__global__
void d_naive_multiply( const MultConstants* const d_multinfo,  const double * d_a, const double* d_b, double* d_results )
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (row < d_multinfo->m && col < d_multinfo->n) 
    {
        float accumulator = 0;
        for (int k = 0; k < d_multinfo->n; ++k) 
        {
            accumulator += d_a[row + k * d_multinfo->lda] * d_b[k + col * d_multinfo->ldb];
        }

        d_results[row + col * d_multinfo->ldc] = accumulator + d_results[row + col * d_multinfo->ldc];
    }

}


__global__
void f_naive_multiply( const MultConstants* const d_multinfo,  const float * d_a, const float* d_b, float* d_results )
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (row < d_multinfo->m && col < d_multinfo->n) 
    {
        float accumulator = 0;
        for (int k = 0; k < d_multinfo->n; ++k) 
        {
            //printf("[%d,%d] += %f * %f\n",row,col,d_a[row + k * d_multinfo->lda],d_b[k + col * d_multinfo->ldb]);
            accumulator += d_a[row + k * d_multinfo->lda] * d_b[k + col * d_multinfo->ldb];
        }

        d_results[row + col * d_multinfo->ldc] = accumulator + d_results[row + col * d_multinfo->ldc];
    }

}

template <class T>
void _naive_multiply( const MultConstants* const d_multinfo,  const T * d_a, const T* d_b, T* d_results, const dim3& block, const dim3& grid )
{
    std::cout << "NOT IMPLEMENTED!\n";
    exit(EXIT_FAILURE);
}

template <>
void _naive_multiply<int>( const MultConstants* const d_multinfo,  const int * d_a, const int* d_b, int* d_results, const dim3& block, const dim3& grid )
{
    i_naive_multiply<<< grid, block >>>( d_multinfo, d_a, d_b, d_results ); 
}

template <>
void _naive_multiply<short>( const MultConstants* const d_multinfo,  const short* d_a, const short* d_b, short* d_results, const dim3& block, const dim3& grid )
{
    s_naive_multiply<<< grid, block >>>( d_multinfo, d_a, d_b, d_results ); 
}

template <>
void _naive_multiply<double>( const MultConstants* const d_multinfo,  const double* d_a, const double* d_b, double* d_results, const dim3& block, const dim3& grid )
{
    d_naive_multiply<<< grid, block >>>( d_multinfo, d_a, d_b, d_results ); 
}

template <>
void _naive_multiply<float>( const MultConstants* const d_multinfo,  const float * d_a, const float* d_b, float* d_results, const dim3& block, const dim3& grid )
{
    f_naive_multiply<<< grid, block >>>( d_multinfo, d_a, d_b, d_results ); 
}

template <class T>
void multiply( const Matrix<T> mat_a, const Matrix<T> mat_b )
{
    cudaError_t status;

    //status = cublasInit();
    T* a; 
    T* b;
    T* results;
    T* d_a;
    T* d_b;
    T* d_results;
    MultConstants* d_multinfo;

    int result_m = mat_a.m_size();
    int result_n = mat_b.n_size();

    init_data<T>(
            status,         // Cublas Status
            mat_a,          // Matrix A
            mat_b,          // Matrix B
            a,              // Pointer to matrix A values
            b,              // Pointer to matrix B values
            results,        // Pointer to multiplication results
            d_a,            // Device pointer to matrix A values 
            d_b,            // Device pointer to matrix B values
            d_results,      // Device pointer to multiplication results
            d_multinfo);       
#ifdef DEBUG
    std::cout << "A: M" << mat_a.m_size() << "\n";
    std::cout << "A: N" << mat_a.n_size() << "\n";
    std::cout << "B: M" << mat_b.m_size() << "\n";
    std::cout << "B: N" << mat_b.n_size() << "\n";
#endif
    
    dim3 block(16, 16);
    dim3 grid(
        ( result_n + block.x - 1 ) / block.x + 1,
        ( result_m + block.y - 1 ) / block.y + 1
    );
   
    auto start = get_clock_time();
    _naive_multiply<T>( d_multinfo, d_a, d_b, d_results, block, grid );
    auto stop = get_clock_time();
    
    cudaError_t cuda_result = cudaMemcpy( results, d_results, result_m * result_n * sizeof(T), cudaMemcpyDeviceToHost);
    if ( cuda_result != cudaSuccess )
    {
        std::cerr << "Failed to copy result post multiply: \n" << cudaGetErrorString(cuda_result) << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << get_duration_seconds(start, stop) << "\n";
    //MatrixHelper::print_matrix<T>(Orientation::ROW_MAJOR, result_m, result_n, results);
    //MatrixHelper::print_matrix<T>(Orientation::COLUMN_MAJOR, result_m, result_n, results);

    free_data<T>(
            a,
            b,
            results,
            d_a,
            d_b,
            d_results,
            d_multinfo);   
}



template <class T>
void NaiveDriver<T>::multiply_matrices()
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

template class NaiveDriver<int>;
template class NaiveDriver<short>;
template class NaiveDriver<double>;
template class NaiveDriver<float>;
