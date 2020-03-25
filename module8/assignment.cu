#include "assignment.h"

#include <chrono>
#include <fstream>
#include <stdio.h>
#include <string>
#include <iostream>
#include <math.h>
#include <random>

#include <cublas.h>
#include <curand.h>
#include <curand_kernel.h>

typedef struct {
    int a;
    int b;
} MatricesStruct;

typedef struct {
    int add;
    int sub;
    int mul;
    int div;
} MatrixResultsStruct;

static std::string TestInput = "I am a test string to be encrypted";
static int InputSize = TestInput.length(); 
static int seed = 5;

__constant__ const int D_MAT_MULT_MAX = 5;

__device__
void d_encrypt( char* text, const int* const shift, const int& index )
{
    if ( index % 2 == 0 )
        text[index] += shift[index];
    else
        text[index] -= shift[index];
}

__device__
void d_decrypt( char* text, const int* const shift, const int& index )
{
    if ( index % 2 == 0 )
        text[index] -= shift[index];
    else
        text[index] += shift[index];
}

__global__
void init( int seed, curandState_t* states)
{
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__
void randomsI(curandState_t* states,  int* numbers) 
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  numbers[index] = curand(&states[index]) % 25;
}

__global__
void randoms(curandState_t* states,  float* numbers) 
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
  numbers[index] = curand(&states[index]) % D_MAT_MULT_MAX;
}

__global__
void encrypt( char* text, const int* const shift)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    d_encrypt( text, shift, index );
}

__global__
void decrypt( char* text, const int* const shift)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    d_decrypt( text, shift, index );
}


__host__
int get_duration_ns( const std::chrono::time_point<std::chrono::high_resolution_clock>& start,
                     const std::chrono::time_point<std::chrono::high_resolution_clock>& end )
{
    return int(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

__host__
std::chrono::time_point<std::chrono::high_resolution_clock> get_clock_time()
{
    return std::chrono::high_resolution_clock::now();
}

// Helper function for cleaning up allocated memory used by math functionality.
__host__ 
void cleanup_data( 
    MatricesStruct*& data, MatricesStruct*& d_data,
    MatrixResultsStruct*& results, MatrixResultsStruct*& d_results)
{
    cudaFree(d_data);
    cudaFree(d_results);
    
    cudaFreeHost(data);
    cudaFreeHost(results);
}

__host__
void init_encryption_data( curandState_t*& d_states, char*& results, char*& text,
        char*& d_text, int*& d_shift, const std::size_t& charSize,
        const std::size_t& intSize, const std::string& input )
{
    results = (char*)malloc(charSize);
    text = (char*)malloc(charSize);
    
    for( int i = 0; i < InputSize; i++ ) text[i] = input[i];
    cudaMalloc((void**) &d_states, InputSize * sizeof(curandState_t));
    cudaMalloc((void**) &d_shift, intSize);
    cudaMalloc((void**) &d_text, charSize);

    init<<<InputSize, 1>>>(seed, d_states);
    randomsI<<<InputSize, 1>>>(d_states, d_shift);
    
    cudaMemcpy(d_text, text, charSize, cudaMemcpyHostToDevice);
}   

__host__
void init_encryption( curandState_t*& d_states, char*& results, char*& text,
        char*& d_text, int*& d_shift, const std::size_t& charSize,
        const std::size_t& intSize)
{
    init_encryption_data( d_states, results, text, d_text, d_shift, charSize, intSize,
                          TestInput);
}

__host__
void init_decryption( curandState_t*& d_states, char*& results, char*& text,
        char*& d_text, int*& d_shift, const std::size_t& charSize,
        const std::size_t& intSize, const std::string& encrypted_text)
{
    init_encryption_data( d_states, results, text, d_text, d_shift, charSize, intSize,
                          encrypted_text);
}


__host__
void cleanup_encryption_data( curandState_t*& d_states, char*& results, char*& text,
                              char*& d_text, int*& d_shift )
{
    free(text);
    free(results);
    cudaFree(d_shift);
    cudaFree(d_text);
    cudaFree(d_states);
}   
    
__host__
std::string execute_curand_encryption()
{
    curandState_t* d_states;
    char* results;
    int* d_shift;
    char* d_text;
    char* text;
    
    const std::size_t charSize = InputSize * sizeof(char);
    const std::size_t intSize = InputSize * sizeof(int); 

    init_encryption( d_states, results, text, d_text, d_shift, charSize, intSize );
    encrypt<<<InputSize, 1>>>(d_text, d_shift);
    
    cudaMemcpy(results, d_text, charSize, cudaMemcpyDeviceToHost);

    std::string encrypted(results);
    cleanup_encryption_data( d_states, results, text, d_text, d_shift );
    return encrypted;
}

__host__
std::string execute_curand_decryption(std::string input)
{
    curandState_t* d_states;
    char* results;
    int* d_shift;
    char* d_text;
    char* text;
    
    const std::size_t charSize = InputSize * sizeof(char);
    const std::size_t intSize = InputSize * sizeof(int); 
    
    init_decryption( d_states, results, text, d_text, d_shift, charSize, intSize, input );
    
    decrypt<<<InputSize, 1>>>(d_text, d_shift);

    cudaMemcpy(results, d_text, charSize, cudaMemcpyDeviceToHost);

    std::string decrypted(results);
    cleanup_encryption_data( d_states, results, text, d_text, d_shift);
    return decrypted;
}
__host__ 
void init_matops_data( curandState_t*& d_states, float*& d_mat, float*& mat_a, float *& mat_b,
                       float*& mat_c, float*& cublas_mat_a, float*& cublas_mat_b, float*& cublas_mat_c,
                       const int& matSize, const int& matDimSize, const int& blockSize )
{
    cublasStatus status;

    mat_a = (float*)malloc(matSize * sizeof(float));
    mat_b = (float*)malloc(matSize * sizeof(float));
    mat_c = (float*)malloc(matSize * sizeof(float));


    cublasInit();
    cudaMalloc((void**) &d_mat, matSize * sizeof( float));
    cudaMalloc((void**) &d_states, matSize * sizeof(curandState_t));
    init<<<matSize, blockSize>>>(time(0), d_states);
    randoms<<<matSize, blockSize>>>(d_states, d_mat);    
    cudaMemcpy(mat_a, d_mat, matSize * sizeof( float), cudaMemcpyDeviceToHost);
 
    randoms<<<matSize, blockSize>>>(d_states, d_mat);    
    cudaMemcpy(mat_b, d_mat, matSize * sizeof( float), cudaMemcpyDeviceToHost);
    
    printf("POST RANDOM\n");
   
    status = cublasAlloc(matSize, sizeof(float), (void**)&cublas_mat_a);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
  
    status = cublasAlloc(matSize, sizeof(float), (void**)&cublas_mat_b);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }

    status = cublasAlloc(matSize, sizeof(float), (void**)&cublas_mat_c);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }


    status = cublasSetMatrix( matDimSize, matDimSize, sizeof(float), mat_a, matDimSize, cublas_mat_a, matDimSize );
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }

    status = cublasSetMatrix( matDimSize, matDimSize, sizeof(float), mat_b, matDimSize, cublas_mat_b, matDimSize );
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }

}

__host__
void cleanup_matops( curandState_t*& d_states, float*& d_mat, float *& mat_a, float *& mat_b,
                     float*& mat_c, float*& cublas_mat_a, float*& cublas_mat_b, float*& cublas_mat_c )
{
    free(mat_a);
    free(mat_b);
    free(mat_c);
    cudaFree(d_states);
    cudaFree(d_mat);
    cublasFree(cublas_mat_a);
    cublasFree(cublas_mat_b);
    cublasFree(cublas_mat_c);
    cublasShutdown();
}

__host__
void execute_matops( const int& matDimSize, const int& blockSize, const std::string& outputName)
{
    int matSize = matDimSize * matDimSize;
    curandState_t* d_states;
    cublasStatus status;
    float* d_mat;
    float* mat_a;
    float* mat_b;
    float* mat_c;
    float* cublas_mat_a;
    float* cublas_mat_b;
    float* cublas_mat_c;
    

    init_matops_data( d_states, d_mat, mat_a, mat_b, mat_c, cublas_mat_a, cublas_mat_b,
                      cublas_mat_c, matSize, matDimSize, blockSize );
    auto startTime = get_clock_time();

    cublasSgemm( 
            'n',
            'n', 
            matDimSize, 
            matDimSize, 
            matDimSize, 
            1, 
            cublas_mat_a, 
            matDimSize, 
            cublas_mat_b, 
            matDimSize, 
            1, 
            cublas_mat_c, 
            matDimSize);
    auto stopTime = get_clock_time();

    auto duration = get_duration_ns(startTime, stopTime);
    printf("%d x %d Matrix took %d ns\n", matDimSize, matDimSize, duration);
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
    
    cublasGetMatrix(matDimSize, matDimSize, sizeof(float), cublas_mat_c, matDimSize, mat_c, matDimSize);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
#if PRINT_MATRIX      
    printf("MAT A\n");
    for ( int i = 0; i < matSize; i++ )
        printf("%f\n", mat_a[i]);
    
    printf("MAT B\n");
    for ( int i = 0; i < matSize; i++ )
        printf("%f\n", mat_b[i]);
    
    printf("MAT C\n");
    for ( int i = 0; i < matSize; i++ )
        printf("%f\n", mat_c[i]);
#endif
    cleanup_matops( d_states, d_mat, mat_a, mat_b, mat_c, cublas_mat_a, cublas_mat_b,
                    cublas_mat_c );
}

int main(int argc, char** argv)
{
    // read command line arguments
    int matDimSize = 512;
    int totalThreads = 512 * 512;
    int blockSize = 1;
    std::string outputName;
    
    if (argc >= 2) 
    {
        matDimSize = atoi(argv[1]);
        totalThreads = matDimSize * matDimSize;
    }
    if (argc >= 3)
    {
        outputName = argv[2];
    }

    std::cout << "cuRand encryption test:\n";
    std::cout << "Original string: \"" << TestInput << "\"\n";
    std::string encrypted = execute_curand_encryption();
    std::cout << "ENCRYPTED: " << encrypted << "\n";
    std::string decrypted = execute_curand_decryption(encrypted);
    std::cout << "DECRYPTED: " << decrypted << "\n";
    std::cout << "##################################################################\n";
    printf("%d x %d matix multiplication (%d elements)\n", matDimSize, matDimSize, totalThreads);
    execute_matops( matDimSize, blockSize, outputName);
}
