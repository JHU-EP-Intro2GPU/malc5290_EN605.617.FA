#include <chrono>
#include <fstream>
#include <stdio.h>
#include <string>
#include <iostream>
#include <math.h>
#include <random>

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

__constant__ int ScaleFactor = 5;

__device__
void d_add_matrices( const MatricesStruct& matrices, MatrixResultsStruct& results )
{
    results.add = matrices.a + matrices.b;
}

__device__
void d_sub_matrices( const MatricesStruct& matrices, MatrixResultsStruct& results )
{
    results.sub = matrices.a - matrices.b;
}

__device__
void d_mult_matrices( const MatricesStruct& matrices, MatrixResultsStruct& results )
{
    results.mul = matrices.a * ScaleFactor;
}

__device__
void d_div_matrices( const MatricesStruct& matrices, MatrixResultsStruct& results)
{
    results.div = matrices.a / ScaleFactor;
}

__global__
void matrix_ops(const MatricesStruct* const matrices, MatrixResultsStruct* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    d_add_matrices( matrices[index], results[index] );
    d_sub_matrices( matrices[index], results[index] );
    d_mult_matrices( matrices[index], results[index] );
    d_div_matrices( matrices[index], results[index] );
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

// Helper function for writing the add math results to file. 
__host__
void write_results(
    const std::string& outputName, const int& totalThreads, const int& blockSize,
    const MatrixResultsStruct* const results, const float& proc_time)
{
    int width = sqrt(totalThreads);
    std::ofstream add_stream(outputName + "_add");
    if (add_stream.is_open())
    {
        add_stream << "Matrix Add: " << totalThreads << " and Block Size: " << blockSize << "\n";
        add_stream << "matops took " << proc_time << "ns\n[ ";
        
        for( int i = 0; i < totalThreads; i++ )
        {
            add_stream << results[i].add << ",";
            if ( (i + 1) % width == 0 )
                add_stream << "\n";
        }
        add_stream << "]\n";
    }
    else{
        printf("FILE NOT OPEN?\n");
    }
    add_stream.close();

    std::ofstream sub_stream(outputName + "_sub");
    if (sub_stream.is_open())
    {
        sub_stream << "Matrix SUB: " << totalThreads << " and Block Size: " << blockSize << "\n";
        sub_stream << "matops took " << proc_time << "ns\n [ ";
        
        for( int i = 0; i < totalThreads; i++ )
        {
            sub_stream << results[i].sub << ",";
            if ( (i + 1) % width == 0 )
                sub_stream << "\n";
        }
        sub_stream << "]\n";
    }
    else{
        printf("FILE NOT OPEN?\n");
    }
    sub_stream.close();

    std::ofstream mul_stream(outputName + "_mull");
    if (mul_stream.is_open())
    {
        mul_stream << "Matrix MUL: " << totalThreads << " and Block Size: " << blockSize << "\n";
        mul_stream << "matops took " << proc_time << "ns\n[ ";
        
        for( int i = 0; i < totalThreads; i++ )
        {
            mul_stream << results[i].mul << ",";
            if ( (i + 1) % width == 0 )
                mul_stream << "\n";
        }
        mul_stream << "]\n";
    }
    else{
        printf("FILE NOT OPEN?\n");
    }
    mul_stream.close();

    std::ofstream div_stream(outputName + "_div");
    if (div_stream.is_open())
    {
        div_stream << "Matrix DIV: " << totalThreads << " and Block Size: " << blockSize << "\n";
        div_stream << "matops took " << proc_time << "ns\n[ ";
        
        for( int i = 0; i < totalThreads; i++ )
        {
            div_stream << results[i].div << ",";
            if ( (i + 1) % width == 0 )
                div_stream << "\n";
        }
        div_stream << "]\n";
    }
    else{
        printf("FILE NOT OPEN?\n");
    }
    div_stream.close();

}

__host__
void generate_data( const int& totalThreads, MatricesStruct* const host_data )
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,3);   

    for( int i = 0; i < totalThreads; i++ )
    {
        host_data[i].a = i * (distribution( generator ) + 1);
        host_data[i].b = distribution( generator );
    }
}
// Helper function for executing the matrix functionality via pageable memory
// calls matrix_ops and copies the results to an interleaved
// struct
__host__
void run_kernal(
    const int& blockSize, const int& totalThreads, const int& numBlocks, const std::string& outputName,
    MatricesStruct* const data, MatrixResultsStruct*& results, MatricesStruct* d_data, MatrixResultsStruct* d_results)
{
    auto data_size = totalThreads * sizeof(MatricesStruct);
    auto results_size = totalThreads * sizeof(MatrixResultsStruct);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    float proc_time;

    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    
    cudaEventRecord(start);
    cudaMemcpyAsync(d_data, data, data_size, cudaMemcpyHostToDevice, stream);
    
    matrix_ops<<<numBlocks, blockSize, 1, stream>>>(d_data, d_results); 

    cudaMemcpyAsync(results, d_results, results_size, cudaMemcpyDeviceToHost, stream);
   
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&proc_time, start, stop);
    cudaStreamDestroy(stream);
    printf("matrix_ops took %f ms\n", proc_time);


    if ( !outputName.empty() )
    {
        write_results( outputName, totalThreads, totalThreads / numBlocks, results, proc_time );
    }
}

// Helper function for initilizing the data used by the math functions will output results of pageable and
// pinned memory allocation.
__host__
void init_data(const int& totalThreads, MatricesStruct*& host_data, MatrixResultsStruct*& host_results,
    MatricesStruct*& d_data, MatrixResultsStruct*& d_results)
{
    auto data_size = totalThreads * sizeof(MatricesStruct);
    auto results_size = totalThreads * sizeof(MatrixResultsStruct);
    cudaMalloc((void**)&d_data, data_size);
    cudaMalloc((void**)&d_results, results_size);

    cudaHostAlloc((void **)&host_data, data_size, cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_results, results_size, cudaHostAllocDefault);

    generate_data( totalThreads, host_data );
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
void execute_matops(
    const int& blockSize, const int& totalThreads, const int& numBlocks,
    const std::string& outputName)
{
    MatricesStruct* data = nullptr;
    MatricesStruct* d_data = nullptr;
    MatrixResultsStruct* results = nullptr;
    MatrixResultsStruct* d_results = nullptr;
    init_data(totalThreads, data, results, d_data, d_results);
    run_kernal(blockSize, totalThreads, numBlocks, outputName, data, results,
        d_data, d_results);
    cleanup_data(data, d_data, results, d_results);
}

int main(int argc, char** argv)
{
    // read command line arguments
    int matDimSize = 512;
    int totalThreads = 512 * 512;
    int blockSize = 256;
    std::string outputName;

    if (argc >= 2) 
    {
        matDimSize = atoi(argv[1]);
        totalThreads = matDimSize * matDimSize;
    }
    if (argc >= 3)
    {
        blockSize = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        outputName = argv[3];
    }
    int numBlocks = totalThreads/blockSize;

    // validate command line arguments
    if (totalThreads % blockSize != 0)
    {
        ++numBlocks;
        totalThreads = numBlocks*blockSize;

        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }
    printf("%d x %d martix (%d elements)\n", matDimSize, matDimSize, totalThreads);
    printf("Executing with %d total threads %d blocks @ %d threads\n", totalThreads, numBlocks, blockSize);
    execute_matops( blockSize, totalThreads, numBlocks, outputName);
}
