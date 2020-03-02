#include "assignment.h"

#include <chrono>
#include <fstream>
#include <random>
#include <stdio.h>
#include <string>
#include <iostream>

typedef struct {
    unsigned int a;
    unsigned int b;
} MathStruct; 

typedef struct {
    int add;
    int sub;
    int mult;
    int mod;
} ResultsStruct;


__constant__ int ConstantMemGPU[ARRAY_SIZE];

// Uses the GPU to add the block + thread index in array_a to array_b to array_results
__device__
void add_arrays( 
    const MathStruct* const data,
    ResultsStruct* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    results[index].add = data[index].a + data[index].b;
}

// Uses the GPU to subtract the block + thread index in array_b from array_a to array_results
__device__
void sub_arrays( 
    const MathStruct* const data,
    ResultsStruct* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    results[index].sub = data[index].a - data[index].b;
}


// Uses the GPU to multiply the block + thread index in array_a by array_b to array_results
__device__
void mult_arrays( 
    const MathStruct* const data,
    ResultsStruct* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    results[index].mult = data[index].a * data[index].b;
}


// Uses the GPU to mudulo the block + thread index in array_a by array_b to array_results
__device__
void mod_arrays( 
    const MathStruct* const data,
    ResultsStruct* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    results[index].mod = data[index].a % data[index].b;
}


__global__
void shared_memtest(const MathStruct* const d, ResultsStruct* const results, int n)
{
    extern __shared__ ResultsStruct s[];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int shared_index = threadIdx.x;
    //int tr = n - t - 1;
    add_arrays(d, results);
    sub_arrays(d, results);
    mult_arrays(d, results);
    mod_arrays(d, results);
    
    s[shared_index] = results[index];
    __syncthreads();
    
    if ( shared_index > 0 )
    {
        results[index].add += s[shared_index-1].add;
        results[index].sub -= s[shared_index-1].sub;
        results[index].mult *= s[shared_index-1].mult;
        results[index].mod %= s[shared_index-1].mod;
    }
    //d[t].a = s[tr];
}

__global__
void const_memtest_pre(const MathStruct* const d, ResultsStruct* const results)
{
    add_arrays(d, results);
    sub_arrays(d, results);
    mult_arrays(d, results);
    mod_arrays(d, results);
}

__global__
void const_memtest_post(const MathStruct* const d, ResultsStruct* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    results[index].add += ConstantMemGPU[index];
    results[index].sub -= ConstantMemGPU[index];
    results[index].mult *= ConstantMemGPU[index];
    results[index].mod ^= ConstantMemGPU[index];
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
    const std::string& outputName, const int& totalThreads, const int& blockSize, const int& add_time,
    const int& sub_time, const int& mult_time, const int& mod_time, const MathStruct* const  data,
    const ResultsStruct* const results)
{
    std::ofstream stream(outputName);
    if (stream.is_open())
    {
        stream << "Results with Thread Count: " << totalThreads << " and Block Size: " << blockSize << "\n";
        stream << "Add Time nanoseconds:\t" << add_time << "\n";
        stream << "Sub Time nanoseconds:\t" << sub_time << "\n";
        stream << "Mult Time nanoseconds:\t" << mult_time << "\n";
        stream << "Mod Time nanoseconds:\t" << mod_time << "\n";

        stream << "Add Results:\n";
        for( int i = 0; i < totalThreads; i++ )
        {
            stream << "A(" << data[i].a << ") + B("  << data[i].b << ") = " <<  results[i].add << "\n";
        }
        
        stream << "\n\nSub Results:\n";
        for( int i = 0; i < totalThreads; i++ )
        {
            stream << "A(" << data[i].a << ") - B("  << data[i].b << ") = " <<  results[i].sub << "\n";
        }
        
        stream << "\n\nMult Results:\n";
        for( int i = 0; i < totalThreads; i++ )
        {
            stream << "A(" << data[i].a << ") * B("  << data[i].b << ") = " <<  results[i].mult << "\n";
        }
        
        stream << "\n\nMult Results:\n";
        for( int i = 0; i < totalThreads; i++ )
        {
            stream << "A(" << data[i].a << ") % B("  << data[i].b << ") = " <<  results[i].mod << "\n";
        }
    
    }
    else{
        printf("FILE NOT OPEN?\n");
    }
    stream.close();
}



// Helper function for executing the math functionality via pinned or pageable memory
// calls the add_array, sub_array, mult_array, and mod_array and copies the results to an interleaved
// struct
__host__
void run_math_kernal(
    const int& blockSize, const int& totalThreads, const int& numBlocks, const std::string& outputName,
    const MathStruct* const data, ResultsStruct*& results, MathStruct* d_data, ResultsStruct* d_results)
{
    auto start = get_clock_time();
    shared_memtest<<<numBlocks, blockSize, (blockSize) * sizeof(ResultsStruct)>>>(d_data, d_results, blockSize); 
    
    cudaDeviceSynchronize();

    auto stop = get_clock_time();
    
    printf("shared_memtest took %d ns\n", get_duration_ns(start, stop));

    auto results_size = totalThreads * sizeof(ResultsStruct);
    cudaMemcpy(results, d_results, results_size, cudaMemcpyDeviceToHost);
}

// Helper function for executing the math functionality via pinned or pageable memory
// calls the add_array, sub_array, mult_array, and mod_array and copies the results to an interleaved
// struct
__host__
void run_const_math_kernal(
    const int& blockSize, const int& totalThreads, const int& numBlocks, const std::string& outputName,
    const MathStruct* const data, ResultsStruct*& results, MathStruct* d_data, ResultsStruct* d_results)
{
    auto start = get_clock_time();
    const_memtest_pre<<<numBlocks, blockSize>>>(d_data, d_results); 
    
    cudaDeviceSynchronize();

    auto stop = get_clock_time();
    
    int pre_time = get_duration_ns( start, stop );
    printf("const_memtest pre took %d ns\n", pre_time);

    auto results_size = totalThreads * sizeof(ResultsStruct);

    cudaDeviceSynchronize();
    cudaMemcpyToSymbol(ConstantMemGPU, d_results, results_size);
    
    start = get_clock_time();
    const_memtest_post<<<numBlocks, blockSize>>>(d_data, d_results);
    cudaDeviceSynchronize();

    stop = get_clock_time();
    int post_time = get_duration_ns( start, stop );
    printf("const_memtest post took %d ns\n", post_time);
    printf("const_memtest total took %d ns\n", pre_time + post_time);
    cudaMemcpy(results, d_results, results_size, cudaMemcpyDeviceToHost);
}

// Helper function for initilizing the data used by the math functions will output results of pageable and
// pinned memory allocation.
__host__
void init_math_data(const int& totalThreads, MathStruct*& host_data, ResultsStruct*& host_results,
    MathStruct*& d_data, ResultsStruct*& d_results)
{
    auto data_size = totalThreads * sizeof(MathStruct);
    auto results_size = totalThreads * sizeof(ResultsStruct);
    cudaMalloc((void**)&d_data, data_size);
    cudaMalloc((void**)&d_results, results_size);

    host_data = (MathStruct*)malloc(data_size);
    host_results = (ResultsStruct*)malloc(results_size);

    // Used for random number generation
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,3);

    for( int i = 0; i < totalThreads; i++ )
    {
        host_data[i].a = i;
        host_data[i].b = distribution( generator );
    }
    cudaMemcpy(d_data, host_data, data_size, cudaMemcpyHostToDevice);
}

// Helper function for cleaning up allocated memory used by math functionality.
__host__ 
void cleanup_math( 
    MathStruct*& data, MathStruct*& d_data,
    ResultsStruct*& results, ResultsStruct*& d_results)
{
    cudaFree(d_data);
    cudaFree(d_results);
    
    free(data);
    free(results);
}

// Used to run the math functionality with pageable memory
__host__
void execute_math(
    const int& blockSize, const int& totalThreads, const int& numBlocks,
    const bool& writeResults, const std::string& outputName)
{
    MathStruct* data = nullptr;
    MathStruct* d_data = nullptr;
    ResultsStruct* results = nullptr;
    ResultsStruct* d_results = nullptr;
    init_math_data(totalThreads, data, results, d_data, d_results);
    run_math_kernal(blockSize, totalThreads, numBlocks, outputName, data, results,
        d_data, d_results);
    cleanup_math(data, d_data, results, d_results);
}

__host__
void execute_math_const_mem(
    const int& blockSize, const int& totalThreads, const int& numBlocks,
    const bool& writeResults, const std::string& outputName)
{
    MathStruct* data = nullptr;
    MathStruct* d_data = nullptr;
    ResultsStruct* results = nullptr;
    ResultsStruct* d_results = nullptr;
    init_math_data(totalThreads, data, results, d_data, d_results);
    run_const_math_kernal(blockSize, totalThreads, numBlocks, outputName, data, results,
        d_data, d_results);
    cleanup_math(data, d_data, results, d_results);
}

int main(int argc, char** argv)
{
    // read command line arguments
    int totalThreads = 512;
    int blockSize = 256;
    bool outputResults = false;
    std::string outputName;

    if (argc >= 2) 
    {
        totalThreads = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        blockSize = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        outputResults = true;
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
    printf("Executing with %d total threads %d blocks @ %d threads\n", totalThreads, numBlocks, blockSize);
    execute_math( blockSize, totalThreads, numBlocks, outputResults, outputName);
    execute_math_const_mem( blockSize, totalThreads, numBlocks, outputResults, outputName);
}
