//Based on the work of Andrew Krepps
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
    int cipher;
} ResultsStruct;
// Uses the GPU to add the block + thread index in array_a to array_b to array_results
__global__
void add_arrays( 
    const MathStruct* const data,
    ResultsStruct* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    results[index].add = data[index].a + data[index].b;
}

// Uses the GPU to subtract the block + thread index in array_b from array_a to array_results
__global__
void sub_arrays( 
    const MathStruct* const data,
    ResultsStruct* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    results[index].sub = data[index].a - data[index].b;
}


// Uses the GPU to multiply the block + thread index in array_a by array_b to array_results
__global__
void mult_arrays( 
    const MathStruct* const data,
    ResultsStruct* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    results[index].mult = data[index].a * data[index].b;
}


// Uses the GPU to mudulot the block + thread index in array_a by array_b to array_results
__global__
void mod_arrays( 
    const MathStruct* const data,
    ResultsStruct* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    results[index].mod = data[index].a % data[index].b;
}

__host__
void print_results(
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

__host__
void run_kernal(
    const int& blockSize, const int& totalThreads, const int& numBlocks, const std::string& outputName,
    const MathStruct* const data, ResultsStruct*& results, MathStruct* d_data, ResultsStruct* d_results)
{
    auto start = std::chrono::high_resolution_clock::now();
    add_arrays<<<numBlocks, blockSize>>>(d_data, d_results);
    auto stop = std::chrono::high_resolution_clock::now();
    auto add_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    start = std::chrono::high_resolution_clock::now();
    sub_arrays<<<numBlocks, blockSize>>>(d_data, d_results);
    stop = std::chrono::high_resolution_clock::now();
    auto sub_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    mult_arrays<<<numBlocks, blockSize>>>(d_data, d_results);
    stop = std::chrono::high_resolution_clock::now();
    auto mult_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    start = std::chrono::high_resolution_clock::now();
    mod_arrays<<<numBlocks, blockSize>>>(d_data, d_results);
    stop = std::chrono::high_resolution_clock::now();
    auto mod_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    
    auto results_size = totalThreads * sizeof(ResultsStruct);
   
    cudaDeviceSynchronize();
    
    // Copy results to host
    cudaMemcpy(results, d_results, results_size, cudaMemcpyDeviceToHost);
    
   
    printf("Results with Thread Count: %d and Block Size: %d\n", totalThreads, blockSize);
    printf("Add Time nanoseconds:\t %ld\n", add_time);
    printf("Sub Time nanoseconds:\t %ld\n", sub_time);
    printf("Mult Time nanoseconds:\t %ld\n", mult_time);
    printf("Mod Time nanoseconds:\t %ld\n", mod_time);
   
    auto test = results[0];
    
    if ( !outputName.empty() )
    {
        print_results(outputName, totalThreads, blockSize, add_time, sub_time, mult_time, mod_time,
            data, results);
    }
}

__host__
void init_data(const int& totalThreads, const bool& pageable, MathStruct*& host_data, ResultsStruct*& host_results,
    MathStruct*& d_data, ResultsStruct*& d_results)
{
    auto data_size = totalThreads * sizeof(MathStruct);
    auto results_size = totalThreads * sizeof(ResultsStruct);
    cudaMalloc((void**)&d_data, data_size);
    cudaMalloc((void**)&d_results, results_size);

    if ( pageable )
    {
        host_data = (MathStruct*)malloc(data_size);
        host_results = (ResultsStruct*)malloc(results_size);
    }
    else
    {
        cudaMallocHost((void**)&host_data, data_size);
        cudaMallocHost((void**)&host_results, results_size);
    }

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

__host__ 
void cleanup( 
    const bool& pageable, MathStruct*& data, MathStruct*& d_data,
    ResultsStruct*& results, ResultsStruct*& d_results)
{
    cudaFree(d_data);
    cudaFree(d_results);
    
    if ( pageable )
    {
        free(data);
        free(results);
    }
    else
    {
        cudaFree(data);
        cudaFree(results);
    }
}

__host__
void execute_math_pageable_mem(
    const int& blockSize, const int& totalThreads, const int& numBlocks,
    const bool& writeResults, const std::string& outputName)
{
    MathStruct* data = nullptr;
    MathStruct* d_data = nullptr;
    ResultsStruct* results = nullptr;
    ResultsStruct* d_results = nullptr;
    init_data(totalThreads, true, data, results, d_data, d_results);
    run_kernal(blockSize, totalThreads, numBlocks, outputName, data, results,
        d_data, d_results);
    cleanup(true, data, d_data, results, d_results);
}

__host__
void execute_math_pinnable_mem(
    const int& blockSize, const int& totalThreads, const int& numBlocks,
    const bool& writeResults, const std::string& outputName)
{
    MathStruct* data = nullptr;
    MathStruct* d_data = nullptr;
    ResultsStruct* results = nullptr;
    ResultsStruct* d_results = nullptr;
    init_data(totalThreads, false, data, results, d_data, d_results);
    run_kernal(blockSize, totalThreads, numBlocks, outputName, data, results,
         d_data, d_results);
    cleanup(false, data, d_data, results, d_results);

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
    execute_math_pageable_mem( blockSize, totalThreads, numBlocks, outputResults, outputName);
}
