#include <chrono>
#include <fstream>
#include <stdio.h>
#include <string>
#include <iostream>
#include <math.h>

__constant__ int DbMToWattConstDivisor = 1000;
__constant__ double CTen = 10;
__constant__ int P = 1;

__device__
void d_dbm_to_watts( const int& dbm, double& results )
{
    const double pow = (P * dbm) / CTen;
    const double numerator = std::pow( CTen, pow );
    const double watts = numerator / DbMToWattConstDivisor;
    results = watts;
}


__global__
void convert_dbm_to_watts(const int* const d_dbms, double* const results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    d_dbm_to_watts( d_dbms[index], results[index] );
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
    const double* const results, const int& proc_time, const int& memcpy_time)
{
    std::ofstream stream(outputName);
    if (stream.is_open())
    {
        stream << "dBm -> Watts: " << totalThreads << " and Block Size: " << blockSize << "\n";

        for( int i = 0; i < totalThreads; i++ )
        {
            stream << i <<  " dBm " << " -> " << results[i] << " Watts\n";
        }
        
        stream << "dbm -> watts took " << proc_time << "ns\n";
        stream << "mem copy took dev->host" << memcpy_time << "ns\n";
    
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
void run_kernal(
    const int& blockSize, const int& totalThreads, const int& numBlocks, const std::string& outputName,
    double*& results, int* d_data, double* d_results)
{
    auto start = get_clock_time();
    convert_dbm_to_watts<<<numBlocks, blockSize>>>(d_data, d_results); 
    
    cudaDeviceSynchronize();

    auto stop = get_clock_time();
    
    int proc_time = get_duration_ns( start, stop );
    printf("dbm -> watts took %d ns\n", proc_time);

    auto results_size = totalThreads * sizeof(double);

    start = get_clock_time();
    cudaMemcpy(results, d_results, results_size, cudaMemcpyDeviceToHost);
    stop = get_clock_time();

    int memcpy_time = get_duration_ns( start, stop );
    printf("mem copy took %d ns\n", memcpy_time);

    if ( !outputName.empty() )
    {
        write_results( outputName, totalThreads, totalThreads / numBlocks, results, proc_time, memcpy_time );
    }
}

// Helper function for initilizing the data used by the math functions will output results of pageable and
// pinned memory allocation.
__host__
void init_data(const int& totalThreads, int*& host_data, double*& host_results,
    int*& d_data, double*& d_results)
{
    auto data_size = totalThreads * sizeof(int);
    auto results_size = totalThreads * sizeof(double);
    cudaMalloc((void**)&d_data, data_size);
    cudaMalloc((void**)&d_results, results_size);

    host_data = (int*)malloc(data_size);
    host_results = (double*)malloc(results_size);

    for( int i = 0; i < totalThreads; i++ )
    {
        host_data[i] = i;
    }
    cudaMemcpy(d_data, host_data, data_size, cudaMemcpyHostToDevice);
}

// Helper function for cleaning up allocated memory used by math functionality.
__host__ 
void cleanup_data( 
    int*& data, int*& d_data,
    double*& results, double*& d_results)
{
    cudaFree(d_data);
    cudaFree(d_results);
    
    free(data);
    free(results);
}

// Used to run the math functionality with pageable memory
__host__
void execute_dBm_to_watts(
    const int& blockSize, const int& totalThreads, const int& numBlocks,
    const bool& writeResults, const std::string& outputName)
{
    int* data = nullptr;
    int* d_data = nullptr;
    double* results = nullptr;
    double* d_results = nullptr;
    init_data(totalThreads, data, results, d_data, d_results);
    run_kernal(blockSize, totalThreads, numBlocks, outputName, results,
        d_data, d_results);
    cleanup_data(data, d_data, results, d_results);
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
    execute_dBm_to_watts( blockSize, totalThreads, numBlocks, outputResults, outputName);
}
