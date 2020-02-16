//Based on the work of Andrew Krepps
#include <chrono>
#include <fstream>
#include <random>
#include <stdio.h>
#include <string>

// Uses the GPU to add the block + thread index in array_a to array_b to array_results
__global__
void add_arrays( 
    const int* const array_a,
    const int* const array_b,
    int* const array_results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    array_results[index] = array_a[index] + array_b[index];
}

// Uses the GPU to subtract the block + thread index in array_b from array_a to array_results
__global__
void sub_arrays( 
    const int* const array_a,
    const int* const array_b,
    int* const array_results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    array_results[index] = array_a[index] - array_b[index];
}


// Uses the GPU to multiply the block + thread index in array_a by array_b to array_results
__global__
void mult_arrays( 
    const int* const array_a,
    const int* const array_b,
    int* const array_results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    array_results[index] = array_a[index] * array_b[index];
}


// Uses the GPU to mudulot the block + thread index in array_a by array_b to array_results
__global__
void mod_arrays( 
    const int* const array_a,
    const int* const array_b,
    int* const array_results)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    array_results[index] = array_a[index] % array_b[index];
}

int main(int argc, char** argv)
{
    // read command line arguments
    int totalThreads = (1 << 20);
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
    
    // Used for random number generation
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,3);

    // Device variables
    int* d_array_a;
    int* d_array_b;
    int* d_add_results;
    int* d_sub_results;
    int* d_mult_results;
    int* d_mod_results;
    
    // Host Variables
    int array_a[totalThreads];
    int array_b[totalThreads];
    int add_results[totalThreads];
    int sub_results[totalThreads];
    int mult_results[totalThreads];
    int mod_results[totalThreads];
    
    // Generate values for arrays.
    for( int i = 0; i < totalThreads; i++ )
    {
        array_a[i] = i;
        array_b[i] = distribution( generator );
    }

    auto array_size = totalThreads * sizeof(int);

    // Malloc GPU arrays
    cudaMalloc((void **)&d_array_a, array_size);
    cudaMalloc((void **)&d_array_b, array_size);
    cudaMalloc((void **)&d_add_results, array_size);
    cudaMalloc((void **)&d_sub_results, array_size);
    cudaMalloc((void **)&d_mult_results, array_size);
    cudaMalloc((void **)&d_mod_results, array_size);

    // Copy array values to Device
    cudaMemcpy( d_array_a, array_a, array_size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_array_b, array_b, array_size, cudaMemcpyHostToDevice );
 
   
    // Execute assignment operations
    auto start = std::chrono::high_resolution_clock::now();
    add_arrays<<<numBlocks, blockSize>>>(d_array_a, d_array_b, d_add_results);
    auto stop = std::chrono::high_resolution_clock::now();
    auto add_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    start = std::chrono::high_resolution_clock::now();
    sub_arrays<<<numBlocks, blockSize>>>(d_array_a, d_array_b, d_sub_results);
    stop = std::chrono::high_resolution_clock::now();
    auto sub_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    mult_arrays<<<numBlocks, blockSize>>>(d_array_a, d_array_b, d_mult_results);
    stop = std::chrono::high_resolution_clock::now();
    auto mult_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    start = std::chrono::high_resolution_clock::now();
    mod_arrays<<<numBlocks, blockSize>>>(d_array_a, d_array_b, d_mod_results);
    stop = std::chrono::high_resolution_clock::now();
    auto mod_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
   
    // Copy results to host
    cudaMemcpy( &add_results, d_add_results, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy( &sub_results, d_sub_results, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy( &mult_results, d_mult_results, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy( &mod_results, d_mod_results, array_size, cudaMemcpyDeviceToHost);
    
    // cleanup
    cudaFree(d_array_a);
    cudaFree(d_array_b);
    cudaFree(d_add_results);
    cudaFree(d_sub_results);
    cudaFree(d_mult_results);
    cudaFree(d_mod_results);
    
    printf("Results with Thread Count: %d and Block Size: %d\n", totalThreads, blockSize);
    printf("Add Time nanoseconds:\t %d\n", add_time);
    printf("Sub Time nanoseconds:\t %d\n", sub_time);
    printf("Mult Time nanoseconds:\t %d\n", mult_time);
    printf("Mod Time nanoseconds:\t %d\n", mod_time);

    if (outputResults)
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
                stream << "A(" << array_a[i] << ") + B("  << array_b[i] << ") = " <<  add_results[i] << "\n";
            }
            
            stream << "\n\nSub Results:\n";
            for( int i = 0; i < totalThreads; i++ )
            {
                stream << "A(" << array_a[i] << ") - B("  << array_b[i] << ") = " <<  sub_results[i] << "\n";
            }
            
            stream << "\n\nMult Results:\n";
            for( int i = 0; i < totalThreads; i++ )
            {
                stream << "A(" << array_a[i] << ") * B("  << array_b[i] << ") = " <<  mult_results[i] << "\n";
            }
            
            stream << "\n\nMult Results:\n";
            for( int i = 0; i < totalThreads; i++ )
            {
                stream << "A(" << array_a[i] << ") % B("  << array_b[i] << ") = " <<  mod_results[i] << "\n";
            }
        
        }
        else{
            printf("FILE NOT OPEN?\n");
        }
        stream.close();
    }
    /*   
    printf("Add results\n");
    for( int i = 0; i < totalThreads; i++ )
    {
        printf("A(%d) + B(%d) = %d, ", array_a[i], array_b[i], add_results[i]);
    }
    
    printf("SUB results\n");
    for( int i = 0; i < totalThreads; i++ )
    {
        printf("A(%d) - B(%d) = %d, ", array_a[i], array_b[i], sub_results[i]);
    }
    
    printf("MULT results\n");
    for( int i = 0; i < totalThreads; i++ )
    {
        printf("A(%d) * B(%d) = %d, ", array_a[i], array_b[i], mult_results[i]);
    }
    
    printf("MOD results\n");
    for( int i = 0; i < totalThreads; i++ )
    {
        printf("A(%d) mod B(%d) = %d, ", array_a[i], array_b[i], mod_results[i]);
    }
    */
    printf("\n");
}
