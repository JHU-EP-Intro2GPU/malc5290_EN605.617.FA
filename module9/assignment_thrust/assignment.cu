#include "my_helper.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <stdlib.h>
#include <chrono>

enum class MathOp
{
    ADD, SUB, MUL, MOD
};

__host__
void generate_random( thrust::host_vector<int>& H )
{
    for( size_t i = 0; i < H.size(); i++ )
    {
        H[i] = rand() % 2000;
    }
}

__host__
void print_host( const thrust::host_vector<int>& H )
{
    std::cout << "Vector: { ";
    thrust::copy(H.begin(), H.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "}\n";
}

__host__
void print_device( const thrust::device_vector<int>& D )
{
    std::cout << "Vector: { ";
    thrust::copy(D.begin(), D.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "}\n";
}

__host__
void perform_op( 
    const thrust::device_vector<int>& d_A, 
    const thrust::device_vector<int>& d_B, 
    thrust::device_vector<int>& d_results,
    const MathOp& op, const bool& print_vectors)
{
    HR_TimePoint start;
    HR_TimePoint end;
    switch ( op )
    {
        case MathOp::ADD:
            std::cout << "ADD: ";
            start = get_clock_time();
            thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_results.begin(),  thrust::plus<int>());
            end = get_clock_time();
        break;
        case MathOp::SUB:
            std::cout << "SUB: ";
            start = get_clock_time();
            thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_results.begin(),  thrust::minus<int>());
            end = get_clock_time();
        break;
        case MathOp::MUL:
            std::cout << "MULT: ";
            start = get_clock_time();
            thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_results.begin(),  thrust::multiplies<int>());
            end = get_clock_time();
        break;
        case MathOp::MOD:
            std::cout << "MOD: ";
            start = get_clock_time();
            thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_results.begin(),  thrust::modulus<int>());
            end = get_clock_time();
        break;
    }
    if ( print_vectors )
        print_device(d_results);
    std::cout << "Operation took " << get_duration_ns(start, end) << " ns\n\n";
}

void perform_ops( const int& size, const bool& print_vectors )
{
    thrust::host_vector<int> H(size);
    thrust::device_vector<int> d_results(size);
    generate_random(H);
    
    if ( print_vectors )
    {   
        std::cout << "Vector A:\n";
        print_host( H );
    }

    thrust::device_vector<int> d_A = H;

    generate_random(H);
    
    if ( print_vectors )
    {
        std::cout << "Vector B:\n";
        print_host( H );
        std::cout << "\n";
    }
    thrust::device_vector<int> d_B = H;
    
    perform_op(d_A, d_B, d_results, MathOp::ADD, print_vectors);
    perform_op(d_A, d_B, d_results, MathOp::SUB, print_vectors);
    perform_op(d_A, d_B, d_results, MathOp::MUL, print_vectors);
    perform_op(d_A, d_B, d_results, MathOp::MOD, print_vectors);
}

int main(int argc, char** argv)
{
    int vector_size = 5;
    bool print_vectors = true;
    if ( argc == 2 )
    {
        vector_size = atoi(argv[1]);
    }
    else if( argc > 2 )
    {
        std::cerr << "Bad Args";
        exit(1);
    }
    if ( vector_size > 25 )
        print_vectors = false;
    std::cout << "executing with vector size " << vector_size << "\n";
    perform_ops(vector_size, print_vectors);

    return 0;
}

