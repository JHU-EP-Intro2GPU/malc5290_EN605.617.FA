#include "driver/cublas_driver.hpp"
#include "bin/bin_helper.hpp"


template <class T>
void multiply_matrices( const std::string& matrix_a_in, const std::string& matrix_b_in, bool write_results )
{
    auto mat_a = generate_matrix<T>( matrix_a_in );
    auto mat_b = generate_matrix<T>( matrix_b_in );

    std::cout << mat_a.m_size() << " " << mat_a.n_size() << " " << mat_b.n_size() << " ";
    CublasDriver<T> driver(mat_a, mat_b);
    driver.multiply_matrices();
}

int main(int argc, char* argv[])
{
    std::string matrix_a_in;
    std::string matrix_b_in;
    InputDataType data_type;
    bool write_results = false;
   
    if ( parse_args( matrix_a_in, matrix_b_in, data_type, write_results, argc, argv) )
    {
        std::cerr << "Parse error\n";
        exit(1);
    }
    
    std::cout << "cublas ";
    switch (data_type)
    {
        case InputDataType::INT:
            std::cout << "int ";
            multiply_matrices<int>( matrix_a_in, matrix_b_in, write_results );
            break;
        case InputDataType::SHORT:
            std::cout << "short ";
            multiply_matrices<short>( matrix_a_in, matrix_b_in, write_results );
            break;
        case InputDataType::DOUBLE:
            std::cout << "double ";
            multiply_matrices<double>( matrix_a_in, matrix_b_in, write_results );
            break;
        case InputDataType::FLOAT:
            std::cout << "float ";
            multiply_matrices<float>( matrix_a_in, matrix_b_in, write_results );
            break;
    }

    return 0;
}
