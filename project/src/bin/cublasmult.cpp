#include "matrix/matrix_helper.hpp"
#include "driver/cublas_driver.hpp"

enum class InputDataType { INT, SHORT, DOUBLE, FLOAT };

void print_help()
{
    std::cout << "Runs matrix multiplication algorithms utilizing the GPU\n";
    std::cout << "-a <input file> the absolute path of the input file for matrix A\n";
    std::cout << "-b <input file> the absolute path of the input file for matrix B\n";
    std::cout << "-i matrix data type are integers ( select one between -i -s -d -f or last will get used )\n";
    std::cout << "-s matrix data type are shorts   ( select one between -i -s -d -f or last will get used )\n";
    std::cout << "-d matrix data type are doubles  ( select one between -i -s -d -f or last will get used )\n";
    std::cout << "-f matrix data type are floats   ( select one between -i -s -d -f or last will get used )\n";
    std::cout << "-o --out Flag for outputting a result file to the current directory\n";
}

template <class T>
Matrix<T> generate_matrix( const std::string& input_file )
{
    std::fstream fs(input_file);
    if (!fs.is_open())
    {
        std::cout << "File not found \"" << input_file << "\"\n";
        exit(1);
    }
    auto matrix = MatrixHelper::load_data<T>(fs);

    return matrix;
}

template <class T>
void multiply_matrices( const std::string& matrix_a_in, const std::string& matrix_b_in, bool write_results )
{
    auto mat_a = generate_matrix<T>( matrix_a_in );
    auto mat_b = generate_matrix<T>( matrix_b_in );
    std::cout << "Matrix A\n";
    MatrixHelper::print_matrix(Orientation::ROW_MAJOR, mat_a.m_size(), mat_a.n_size(), mat_a.matrix());
    std::cout << "Matrix B\n";
    MatrixHelper::print_matrix(Orientation::ROW_MAJOR, mat_b.m_size(), mat_b.n_size(), mat_b.matrix());

    MatrixHelper::change_orientation<T>( mat_a, Orientation::COLUMN_MAJOR );
    MatrixHelper::change_orientation<T>( mat_a, Orientation::ROW_MAJOR );
    CublasDriver<T> driver(mat_a, mat_b);
    driver.multiply_matrices();
}

int main(int argc, char* argv[])
{
    std::string matrix_a_in;
    std::string matrix_b_in;
    InputDataType data_type;
    bool write_results = false;

    if( argc < 3 || argc > 7 )
    {
        print_help();
        std::cout << "Incorrect number of arguments...exiting\n";
        return 1;
    }
    for( int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if ( arg == "-a" )
        {
            matrix_a_in = argv[++i];
            std::ifstream file(matrix_a_in);
            if(!file)
            {
                print_help();
                std::cout << "Failed to open file at \"" << matrix_a_in << "\"...exiting\n";
                return 1;
            }
            file.close();
        }
        else if( arg == "-b" )
        {
            matrix_b_in = argv[++i];
            std::ifstream file(matrix_b_in);
            if(!file)
            {
                print_help();
                std::cout << "Failed to open file at \"" << matrix_b_in << "\"...exiting\n";
                return 1;
            }
            file.close();
        }
        else if( arg == "-i" || arg == "-I" )
        {
            data_type = InputDataType::INT;
        }
        else if( arg == "-s" || arg == "-S" )
        {
            data_type = InputDataType::SHORT;
        }
        else if( arg == "-d" || arg == "-D" )
        {
            data_type = InputDataType::DOUBLE;
        }
        else if( arg == "-f" || arg == "-F" )
        {
            data_type = InputDataType::FLOAT;
        }
        else if( arg == "-o" || arg == "--out" )
            write_results = true;
    }

    if(!matrix_a_in.length() || !matrix_b_in.length())
    {
        print_help();
        return 1;
    }

    switch (data_type)
    {
        case InputDataType::INT:
            multiply_matrices<int>( matrix_a_in, matrix_b_in, write_results );
            break;
        case InputDataType::SHORT:
            multiply_matrices<short>( matrix_a_in, matrix_b_in, write_results );
            break;
        case InputDataType::DOUBLE:
            multiply_matrices<double>( matrix_a_in, matrix_b_in, write_results );
            break;
        case InputDataType::FLOAT:
            multiply_matrices<float>( matrix_a_in, matrix_b_in, write_results );
            break;
    }

    return 0;
}
