#ifndef en605_617_matrix_helper_hpp
#define en605_617_matrix_helper_hpp
#include "matrix/Matrix.hpp"

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>

namespace MatrixHelper
{
    template <class T>
    Matrix<T> load_data(std::istream& stream)
    {
        std::string line;
        std::getline(stream, line);
        std::string size;
        std::stringstream ss(line);
        int m = -1;
        int n = -1;
        while ( std::getline(ss, size, ',') )
        {
            int tempSize(stoi(size));
            if( m == -1 )
                m = tempSize;
            else
                n = tempSize;
        }

        std::stringstream valueStream;
        std::vector<T> matrix;
        matrix.reserve(m * n);
        T matValue;
        ss.str(std::string());
        ss.clear();
        while( std::getline(stream, line) )
        {  
            ss << line;
            std::string value;
            
            while(std::getline(ss, value, ','))
            {   
                valueStream.str(value);
                valueStream.clear();
                valueStream >> matValue;
                matrix.push_back(matValue);
            }
            ss.str(std::string());
            ss.clear();
        }
        return Matrix<T>(m, n, matrix);
    }

    template <class T>
    void copy_data( const Matrix<T>& data, T*& copy )
    {
        int index = 0;
        for ( T val : data.matrix() )
        {
            copy[index] = val;
            index++;
#ifdef DEBUG
            std::cout << "COPYING: " << val << "\n";
#endif
        }
    }

    template <class T>
    void change_orientation( Matrix<T>& matrix, const Orientation& orientation )
    {
        if ( matrix.orientation() == orientation )
        {
            std::cout << "NOTHING TO DO\n";
        }

        std::vector<T> updatedMatrix;
        updatedMatrix.reserve(matrix.size());
        if ( matrix.orientation() == Orientation::ROW_MAJOR )
        {
            matrix.setOrientation(Orientation::COLUMN_MAJOR);
            for ( int n = 0; n < matrix.n_size(); n++ )
            {
                for ( int m = 0; m < matrix.m_size(); m++ )
                {
                    updatedMatrix.push_back(matrix.matrix()[matrix.n_size() * m + n]); 
                }
            }
        }
        else
        {
            matrix.setOrientation(Orientation::ROW_MAJOR);
            for ( int m = 0; m < matrix.m_size(); m++ )
            {
                for ( int n = 0; n < matrix.n_size(); n++ )
                {
                    updatedMatrix.push_back(matrix.matrix()[matrix.m_size() * n + m]); 
                }
            }
        }
        matrix.setMatrix(updatedMatrix);
    }

    template <class T>
    void print_matrix( const Orientation& orientation, const int& m_size, const int& n_size, const std::vector<T>& matrix)
    {
        if ( orientation == Orientation::COLUMN_MAJOR )
        {
            for ( int m = 0; m < m_size; m++ )
            {
                for ( int n = 0; n < n_size; n++ )
                {
                    T val = matrix[m_size * n + m];
                    std::cout << val << (n + 1 == n_size ? "\n" : " , ");
                }
            }
        }
        else
        {
            for ( int m = 0; m < m_size; m++ )
            {
                for ( int n = 0; n < n_size; n++ )
                {
                    T val = matrix[n_size *  m + n];
                    std::cout << val << (n + 1 == n_size ? "\n" : " , ");
                }
            }
        }
    }

    template <class T>
    void print_matrix( const Orientation& orientation, const int& m_size, const int& n_size, const T* const matrix)
    {
        if ( orientation == Orientation::COLUMN_MAJOR )
        {
            for ( int m = 0; m < m_size; m++ )
            {
                for ( int n = 0; n < n_size; n++ )
                {
                    T val = matrix[m_size * n + m];
                    std::cout << val << (n + 1 == n_size ? "\n" : " , ");
                }
            }
        }
        else
        {
            for ( int m = 0; m < m_size; m++ )
            {
                for ( int n = 0; n < n_size; n++ )
                {
                    T val = matrix[n_size *  m + n];
                    std::cout << val << (n + 1 == n_size ? "\n" : " , ");
                }
            }
        }
    }
};
#endif //en605_617_matrix_helper_hpp
