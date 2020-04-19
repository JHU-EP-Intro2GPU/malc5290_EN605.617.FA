#ifndef en605_617_base_driver_hpp
#define en605_617_base_driver_hpp
#include "driver/Matrix.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>

class BaseDriver
{
    public:
    virtual ~BaseDriver(){};
    
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
            std::cout << "SS: " << ss.str() << "\n";
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
};

#endif //en605_617_base_driver_hpp
