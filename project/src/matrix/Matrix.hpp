#ifndef en605_617_matrix_hpp
#define en605_617_matrix_hpp
#include <vector>
enum class Orientation{ ROW_MAJOR, COLUMN_MAJOR };
template <class T>
class Matrix
{
public:
    Matrix(){}
    Matrix( const int& m, const int& n, const std::vector<T>& matrix_values )
        : _m( m )
        , _n( n )
        , _matrix_values( matrix_values )
        , _orientation( Orientation::ROW_MAJOR ){}
    
    Matrix( const Matrix<T>& cp )
        : _m( cp._m )
        , _n( cp._n )
        , _matrix_values( cp._matrix_values )
        , _orientation( cp._orientation ){}
    
    int m_size() const {return _m;}
    int n_size() const {return _n;}
    int size() const {return _m * _n;}

    Orientation orientation() const {return _orientation;}
    void setOrientation( const Orientation& orientation){_orientation = orientation;}
    
    std::vector<T> matrix() const{ return _matrix_values; }
    std::vector<T>& matrix_ref() { return _matrix_values; }
    void setMatrix(const std::vector<T>& matrix){_matrix_values = matrix;}

private:
    int             _m;
    int             _n;
    std::vector<T>  _matrix_values;
    Orientation     _orientation;
};

#endif //en605_617_matrix_hpp
