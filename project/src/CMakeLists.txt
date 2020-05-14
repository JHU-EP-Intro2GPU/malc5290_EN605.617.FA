set(BIN_INCLUDES
    bin/bin_helper.hpp
    )
set(MATRIX_INCLUDES
    matrix/matrix_helper.hpp
    matrix/Matrix.hpp)

set(DRIVER_INCLUDES
    driver/MatrixDriver.hpp
    #    driver/IMatrixDriver.hpp
    #driver/SMatrixDriver.hpp
    #driver/FMatrixDriver.hpp
    #driver/DMatrixDriver.hpp
    driver/cublas_driver.hpp
    driver/cutlass_driver.hpp
    driver/cutlass_helper.h
    )

include_directories(matrix 
    .)

add_executable(cublasmult-driver 
    bin/cublasmult.cpp
    driver/cublas_driver.cu
    )

add_executable(cutlassmult-driver 
    bin/cutlassmult.cpp
    driver/cutlass_driver.cu
    )

add_executable(naivemult-driver
    bin/naivemult.cpp
    driver/naive_driver.cu
    )

target_link_libraries(cutlassmult-driver
    PRIVATE
    /opt/cutlas-comp75/lib/libcutlass.so)

target_link_libraries(cublasmult-driver
    PUBLIC
    -lcublas)

target_include_directories(cublasmult-driver 
    PUBLIC
    ${PROJECT_BINARY_DIR})

target_include_directories(cutlassmult-driver 
    PUBLIC
    /opt/cutlas-comp75/include
    ${PROJECT_BINARY_DIR})

target_include_directories(naivemult-driver
    PUBLIC
    ${PROJECT_BINARY_DIR})

#MATRIX_INCLUDES)

    #target_link_libraries(cublasmult-driver )
