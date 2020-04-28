
#add_library(DRIVER
#    driver)
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
    )

include_directories(matrix .)

add_executable(cublasmult-driver bin/cublasmult.cpp
    driver/cublas_driver.cu
    driver/temp_driver.cpp
    )
    #${MATRIX_INCLUDES}
    #${DRIVER_INCLUDES})

target_link_libraries(cublasmult-driver
    PUBLIC
    -lcublas)

target_include_directories(cublasmult-driver PUBLIC
    ${PROJECT_BINARY_DIR})
    #MATRIX_INCLUDES)

    #target_link_libraries(cublasmult-driver )
