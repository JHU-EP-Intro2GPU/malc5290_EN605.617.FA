
#add_library(DRIVER
#    driver)
set(DRIVERINCLUDES 
    driver/base_driver.hpp
    driver/Matrix.hpp)

include_directories(driver .)

add_executable(test-driver bin/test.cpp
    ${DRIVERINCLUDES})

target_link_libraries(test-driver
    PUBLIC)

target_include_directories(test-driver PUBLIC
    ${PROJECT_BINARY_DIR}
    "driver")

