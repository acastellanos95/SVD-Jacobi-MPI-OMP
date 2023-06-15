cmake_minimum_required(VERSION 3.23)
project(SVD_Jacobi_MPI_OMP)

set(CMAKE_CXX_STANDARD 11)


find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER  mpicc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")

add_executable(SVD_Jacobi_MPI_OMP main.cpp lib/Matrix.h lib/Utils.cpp lib/Utils.h lib/global.h lib/JacobiMethods.cpp lib/JacobiMethods.h tests/Tests.cpp tests/Tests.h)

target_compile_options(SVD_Jacobi_MPI_OMP PUBLIC ${LAPACK_LINKER_FLAGS})
target_link_libraries(SVD_Jacobi_MPI_OMP ${MPI_C_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX MPI::MPI_CXX)

if(MKL_ENV)
    set_tests_properties(mytest PROPERTIES ENVIRONMENT "${MKL_ENV}")
endif()