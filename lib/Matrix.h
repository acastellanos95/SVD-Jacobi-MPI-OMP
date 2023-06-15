//
// Created by andre on 5/31/23.
//

#ifndef SVD_JACOBI_MPI_OMP_CMAKE_BUILD_DEBUG_MATRIX_H_
#define SVD_JACOBI_MPI_OMP_CMAKE_BUILD_DEBUG_MATRIX_H_

#include <vector>
#include <cstring>

struct Matrix{
  unsigned long width{};
  unsigned long height{};
  double *elements{};

  Matrix()= default;

  Matrix(unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

    this->elements = new double [height * width];
  }

  ~Matrix(){
    delete []elements;
  }
};

struct MatrixMPI{
  unsigned long width{};
  unsigned long height{};
  double *elements{};

  MatrixMPI()= default;

  MatrixMPI(unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

    this->elements = new double [height * width];
  }

  MatrixMPI(std::vector<double> &A, unsigned long height, unsigned long width){
    this->width = width;
    this->height = height;

    this->elements = new double [height * width];
    size_t all_size = height * width;

    std::copy(A.begin(), A.end(), this->elements);
  }

  MatrixMPI(MatrixMPI &matrix_mpi){
    this->width = matrix_mpi.width;
    this->height = matrix_mpi.height;

    this->elements = new double [height * width];
    std::copy(matrix_mpi.elements, matrix_mpi.elements + height * width, this->elements);
  }

  void free(){
    delete []elements;
  }
};

struct Vector{
  unsigned long length;
  double *elements;
  ~Vector(){
    delete []elements;
  }
};

#endif //SVD_JACOBI_MPI_OMP_CMAKE_BUILD_DEBUG_MATRIX_H_
