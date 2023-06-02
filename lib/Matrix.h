//
// Created by andre on 5/31/23.
//

#ifndef SVD_JACOBI_MPI_OMP_CMAKE_BUILD_DEBUG_MATRIX_H_
#define SVD_JACOBI_MPI_OMP_CMAKE_BUILD_DEBUG_MATRIX_H_

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
