//
// Created by andre on 5/31/23.
//

#ifndef SVD_JACOBI_MPI_OMP_LIB_JACOBIMETHODS_H_
#define SVD_JACOBI_MPI_OMP_LIB_JACOBIMETHODS_H_

#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <random>
#include <omp.h>
#include <mpi.h>
#include "Matrix.h"
#include "global.h"
#include "Utils.h"

namespace Thesis {

enum SVD_OPTIONS {
  AllVec,
  SomeVec,
  NoVec
};

#define V_OPTION

void sequential_dgesvd(SVD_OPTIONS jobu,
                       SVD_OPTIONS jobv,
                       size_t m,
                       size_t n,
                       MATRIX_LAYOUT matrix_layout_A,
                       Matrix &A,
                       size_t lda,
                       Matrix &s,
                       Matrix &U,
                       size_t ldu,
                       Matrix &V,
                       size_t ldv);
void blas_dgesvd(SVD_OPTIONS jobu,
                 SVD_OPTIONS jobv,
                 size_t m,
                 size_t n,
                 MATRIX_LAYOUT matrix_layout_A,
                 const Matrix &A,
                 size_t lda,
                 Matrix &s,
                 Matrix &U,
                 size_t ldu,
                 Matrix &V,
                 size_t ldv);
void omp_dgesvd(SVD_OPTIONS jobu,
                SVD_OPTIONS jobv,
                size_t m,
                size_t n,
                MATRIX_LAYOUT matrix_layout_A,
                Matrix &A,
                size_t lda,
                Matrix &s,
                Matrix &U,
                size_t ldu,
                Matrix &V,
                size_t ldv);

/**
 *
 * @param jobu
 * @param jobv
 * @param m
 * @param n
 * @param A Matrix in a column major order
 * @param lda
 * @param s
 * @param U
 * @param ldu
 * @param V
 * @param ldv
 */
void omp_mpi_dgesvd(SVD_OPTIONS jobu,
                SVD_OPTIONS jobv,
                size_t m,
                size_t n,
                MatrixMPI &A,
                size_t lda,
                MatrixMPI &s,
                MatrixMPI &V,
                size_t ldv);
}

#endif //SVD_JACOBI_MPI_OMP_LIB_JACOBIMETHODS_H_
