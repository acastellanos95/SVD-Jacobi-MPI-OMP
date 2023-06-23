//
// Created by andre on 6/1/23.
//

#ifndef SVD_JACOBI_MPI_OMP_TESTS_TESTS_H_
#define SVD_JACOBI_MPI_OMP_TESTS_TESTS_H_

#include <mpi.h>
#include <omp.h>
#include <random>
#include <vector>
#include <set>
#include <unordered_map>
#include <map>
#include "../lib/Matrix.h"
#include "../lib/global.h"
#include "../lib/Utils.h"

namespace Thesis {

class Tests {
 public:
  void static check_mpi_rank();
  void static test_local_matrix_distribution_in_sublocal_matrices_blocking(size_t m,
                                                                  size_t n,
                                                                  MatrixMPI &A,
                                                                  size_t lda);
  void static test_local_matrix_distribution_on_the_fly_blocking(size_t m,
                                                        size_t n,
                                                        MatrixMPI &A,
                                                        size_t lda);
  void static test_local_matrix_distribution_in_sublocal_matrices_concurrent(size_t m,
                                                                  size_t n,
                                                                  MatrixMPI &A,
                                                                  size_t lda);
  void static test_local_matrix_distribution_on_the_fly_concurrent(size_t m,
                                                        size_t n,
                                                        MatrixMPI &A,
                                                        size_t lda);
  static void test_MPI_Isend_Recv();
};

}

#endif //SVD_JACOBI_MPI_OMP_TESTS_TESTS_H_
