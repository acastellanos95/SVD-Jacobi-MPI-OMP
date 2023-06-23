//
// Created by andre on 5/31/23.
//

#ifndef SVD_JACOBI_MPI_OMP_LIB_GLOBAL_H_
#define SVD_JACOBI_MPI_OMP_LIB_GLOBAL_H_

//#define LAPACK

// For double precision accuracy in the eigenvalues and eigenvectors, a tolerance of order 10−16 will suffice. Erricos
#define TOLERANCE 1e-20

#define ROOT_RANK 0

#define iteratorR(i,j,ld)(((i)*(ld))+(j))
#define iteratorC(i,j,ld)(((j)*(ld))+(i))

//#define TESTS

#endif //SVD_JACOBI_MPI_OMP_LIB_GLOBAL_H_
