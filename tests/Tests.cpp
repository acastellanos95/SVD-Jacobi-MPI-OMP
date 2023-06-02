//
// Created by andre on 6/1/23.
//

#include "Tests.h"

void Thesis::Tests::check_mpi_rank() {
  // Get rank of mpi proccess and size of process
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << "rank in function: " << rank << "\n";
//  std::cout << "size in function: " << size << "\n";
}
