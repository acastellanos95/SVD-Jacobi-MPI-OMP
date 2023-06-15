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

void Thesis::Tests::test_local_matrix_distribution_in_sublocal_matrices(size_t m, size_t n, MatrixMPI &A, size_t lda) {
  // Get rank of mpi proccess and size of process
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* --------------------------------------- Test A distribution ----------------------------------------------*/
  MatrixMPI A_test(m, n);
  std::fill_n(A_test.elements, A_test.height * A_test.width, 0.0);
  // Create R matrix
  std::default_random_engine e(1000000);
  std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
  for (size_t indexRow = 0; indexRow < std::min<size_t>(m, n); ++indexRow) {
    for (size_t indexCol = indexRow; indexCol < std::min<size_t>(m, n); ++indexCol) {
      double value = uniform_dist(e);
      A_test.elements[iteratorC(indexRow, indexCol, m)] = value;
    }
  }

  /* --------------------------------------- Jacobi ordering equality ----------------------------------------------*/
  size_t m_ordering = (n + 1) / 2;
  size_t k_ordering_len = n / 2;
  size_t maxIterations = 1;
  for(auto number_iterations = 0; number_iterations < maxIterations; ++number_iterations){
    // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
    // 1971
    for (size_t k = 1; k < m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;

      /* --------------------------------------- Local variables ----------------------------------------------*/
      // All points vector tuple
      std::vector<std::tuple<size_t, size_t>> local_points;
      // Ordered set of coordinates
      std::set<size_t> local_set_columns;
      // Set converted to vector
      std::vector<size_t> local_set_to_vector;
      // Local map that convert point coordinate to local column index
      std::unordered_map<size_t, size_t> column_index_to_local_column_index;
      /* --------------------------------------- Root rank variables ----------------------------------------------*/
      // Distribution of points by node (index is rank, and content is vector of points). To use for data distribution and extraction.
      std::vector<std::vector<std::tuple<size_t, size_t>>> root_distribution(size);
      // ordered set of points by rank. To use for data distribution and extraction.
      std::vector<std::set<size_t>> root_set_columns_by_rank(size);
      // convert to vector to map
      std::vector<std::vector<size_t>> root_set_columns_vector_by_rank(size);
      // Assign column index to set index.
      std::vector<std::unordered_map<size_t, size_t>> root_column_index_to_local_column_index(size);


      /* ------------------------------------- Query points for k -------------------------------------------------*/
      #pragma omp parallel for num_threads(size) private(p, p_trans, q_trans)
      for (size_t q = m_ordering - k + 1; q <= n - k; ++q) {
        if (m_ordering - k + 1 <= q && q <= (2 * m_ordering) - (2 * k)) {
          p = ((2 * m_ordering) - (2 * k) + 1) - q;
        } else if ((2 * m_ordering) - (2 * k) < q && q <= (2 * m_ordering) - k - 1) {
          p = ((4 * m_ordering) - (2 * k)) - q;
        } else if ((2 * m_ordering) - k - 1 < q) {
          p = n;
        }

        // Translate to (0,0)
        p_trans = p - 1;
        q_trans = q - 1;

        if(rank == ROOT_RANK){
//          std::cout << "(" << p_trans << ", " << q_trans << ")\n";
//          std::cout << "(" << omp_get_thread_num() << ")\n";
          root_distribution[omp_get_thread_num()].emplace_back(p_trans, q_trans);
          root_set_columns_by_rank[omp_get_thread_num()].insert(p_trans);
          root_set_columns_by_rank[omp_get_thread_num()].insert(q_trans);
        }

        if(rank == omp_get_thread_num()){
//          std::cout << "local point rank: " << rank << ", (" << p_trans << ", " << q_trans << ")\n";
          local_points.emplace_back(p_trans,q_trans);
          local_set_columns.insert(p_trans);
          local_set_columns.insert(q_trans);
        }
      }

      // convert local set to vector
      local_set_to_vector = std::vector<size_t>(local_set_columns.begin(), local_set_columns.end());
      // map coordinates to local column indices
      size_t local_set_to_vector_size = local_set_to_vector.size();
      for(auto i = 0; i < local_set_to_vector_size; ++i){
        column_index_to_local_column_index[local_set_to_vector[i]] = i;
      }

      if(rank == ROOT_RANK){
        for(auto index_rank = 0; index_rank < size; ++index_rank){
          root_set_columns_vector_by_rank[index_rank] = std::vector<size_t>(root_set_columns_by_rank[index_rank].begin(), root_set_columns_by_rank[index_rank].end());
          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            root_column_index_to_local_column_index[index_rank][root_set_columns_vector_by_rank[index_rank][index_column]] = index_column;
          }
        }
      }

      // Create local matrix
      MatrixMPI A_local(m, local_set_columns.size());

      /* ------------------------------------- Distribute A -------------------------------------------------*/
      if(rank == ROOT_RANK){

        // Create matrix by rank and send
        for(auto index_rank = 0; index_rank < size; ++index_rank){
          std::vector<double> tmp_matrix;
          for(auto column_index: root_set_columns_vector_by_rank[index_rank]){
            tmp_matrix.insert(tmp_matrix.end(), A.elements + m*column_index, A.elements + (m*(column_index + 1)));
          }
          MatrixMPI A_rank(tmp_matrix, m, root_set_columns_by_rank[index_rank].size());

          tmp_matrix.clear();

          if(index_rank == ROOT_RANK){
            A_local = MatrixMPI(A_rank);
          } else {
            auto return_status = MPI_Send(A_rank.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD);
            if(return_status != MPI_SUCCESS){
              std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
            }
          }

          A_rank.free();
        }
      } else {
        MPI_Status status;
        auto return_status = MPI_Recv(A_local.elements, m * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
        }
      }

      /* ------------------------------------- Check distribution send correctness -------------------------------------------------*/
      /*// Calculate frobenius norm of A_local - A
      double frobenius_norm = 0.0;
      // Iterate columns assigned to this rank
      for(auto column_index: local_set_columns){
        for(size_t index_row = 0; index_row < m; ++index_row){
          double sustraction = (A_local.elements[iteratorC(index_row, column_index_to_local_column_index[column_index], m)] - A_test.elements[iteratorC(index_row, column_index, m)]);
          frobenius_norm += sustraction * sustraction;
        }
      }*/

//      std::cout << "local ||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";

      /* ----------------------------------- Gather local solutions ------------------------------------------------*/
      if(rank == ROOT_RANK){
        for(size_t index_rank = 1; index_rank < size; ++index_rank){
          // Create local matrix
          MatrixMPI A_gather(m, root_set_columns_by_rank[index_rank].size());

          MPI_Status status;
          auto return_status = MPI_Recv(A_gather.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD, &status);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
          }

          /* ------------------------------------- Check distribution receive correctness -------------------------------------------------*/
          // Calculate frobenius norm of A_local - A
          /*double frobenius_norm_received = 0.0;
          // Iterate columns assigned to this rank
          for(auto column_index: root_set_columns_by_rank[index_rank]){
            for(size_t index_row = 0; index_row < m; ++index_row){
              double sustraction = (A_gather.elements[iteratorC(index_row, root_column_index_to_local_column_index[index_rank][column_index], m)] - A_test.elements[iteratorC(index_row, column_index, m)]);
              frobenius_norm_received += sustraction * sustraction;
            }
          }*/

//          std::cout << "received ||A-USVt||_F: " << sqrt(frobenius_norm_received) << "\n";

          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
          #pragma omp parallel for
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            for(size_t index_row = 0; index_row < m; ++index_row){
              A.elements[iteratorC(index_row, root_set_columns_vector_by_rank[index_rank][index_column], lda)] = A_gather.elements[iteratorC(index_row, index_column, lda)];
            }
          }

          A_gather.free();
        }

        // Root rank resolve
        size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[0].size();
        #pragma omp parallel for
        for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
          for(size_t index_row = 0; index_row < m; ++index_row){
            A.elements[iteratorC(index_row, root_set_columns_vector_by_rank[0][index_column], lda)] = A_local.elements[iteratorC(index_row, index_column, lda)];
          }
        }

        /*// Calculate frobenius norm of A_local - A
        double frobenius_norm_special = 0.0;
        // Iterate columns assigned to this rank
        for(auto column_index: local_set_columns){
          for(size_t index_row = 0; index_row < m; ++index_row){
            double sustraction = (A_local.elements[iteratorC(index_row, column_index_to_local_column_index[column_index], m)] - A_test.elements[iteratorC(index_row, column_index, m)]);
            frobenius_norm_special += sustraction * sustraction;
          }
        }*/

//        std::cout << "received ||A-USVt||_F: " << sqrt(frobenius_norm_special) << "\n";

      } else {
        auto return_status = MPI_Send(A_local.elements, m * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
        }
      }

      // Free local matrix
      A_local.free();
    }

    for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;

      /* --------------------------------------- Local variables ----------------------------------------------*/
      // All points vector tuple
      std::vector<std::tuple<size_t, size_t>> local_points;
      // Ordered set of coordinates
      std::set<size_t> local_set_columns;
      // Set converted to vector
      std::vector<size_t> local_set_to_vector;
      // Local map that convert point coordinate to local column index
      std::unordered_map<size_t, size_t> column_index_to_local_column_index;
      /* --------------------------------------- Root rank variables ----------------------------------------------*/
      // Distribution of points by node (index is rank, and content is vector of points). To use for data distribution and extraction.
      std::vector<std::vector<std::tuple<size_t, size_t>>> root_distribution(size);
      // ordered set of points by rank. To use for data distribution and extraction.
      std::vector<std::set<size_t>> root_set_columns_by_rank(size);
      // convert to vector to map
      std::vector<std::vector<size_t>> root_set_columns_vector_by_rank(size);
      // Assign column index to set index.
      std::vector<std::unordered_map<size_t, size_t>> root_column_index_to_local_column_index(size);


      /* ------------------------------------- Query points for k -------------------------------------------------*/
#pragma omp parallel for num_threads(size) private(p, p_trans, q_trans)
      for (size_t q = (4 * m_ordering) - n - k; q < (3 * m_ordering) - k; ++q) {
        if (q < (2 * m_ordering) - k + 1) {
          p = n;
        } else if ((2 * m_ordering) - k + 1 <= q && q <= (4 * m_ordering) - (2 * k) - 1) {
          p = ((4 * m_ordering) - (2 * k)) - q;
        } else if ((4 * m_ordering) - (2 * k) - 1 < q) {
          p = ((6 * m_ordering) - (2 * k) - 1) - q;
        }

        // Translate to (0,0)
        p_trans = p - 1;
        q_trans = q - 1;

        if(rank == ROOT_RANK){
//          std::cout << "(" << p_trans << ", " << q_trans << ")\n";
//          std::cout << "(" << omp_get_thread_num() << ")\n";
          root_distribution[omp_get_thread_num()].emplace_back(p_trans, q_trans);
          root_set_columns_by_rank[omp_get_thread_num()].insert(p_trans);
          root_set_columns_by_rank[omp_get_thread_num()].insert(q_trans);
        }

        if(rank == omp_get_thread_num()){
//          std::cout << "local point rank: " << rank << ", (" << p_trans << ", " << q_trans << ")\n";
          local_points.emplace_back(p_trans,q_trans);
          local_set_columns.insert(p_trans);
          local_set_columns.insert(q_trans);
        }
      }

      // convert local set to vector
      local_set_to_vector = std::vector<size_t>(local_set_columns.begin(), local_set_columns.end());
      // map coordinates to local column indices
      size_t local_set_to_vector_size = local_set_to_vector.size();
      for(auto i = 0; i < local_set_to_vector_size; ++i){
        column_index_to_local_column_index[local_set_to_vector[i]] = i;
      }

      if(rank == ROOT_RANK){
        for(auto index_rank = 0; index_rank < size; ++index_rank){
          root_set_columns_vector_by_rank[index_rank] = std::vector<size_t>(root_set_columns_by_rank[index_rank].begin(), root_set_columns_by_rank[index_rank].end());
          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            root_column_index_to_local_column_index[index_rank][root_set_columns_vector_by_rank[index_rank][index_column]] = index_column;
          }
        }
      }

      // Create local matrix
      MatrixMPI A_local(m, local_set_columns.size());

      /* ------------------------------------- Distribute A -------------------------------------------------*/
      if(rank == ROOT_RANK){

        // Create matrix by rank and send
        for(auto index_rank = 0; index_rank < size; ++index_rank){
          std::vector<double> tmp_matrix;
          for(auto column_index: root_set_columns_vector_by_rank[index_rank]){
            tmp_matrix.insert(tmp_matrix.end(), A.elements + m*column_index, A.elements + (m*(column_index + 1)));
          }
/*

//          std::cout << "send rank: " << index_rank << ", local matrix size: " << tmp_matrix.size() << ", expected matrix size: " << m * root_set_columns_by_rank[index_rank].size() << '\n';
//
//          for(auto index_row = 0; index_row < m; ++index_row){
//            size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
//            for(auto index_col = 0; index_col < root_set_columns_vector_by_rank_for_index_rank_size; ++index_col){
//              std::cout << tmp_matrix[iteratorC(index_row, index_col, m)] << ", ";
//            }
//            std::cout << "\n";
//          }
*/

          MatrixMPI A_rank(tmp_matrix, m, root_set_columns_by_rank[index_rank].size());

          tmp_matrix.clear();
/*
//          std::cout << "rank: " << index_rank << ", local matrix size: " << tmp_matrix.size() << ", expected matrix size: " << m * root_set_columns_by_rank[index_rank].size() << '\n';
//
//          for(auto index_row = 0; index_row < m; ++index_row){
//            for(auto index_col = 0; index_col < root_set_columns_vector_by_rank[index_rank].size(); ++index_col){
//              std::cout << A_rank.elements[iteratorC(index_row, index_col, m)] << ", ";
//            }
//            std::cout << "\n";
//          }
*/

          if(index_rank == ROOT_RANK){
            A_local = MatrixMPI(A_rank);
            /*
//            std::cout << "receive rank: " << rank << ", local matrix size: " << A_local.height * A_local.width << ", expected matrix size: " << m * local_set_columns.size() << '\n';
//
//            for(auto index_row = 0; index_row < m; ++index_row){
//              for(auto index_col = 0; index_col < local_set_columns.size(); ++index_col){
//                std::cout << A_local.elements[iteratorC(index_row, index_col, m)] << ", ";
//              }
//              std::cout << "\n";
//            }*/
          } else {
            auto return_status = MPI_Send(A_rank.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD);
            if(return_status != MPI_SUCCESS){
              std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
            }
          }

          A_rank.free();
        }
      } else {
        MPI_Status status;
        auto return_status = MPI_Recv(A_local.elements, m * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
        }
      }
/*
//      std::cout << "receive rank: " << rank << ", local matrix size: " << A_local.height * A_local.width << ", expected matrix size: " << m * local_set_columns.size() << '\n';
//
//      for(auto index_row = 0; index_row < m; ++index_row){
//        for(auto index_col = 0; index_col < local_set_columns.size(); ++index_col){
//          std::cout << A_local.elements[iteratorC(index_row, index_col, m)] << ", ";
//        }
//        std::cout << "\n";
//      }
 */

      /* ------------------------------------- Check distribution send correctness -------------------------------------------------*/
      /*// Calculate frobenius norm of A_local - A
      double frobenius_norm = 0.0;
      // Iterate columns assigned to this rank
      for(auto column_index: local_set_columns){
        for(size_t index_row = 0; index_row < m; ++index_row){
          double sustraction = (A_local.elements[iteratorC(index_row, column_index_to_local_column_index[column_index], m)] - A_test.elements[iteratorC(index_row, column_index, m)]);
          frobenius_norm += sustraction * sustraction;
        }
      }*/

//      std::cout << "local ||A-USVt||_F: " << sqrt(frobenius_norm) << "\n";
      /* ----------------------------------- Gather local solutions ------------------------------------------------*/
      if(rank == ROOT_RANK){
        for(size_t index_rank = 1; index_rank < size; ++index_rank){
          // Create local matrix
          MatrixMPI A_gather(m, root_set_columns_by_rank[index_rank].size());

          MPI_Status status;
          auto return_status = MPI_Recv(A_gather.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD, &status);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
          }

          /* ------------------------------------- Check distribution receive correctness -------------------------------------------------*/
          /*// Calculate frobenius norm of A_local - A
          double frobenius_norm_received = 0.0;
          // Iterate columns assigned to this rank
          for(auto column_index: root_set_columns_by_rank[index_rank]){
            for(size_t index_row = 0; index_row < m; ++index_row){
              double sustraction = (A_gather.elements[iteratorC(index_row, root_column_index_to_local_column_index[index_rank][column_index], m)] - A_test.elements[iteratorC(index_row, column_index, m)]);
              frobenius_norm_received += sustraction * sustraction;
            }
          }*/

//          std::cout << "received ||A-USVt||_F: " << sqrt(frobenius_norm_received) << "\n";

          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
#pragma omp parallel for
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            for(size_t index_row = 0; index_row < m; ++index_row){
              A.elements[iteratorC(index_row, root_set_columns_vector_by_rank[index_rank][index_column], lda)] = A_gather.elements[iteratorC(index_row, index_column, lda)];
            }
          }

          A_gather.free();
        }

        // Root rank resolve
        size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[0].size();
#pragma omp parallel for
        for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
          for(size_t index_row = 0; index_row < m; ++index_row){
            A.elements[iteratorC(index_row, root_set_columns_vector_by_rank[0][index_column], lda)] = A_local.elements[iteratorC(index_row, index_column, lda)];
          }
        }

        /*// Calculate frobenius norm of A_local - A
        double frobenius_norm_special = 0.0;
        // Iterate columns assigned to this rank
        for(auto column_index: local_set_columns){
          for(size_t index_row = 0; index_row < m; ++index_row){
            double sustraction = (A_local.elements[iteratorC(index_row, column_index_to_local_column_index[column_index], m)] - A_test.elements[iteratorC(index_row, column_index, m)]);
            frobenius_norm_special += sustraction * sustraction;
          }
        }*/

//        std::cout << "received ||A-USVt||_F: " << sqrt(frobenius_norm_special) << "\n";
      } else {
        auto return_status = MPI_Send(A_local.elements, m * local_set_columns.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
        }
      }

      // Free local matrix
      A_local.free();
    }

  }

  A_test.free();
}

void Thesis::Tests::test_local_matrix_distribution_on_the_fly(size_t m, size_t n, MatrixMPI &A, size_t lda) {
  // Get rank of mpi proccess and size of process
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* --------------------------------------- Test A distribution ----------------------------------------------*/
  MatrixMPI A_test(m, n);
  std::fill_n(A_test.elements, A_test.height * A_test.width, 0.0);
  // Create R matrix
  std::default_random_engine e(1000000);
  std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
  for (size_t indexRow = 0; indexRow < std::min<size_t>(m, n); ++indexRow) {
    for (size_t indexCol = indexRow; indexCol < std::min<size_t>(m, n); ++indexCol) {
      double value = uniform_dist(e);
      A_test.elements[iteratorC(indexRow, indexCol, m)] = value;
    }
  }

  /* --------------------------------------- Jacobi ordering equality ----------------------------------------------*/
  size_t m_ordering = (n + 1) / 2;
  size_t k_ordering_len = n / 2;
  size_t maxIterations = 1;
  for(auto number_iterations = 0; number_iterations < maxIterations; ++number_iterations){
    // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
    // 1971
    for (size_t k = 1; k < m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;

      /* --------------------------------------- Local variables ----------------------------------------------*/
      // All points vector tuple
      std::vector<std::tuple<size_t, size_t>> local_points;
      // Ordered set of coordinates
      std::set<size_t> local_set_columns;
      // Set converted to vector
      std::vector<size_t> local_set_to_vector;
      // Local map that convert point coordinate to local column index
      std::unordered_map<size_t, size_t> column_index_to_local_column_index;
      /* --------------------------------------- Root rank variables ----------------------------------------------*/
      // Get rank for column index
      std::map<size_t, size_t> root_index_column_to_rank;
      // Distribution of points by node (index is rank, and content is vector of points). To use for data distribution and extraction.
      std::vector<std::vector<std::tuple<size_t, size_t>>> root_distribution(size);
      // ordered set of points by rank. To use for data distribution and extraction.
      std::vector<std::set<size_t>> root_set_columns_by_rank(size);
      // convert to vector to map
      std::vector<std::vector<size_t>> root_set_columns_vector_by_rank(size);
      // Assign column index to set index.
      std::vector<std::unordered_map<size_t, size_t>> root_column_index_to_local_column_index(size);


      /* ------------------------------------- Query points for k -------------------------------------------------*/
      #pragma omp parallel for num_threads(size) private(p, p_trans, q_trans)
      for (size_t q = m_ordering - k + 1; q <= n - k; ++q) {
        if (m_ordering - k + 1 <= q && q <= (2 * m_ordering) - (2 * k)) {
          p = ((2 * m_ordering) - (2 * k) + 1) - q;
        } else if ((2 * m_ordering) - (2 * k) < q && q <= (2 * m_ordering) - k - 1) {
          p = ((4 * m_ordering) - (2 * k)) - q;
        } else if ((2 * m_ordering) - k - 1 < q) {
          p = n;
        }

        // Translate to (0,0)
        p_trans = p - 1;
        q_trans = q - 1;

        if(rank == ROOT_RANK){
//          std::cout << "(" << p_trans << ", " << q_trans << ")\n";
//          std::cout << "(" << omp_get_thread_num() << ")\n";
          root_distribution[omp_get_thread_num()].emplace_back(p_trans, q_trans);
          root_set_columns_by_rank[omp_get_thread_num()].insert(p_trans);
          root_set_columns_by_rank[omp_get_thread_num()].insert(q_trans);
          if(omp_get_thread_num() != 0){
            root_index_column_to_rank[p_trans] = omp_get_thread_num();
            root_index_column_to_rank[q_trans] = omp_get_thread_num();
          }
        }

        if(rank == omp_get_thread_num()){
//          std::cout << "local point rank: " << rank << ", (" << p_trans << ", " << q_trans << ")\n";
          local_points.emplace_back(p_trans,q_trans);
          local_set_columns.insert(p_trans);
          local_set_columns.insert(q_trans);
        }
      }

      // convert local set to vector
      local_set_to_vector = std::vector<size_t>(local_set_columns.begin(), local_set_columns.end());
      // map coordinates to local column indices
      size_t local_set_to_vector_size = local_set_to_vector.size();
      for(auto i = 0; i < local_set_to_vector_size; ++i){
        column_index_to_local_column_index[local_set_to_vector[i]] = i;
      }

      if(rank == ROOT_RANK){
        for(auto index_rank = 0; index_rank < size; ++index_rank){
          root_set_columns_vector_by_rank[index_rank] = std::vector<size_t>(root_set_columns_by_rank[index_rank].begin(), root_set_columns_by_rank[index_rank].end());
          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
            root_column_index_to_local_column_index[index_rank][root_set_columns_vector_by_rank[index_rank][index_column]] = index_column;
          }
        }
      }

      // Create local matrix
//      MatrixMPI A_local(m, local_set_columns.size());

      /* ------------------------------------- Distribute and solve A -------------------------------------------------*/
      if(rank == ROOT_RANK){
        MPI_Request* requests = new MPI_Request[root_index_column_to_rank.size()];
        MPI_Status* statuses = new MPI_Status[root_index_column_to_rank.size()];
        size_t index_request = 0;
        // Create matrix by rank and send
        for(auto column_index: root_index_column_to_rank){
          std::cout << "sending column: " << column_index.first << " to rank: " << column_index.second << '\n';
          auto return_status = MPI_Isend(A.elements + (m * column_index.first), m, MPI_DOUBLE, column_index.second, column_index.first, MPI_COMM_WORLD, &requests[index_request++]);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_ISend on rank: " << rank << ", return status" << return_status << "\n";
          }
          std::cout << "sent column: " << column_index.first << " to rank: " << column_index.second << '\n';
        }

        auto return_status = MPI_Waitall(root_index_column_to_rank.size(), requests, statuses);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Waitall on rank: " << rank << ", return status" << return_status << "\n";
        }

        delete[] statuses;
        delete[] requests;
        /*
        for(auto index_rank = 1; index_rank < size; ++index_rank){
          std::vector<double> tmp_matrix;
          for(auto column_index: root_set_columns_vector_by_rank[index_rank]){
            tmp_matrix.insert(tmp_matrix.end(), A.elements + m*column_index, A.elements + (m*(column_index + 1)));
          }
          MatrixMPI A_rank(tmp_matrix, m, root_set_columns_by_rank[index_rank].size());

          tmp_matrix.clear();

          auto return_status = MPI_Send(A_rank.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
          }

          A_rank.free();
        }
        */
      } else {
        MPI_Request* requests = new MPI_Request[2*local_points.size()];
        MPI_Status* statuses = new MPI_Status[2*local_points.size()];
        size_t index_request = 0;

        for(auto &point: local_points){
          MatrixMPI p_vector(1, m), q_vector(1, m);
          MPI_Status status;
          size_t p_index = std::get<0>(point);
          size_t q_index = std::get<1>(point);
          std::cout << "receiving column: " << p_index << ", " << q_index << " in rank: " << rank << ", from rank: " << 0 << '\n';
          auto return_status = MPI_Recv(p_vector.elements, m, MPI_DOUBLE, 0, p_index, MPI_COMM_WORLD, &status);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
          }
          return_status = MPI_Recv(q_vector.elements, m, MPI_DOUBLE, 0, q_index, MPI_COMM_WORLD, &status);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
          }
          std::cout << "received column: " << p_index << ", " << q_index << " in rank: " << rank << ", from rank: " << 0 << '\n';

          /* ------------------------------------- Check distribution send correctness -------------------------------------------------*/
          // Calculate frobenius norm of A_local - A
          double frobenius_norm = 0.0;
          // Iterate columns assigned to this rank
          for(size_t index_row = 0; index_row < m; ++index_row){
            double sustraction = (A_test.elements[iteratorC(index_row, p_index, m)] - p_vector.elements[index_row]);
            frobenius_norm += sustraction * sustraction;
            sustraction = (A_test.elements[iteratorC(index_row, q_index, m)] - q_vector.elements[index_row]);
            frobenius_norm += sustraction * sustraction;
          }

          std::cout << "received in not root rank ||Apq-A_local_pq||_F: " << sqrt(frobenius_norm) << "\n";

          /* ------------------------------------- Solve -------------------------------------------------*/
          // Do something

          std::cout << "sending column: " << p_index << " to rank: " << 0 << '\n';
          return_status = MPI_Isend(p_vector.elements, m, MPI_DOUBLE, 0, p_index, MPI_COMM_WORLD, &requests[index_request++]);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_ISend on rank: " << rank << ", return status" << return_status << "\n";
          }
          std::cout << "sent column: " << p_index << " to rank: " << 0 << '\n';

          std::cout << "sending column: " << q_index << " to rank: " << 0 << '\n';
          return_status = MPI_Isend(q_vector.elements, m, MPI_DOUBLE, 0, q_index, MPI_COMM_WORLD, &requests[index_request++]);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
          }
          std::cout << "sent column: " << q_index << " to rank: " << 0 << '\n';

          p_vector.free();
          q_vector.free();
        }

        auto return_status = MPI_Waitall(2*local_points.size(), requests, statuses);
        if(return_status != MPI_SUCCESS){
          std::cout << "problem on MPI_Waitall on rank: " << rank << ", return status" << return_status << "\n";
        }

        delete[] requests;
        delete[] statuses;
      }

      /* ----------------------------------- Gather local solutions ------------------------------------------------*/
      if(rank == ROOT_RANK){
        for(auto column_index: root_index_column_to_rank){
          MatrixMPI column_vector(1, m);
          MPI_Status status;
          std::cout << "receiving column: " << column_index.first << " from rank: " << column_index.second << " in rank " << 0 << '\n';
          auto return_status = MPI_Recv(column_vector.elements, m, MPI_DOUBLE, column_index.second, column_index.first, MPI_COMM_WORLD, &status);
          if(return_status != MPI_SUCCESS){
            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
          }
          std::cout << "received column: " << column_index.first << " from rank: " << column_index.second << " in rank " << 0 << '\n';

          /* ------------------------------------- Check distribution send correctness -------------------------------------------------*/
          // Calculate frobenius norm of A_local - A
          double frobenius_norm = 0.0;
          for(size_t index_row = 0; index_row < m; ++index_row){
            double sustraction = (A_test.elements[iteratorC(index_row, column_index.first, m)] - column_vector.elements[index_row]);
            frobenius_norm += sustraction * sustraction;
          }

          std::cout << "received in root rank ||Apq-A_local_pq||_F: " << sqrt(frobenius_norm) << "\n";

          /* ------------------------------------- Return to A -------------------------------------------------*/

          column_vector.free();
        }

      }

      MPI_Barrier(MPI_COMM_WORLD);

      // Free local matrix
//      A_local.free();
    }

//    for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
//      size_t p = 0;
//      size_t p_trans = 0;
//      size_t q_trans = 0;
//
//      /* --------------------------------------- Local variables ----------------------------------------------*/
//      // All points vector tuple
//      std::vector<std::tuple<size_t, size_t>> local_points;
//      // Ordered set of coordinates
//      std::set<size_t> local_set_columns;
//      // Set converted to vector
//      std::vector<size_t> local_set_to_vector;
//      // Local map that convert point coordinate to local column index
//      std::unordered_map<size_t, size_t> column_index_to_local_column_index;
//      /* --------------------------------------- Root rank variables ----------------------------------------------*/
//      // Get rank for column index
//      std::map<size_t, size_t> root_index_column_to_rank;
//      // Distribution of points by node (index is rank, and content is vector of points). To use for data distribution and extraction.
//      std::vector<std::vector<std::tuple<size_t, size_t>>> root_distribution(size);
//      // ordered set of points by rank. To use for data distribution and extraction.
//      std::vector<std::set<size_t>> root_set_columns_by_rank(size);
//      // convert to vector to map
//      std::vector<std::vector<size_t>> root_set_columns_vector_by_rank(size);
//      // Assign column index to set index.
//      std::vector<std::unordered_map<size_t, size_t>> root_column_index_to_local_column_index(size);
//
//
//      /* ------------------------------------- Query points for k -------------------------------------------------*/
//#pragma omp parallel for num_threads(size) private(p, p_trans, q_trans)
//      for (size_t q = (4 * m_ordering) - n - k; q < (3 * m_ordering) - k; ++q) {
//        if (q < (2 * m_ordering) - k + 1) {
//          p = n;
//        } else if ((2 * m_ordering) - k + 1 <= q && q <= (4 * m_ordering) - (2 * k) - 1) {
//          p = ((4 * m_ordering) - (2 * k)) - q;
//        } else if ((4 * m_ordering) - (2 * k) - 1 < q) {
//          p = ((6 * m_ordering) - (2 * k) - 1) - q;
//        }
//
//        // Translate to (0,0)
//        p_trans = p - 1;
//        q_trans = q - 1;
//
//        if(rank == ROOT_RANK){
////          std::cout << "(" << p_trans << ", " << q_trans << ")\n";
////          std::cout << "(" << omp_get_thread_num() << ")\n";
//          root_distribution[omp_get_thread_num()].emplace_back(p_trans, q_trans);
//          root_set_columns_by_rank[omp_get_thread_num()].insert(p_trans);
//          root_set_columns_by_rank[omp_get_thread_num()].insert(q_trans);
//          root_index_column_to_rank[p_trans] = omp_get_thread_num();
//          root_index_column_to_rank[q_trans] = omp_get_thread_num();
//        }
//
//        if(rank == omp_get_thread_num()){
////          std::cout << "local point rank: " << rank << ", (" << p_trans << ", " << q_trans << ")\n";
//          local_points.emplace_back(p_trans,q_trans);
//          local_set_columns.insert(p_trans);
//          local_set_columns.insert(q_trans);
//        }
//      }
//
//      // convert local set to vector
//      local_set_to_vector = std::vector<size_t>(local_set_columns.begin(), local_set_columns.end());
//      // map coordinates to local column indices
//      size_t local_set_to_vector_size = local_set_to_vector.size();
//      for(auto i = 0; i < local_set_to_vector_size; ++i){
//        column_index_to_local_column_index[local_set_to_vector[i]] = i;
//      }
//
//      if(rank == ROOT_RANK){
//        for(auto index_rank = 0; index_rank < size; ++index_rank){
//          root_set_columns_vector_by_rank[index_rank] = std::vector<size_t>(root_set_columns_by_rank[index_rank].begin(), root_set_columns_by_rank[index_rank].end());
//          size_t root_set_columns_vector_by_rank_for_index_rank_size = root_set_columns_vector_by_rank[index_rank].size();
//          for(size_t index_column = 0; index_column < root_set_columns_vector_by_rank_for_index_rank_size; ++index_column){
//            root_column_index_to_local_column_index[index_rank][root_set_columns_vector_by_rank[index_rank][index_column]] = index_column;
//          }
//        }
//      }
//
//      // Create local matrix
////      MatrixMPI A_local(m, local_set_columns.size());
//
//      /* ------------------------------------- Distribute and solve A -------------------------------------------------*/
//      if(rank == ROOT_RANK){
//        // Create matrix by rank and send
//        for(auto column_index: root_index_column_to_rank){
//          MPI_Request request;
//          auto return_status = MPI_Isend(A.elements + (m * column_index.first), m, MPI_DOUBLE, column_index.second, column_index.first, MPI_COMM_WORLD, &request);
//          if(return_status != MPI_SUCCESS){
//            std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
//          }
//        }
//        /*
//        for(auto index_rank = 1; index_rank < size; ++index_rank){
//          std::vector<double> tmp_matrix;
//          for(auto column_index: root_set_columns_vector_by_rank[index_rank]){
//            tmp_matrix.insert(tmp_matrix.end(), A.elements + m*column_index, A.elements + (m*(column_index + 1)));
//          }
//          MatrixMPI A_rank(tmp_matrix, m, root_set_columns_by_rank[index_rank].size());
//
//          tmp_matrix.clear();
//
//          auto return_status = MPI_Send(A_rank.elements, m * root_set_columns_by_rank[index_rank].size(), MPI_DOUBLE, index_rank, 0, MPI_COMM_WORLD);
//          if(return_status != MPI_SUCCESS){
//            std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
//          }
//
//          A_rank.free();
//        }
//        */
//      } else {
//        for(auto &point: local_points){
//          MatrixMPI p_vector(1, m), q_vector(1, m);
//          MPI_Status status;
//          size_t p_index = std::get<0>(point);
//          size_t q_index = std::get<1>(point);
//          auto return_status = MPI_Recv(p_vector.elements, m, MPI_DOUBLE, 0, p_index, MPI_COMM_WORLD, &status);
//          if(return_status != MPI_SUCCESS){
//            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
//          }
//          return_status = MPI_Recv(q_vector.elements, m, MPI_DOUBLE, 0, q_index, MPI_COMM_WORLD, &status);
//          if(return_status != MPI_SUCCESS){
//            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
//          }
//
//          /* ------------------------------------- Check distribution send correctness -------------------------------------------------*/
//          // Calculate frobenius norm of A_local - A
//          double frobenius_norm = 0.0;
//          // Iterate columns assigned to this rank
//          for(size_t index_row = 0; index_row < m; ++index_row){
//            double sustraction = (A_test.elements[iteratorC(index_row, p_index, m)] - p_vector.elements[index_row]);
//            frobenius_norm += sustraction * sustraction;
//            sustraction = (A_test.elements[iteratorC(index_row, q_index, m)] - q_vector.elements[index_row]);
//            frobenius_norm += sustraction * sustraction;
//          }
//
//          std::cout << "received in not root rank ||Apq-A_local_pq||_F: " << sqrt(frobenius_norm) << "\n";
//
//          /* ------------------------------------- Solve -------------------------------------------------*/
//          // Do something
//
//          MPI_Request p_request;
//          return_status = MPI_Isend(p_vector.elements, m, MPI_DOUBLE, 0, p_index, MPI_COMM_WORLD, &p_request);
//          if(return_status != MPI_SUCCESS){
//            std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
//          }
//
//          MPI_Request q_request;
//          return_status = MPI_Isend(q_vector.elements, m, MPI_DOUBLE, 0, q_index, MPI_COMM_WORLD, &q_request);
//          if(return_status != MPI_SUCCESS){
//            std::cout << "problem on MPI_Send on rank: " << rank << ", return status" << return_status << "\n";
//          }
//
//          p_vector.free();
//          q_vector.free();
//        }
//      }
//
//      /* ----------------------------------- Gather local solutions ------------------------------------------------*/
//      if(rank == ROOT_RANK){
//        for(auto column_index: root_index_column_to_rank){
//          std::cout << "sending column: " << column_index.first << " to rank: " << column_index.second << '\n';
//          MatrixMPI column_vector(1, m);
//          MPI_Status status;
//          auto return_status = MPI_Recv(column_vector.elements, m, MPI_DOUBLE, column_index.second, column_index.first, MPI_COMM_WORLD, &status);
//          if(return_status != MPI_SUCCESS){
//            std::cout << "problem on MPI_Recv on rank: " << rank << ", return status" << return_status << "\n";
//          }
//
//          /* ------------------------------------- Check distribution send correctness -------------------------------------------------*/
//          // Calculate frobenius norm of A_local - A
//          double frobenius_norm = 0.0;
//          for(size_t index_row = 0; index_row < m; ++index_row){
//            double sustraction = (A_test.elements[iteratorC(index_row, column_index.first, m)] - column_vector.elements[index_row]);
//            frobenius_norm += sustraction * sustraction;
//          }
//
//          std::cout << "received in root rank ||Apq-A_local_pq||_F: " << sqrt(frobenius_norm) << "\n";
//
//          /* ------------------------------------- Return to A -------------------------------------------------*/
//
//          column_vector.free();
//        }
//
//      }
//
//      // Free local matrix
////      A_local.free();
//    }

  }

  A_test.free();
}
