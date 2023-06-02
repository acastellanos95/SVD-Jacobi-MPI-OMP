#include <iostream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <omp.h>
#include <fstream>
#include <random>
#include <mkl/mkl.h>
#include "mpi.h"
#include "lib/Matrix.h"
#include "lib/JacobiMethods.h"
#include "lib/Utils.h"
#include "tests/Tests.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // SEED!!!
  const unsigned seed = 1000000;

  time_t t;
  tm tm;
  std::stringstream file_output;
  std::ostringstream oss;
  std::string now_time;
  size_t height = 100;
  size_t width = 100;

  MatrixMPI A, V, s, A_copy;

  // Select iterator
  auto iterator = Thesis::IteratorC;

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == ROOT_RANK) {
    // Initialize other variables

    t = std::time(nullptr);
    tm = *std::localtime(&t);
    file_output = std::stringstream();
    oss = std::ostringstream();
    now_time = std::string();
    oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    now_time = oss.str();

    // Matrix initialization and fill
    A = MatrixMPI(height, width);
    V = MatrixMPI(width, width);
    s = MatrixMPI(1, std::min(A.height, A.width));
    A_copy = MatrixMPI(height, width);

    std::fill_n(V.elements, V.height * V.width, 0.0);
    std::fill_n(A.elements, A.height * A.width, 0.0);
    std::fill_n(s.elements, s.height * s.width, 0.0);
    std::fill_n(A_copy.elements, A_copy.height * A_copy.width, 0.0);

    // Create R matrix
    std::default_random_engine e(seed);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    for (size_t indexRow = 0; indexRow < std::min<size_t>(height, width); ++indexRow) {
      for (size_t indexCol = indexRow; indexCol < std::min<size_t>(height, width); ++indexCol) {
        double value = uniform_dist(e);
        A.elements[iterator(indexRow, indexCol, height)] = value;
        A_copy.elements[iterator(indexRow, indexCol, height)] = value;
      }
    }

    file_output << "Number of threads: " << omp_get_num_threads() << '\n';
    file_output << "Dimensions, height: " << height << ", width: " << width << "\n";
    std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";
  }

  // Calculate SVD decomposition
  double ti = omp_get_wtime();
  Thesis::omp_mpi_dgesvd(Thesis::AllVec,
                         Thesis::AllVec,
                         height,
                         width,
                         A,
                         height,
                         s,
                         V,
                         width);
  double tf = omp_get_wtime();
  double time = tf - ti;

  if (rank == ROOT_RANK) {

    file_output << "SVD OMP time with U,V calculation: " << time << "\n";
    std::cout << "SVD OMP time with U,V calculation: " << time << "\n";

    double maxError = 0.0;
    #pragma omp parallel for reduction(max:maxError)
    for (size_t indexRow = 0; indexRow < height; ++indexRow) {
      for (size_t indexCol = 0; indexCol < width; ++indexCol) {
        double value = 0.0;
        for (size_t k_dot = 0; k_dot < width; ++k_dot) {
          value += A.elements[iterator(indexRow, k_dot, height)] * (s.elements[k_dot])
              * V.elements[iterator(indexCol, k_dot, height)];
        }
        double diff = std::abs(A_copy.elements[iterator(indexRow, indexCol, height)] - value);
        maxError = std::max<double>(maxError, diff);
      }
    }

    file_output << "max error between A and USV: " << maxError << "\n";
    std::cout << "max error between A and USV: " << maxError << "\n";

    std::ofstream file("reporte-" + now_time + ".txt", std::ofstream::out | std::ofstream::trunc);
    file << file_output.rdbuf();
    file.close();
    A.free(), V.free(), s.free(), A_copy.free();
  }

  MPI_Finalize();

  return 0;
}

/*

int main1(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // SEED!!!
  const unsigned seed = 1000000;

  if(rank == ROOT_RANK){
    size_t begin = 100;
    size_t end = 100;
    size_t delta = 100;
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    auto now_time = oss.str();

    std::stringstream file_output;
    file_output << "Number of threads: " << omp_get_num_threads() << '\n';
    for (; begin <= end; begin += delta) {
      {
        double time_avg = 0.0;
        int number_repetitions = 10;
        for(auto i_repeat = 0; i_repeat < number_repetitions; ++i_repeat){
          {
            // Build matrix A and R
            // -------------------------------- Test 1 (Squared matrix SVD) MKL Computes the singular value decomposition of a real matrix using Jacobi plane rotations. --------------------------------
            file_output
                << "-------------------------------- Test 1 (Squared matrix SVD) MKL Computes the singular value decomposition of a real matrix using Jacobi plane rotations. --------------------------------\n";
            std::cout
                << "-------------------------------- Test 1 (Squared matrix SVD) MKL Computes the singular value decomposition of a real matrix using Jacobi plane rotations. --------------------------------\n";

            const size_t height = begin;
            const size_t width = begin;

            file_output << "Dimensions, height: " << height << ", width: " << width << "\n";
            std::cout << "Dimensions, height: " << height << ", width: " << width << "\n";

            Matrix A(height, width), U(height, height), V(width, width), s(1, std::min(A.height, A.width)), A_copy(height, width);

            const unsigned long A_height = A.height, A_width = A.width;

            std::fill_n(U.elements, U.height * U.width, 0.0);
            std::fill_n(V.elements, V.height * V.width, 0.0);
            std::fill_n(A.elements, A.height * A.width, 0.0);
            std::fill_n(A_copy.elements, A_copy.height * A_copy.width, 0.0);

            // Select iterator
            auto iterator = Thesis::IteratorC;

            // Create R matrix
            std::default_random_engine e(seed);
            std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
            for (size_t indexRow = 0; indexRow < std::min<size_t>(A_height, A_width); ++indexRow) {
              for (size_t indexCol = indexRow; indexCol < std::min<size_t>(A_height, A_width); ++indexCol) {
                double value = uniform_dist(e);
                A.elements[iterator(indexRow, indexCol, A_height)] = value;
                A_copy.elements[iterator(indexRow, indexCol, A_height)] = value;
              }
            }

#ifdef REPORT
            // Report Matrix A
file_output << std::fixed << std::setprecision(3) << "A: \n";
std::cout << std::fixed << std::setprecision(3) << "A: \n";
for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
  for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
    file_output << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
    std::cout << A.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
  }
  file_output << '\n';
  std::cout << '\n';
}
// Report Matrix A^T * A
//    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
//    for (size_t indexRow = 0; indexRow < A.width; ++indexRow) {
//      for (size_t indexCol = 0; indexCol < A.width; ++indexCol) {
//        double value = 0.0;
//        for(size_t k_dot = 0; k_dot < A.height; ++k_dot){
//          value += A.elements[Thesis::IteratorC(k_dot, indexRow, A.height)] * A.elements[Thesis::IteratorC(k_dot, indexCol, A.height)];
//        }
//        std::cout << value << " ";
//      }
//      std::cout << '\n';
//    }
#endif

            // Calculate SVD decomposition
            double ti = omp_get_wtime();
            Thesis::omp_mpi_dgesvd(Thesis::AllVec,
                               Thesis::AllVec,
                               A.height,
                               A.width,
                               A,
                               A_height,
                               s,
                               U,
                               A_height,
                               V,
                               A_width);
            double tf = omp_get_wtime();
            double time = tf - ti;
            time_avg += time;

            file_output << "SVD OMP time with U,V calculation: " << time << "\n";
            std::cout << "SVD OMP time with U,V calculation: " << time << "\n";

          double maxError = 0.0;
          #pragma omp parallel for reduction(max:maxError)
          for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
            for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
              double value = 0.0;
              for (size_t k_dot = 0; k_dot < A_width; ++k_dot) {
                value += A.elements[iterator(indexRow, k_dot, A_height)] * (s.elements[k_dot])
                    * V.elements[iterator(indexCol, k_dot, A_height)];
              }
              double diff = std::abs(A_copy.elements[iterator(indexRow, indexCol, A_height)] - value);
              maxError = std::max<double>(maxError, diff);
            }
          }

          file_output << "max error between A and USV: " << maxError << "\n";
          std::cout << "max error between A and USV: " << maxError << "\n";

#ifdef REPORT
            // Report Matrix A=USV
std::cout << std::fixed << std::setprecision(3) << "A=USV^T: \n";
for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
  for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
    double value = 0.0;
    for(size_t k_dot = 0; k_dot < A_width; ++k_dot){
      value += U.elements[Thesis::IteratorC(indexRow, k_dot, A_height)] * s.elements[k_dot] * V.elements[Thesis::IteratorC(indexCol, k_dot, A_height)];
    }
    std::cout << value << " ";
  }
  std::cout << '\n';
}

// Report Matrix U
file_output << std::fixed << std::setprecision(3) << "U: \n";
std::cout << std::fixed << std::setprecision(3) << "U: \n";
for (size_t indexRow = 0; indexRow < A_height; ++indexRow) {
  for (size_t indexCol = 0; indexCol < A_height; ++indexCol) {
    file_output << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
    std::cout << U.elements[Thesis::IteratorC(indexRow, indexCol, A_height)] << " ";
  }
  file_output << '\n';
  std::cout << '\n';
}

// Report \Sigma
file_output << std::fixed << std::setprecision(3) << "sigma: \n";
std::cout << std::fixed << std::setprecision(3) << "sigma: \n";
for (size_t indexCol = 0; indexCol < std::min(A_height, A_width); ++indexCol) {
  file_output << s.elements[indexCol] << " ";
  std::cout << s.elements[indexCol] << " ";
}
file_output << '\n';
std::cout << '\n';

// Report Matrix V
file_output << std::fixed << std::setprecision(3) << "V: \n";
std::cout << std::fixed << std::setprecision(3) << "V: \n";
for (size_t indexRow = 0; indexRow < A_width; ++indexRow) {
  for (size_t indexCol = 0; indexCol < A_width; ++indexCol) {
    file_output << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
    std::cout << V.elements[Thesis::IteratorC(indexRow, indexCol, A_width)] << " ";
  }
  file_output << '\n';
  std::cout << '\n';
}
#endif
          }
        }

        std::cout << "Tiempo promedio: " << (time_avg / round((double) number_repetitions)) << "\n";
      }
    }

    std::ofstream file("reporte-" + now_time + ".txt", std::ofstream::out | std::ofstream::trunc);
    file << file_output.rdbuf();
    file.close();
  }

  MPI_Finalize();


  return 0;
}
*/
