//
// Created by andre on 5/31/23.
//

#include "JacobiMethods.h"

namespace Thesis {

/***************************************************************************
    Purpose
    -------
    sequential_dgesvd computes the singular value decomposition (SVD)
    of a real M-by-N with m>>n matrix A using Jacobi one sided
    algorithm with no parallelism, optionally computing the left
    and/or right singular vectors. The SVD is written like

        A = U * SIGMA * transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
    V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note on one sided Jacobi:

        V = ((IxJ_0)xJ_1,...)
        U = A\sigma^{-1}

    Note that the routine returns VT = V**T, not V.

    Arguments
    ---------
    @param[in]
    jobu    SVD_OPTIONS
            Specifies options for computing all or part of the matrix U:
      -     = AllVec:        all M columns of U are returned in array U:
      -     = SomeVec:       the first min(m,n) columns of U (the left singular
                                  vectors) are returned in the array U;
      -     = NoVec:         no columns of U (no left singular vectors) are
                                  computed.

    @param[in]
    jobvt   SVD_OPTIONS
            Specifies options for computing all or part of the matrix V**T:
      -     = AllVec:        all N rows of V**T are returned in the array VT;
      -     = SomeVec:       the first min(m,n) rows of V**T (the right singular
                                  vectors) are returned in the array VT;
      -     = NoVec:         no rows of V**T (no right singular vectors) are
                                  computed.
    \n

    @param[in]
    m       INTEGER
            The number of rows of the input matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the input matrix A.  N >= 0.

    @param[in]
    matrix_layout_A MATRIX_LAYOUT
            The layout of the matrix A. It can only be
            ROW_MAJOR or COL_MAJOR.

    @param[in,out]
    A       DOUBLE PRECISION array, dimension (M,N)
            On entry, the M-by-N matrix A.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.

    @param[out]
    s       DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).

    @param[out]
    U       DOUBLE PRECISION array in major column order, dimension (LDU,UCOL)
            (LDU,M) if JOBU = AllVec or (LDU,min(M,N)) if JOBU = SomeVec.
      -     If JOBU = AllVec, U contains the M-by-M orthogonal matrix U;
      -     if JOBU = SomeVec, U contains the first min(m,n) columns of U
            (the left singular vectors, stored columnwise);
      -     if JOBU = NoVec, U is not referenced.

    @param[in]
    ldu     INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBU = SomeVec or AllVec, LDU >= M.

    @param[out]
    V      DOUBLE PRECISION array in major column order, dimension (LDV,N)
      -     If JOBVT = AllVec, VT contains the N-by-N orthogonal matrix V**T;
      -     if JOBVT = SomeVec, VT contains the first min(m,n) rows of V**T
            (the right singular vectors, stored rowwise);
      -     if JOBVT = NoVec, VT is not referenced.

    @param[in]
    ldv    INTEGER
            The leading dimension of the array VT.  LDVT >= 1;
      -     if JOBVT = AllVec, LDVT >= N;
      -     if JOBVT = SomeVec , LDVT >= min(M,N).

    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the required LWORK.
            if INFO > 0, WORK(2:MIN(M,N)) contains the unconverged
            superdiagonal elements of an upper bidiagonal matrix B
            whose diagonal is in S (not necessarily sorted). B
            satisfies A = U * B * VT, so it has the same singular values
            as A, and singular vectors related by U and VT.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If lwork = -1, a workspace query is assumed.  The optimal
            size for the WORK array is calculated and stored in WORK[0],
            and no other work except argument checking is performed.
    \n
            Let mx = max(M,N) and mn = min(M,N).
            The threshold for mx >> mn is currently mx >= 1.6*mn.
            For job: N=None, O=Overwrite, S=Some, A=All.
            Paths below assume M >= N; for N > M swap jobu and jobvt.
    \n
            Because of varying nb for different subroutines, formulas below are
            an upper bound. Querying gives an exact number.
            The optimal block size nb can be obtained through magma_get_dgesvd_nb(M,N).
            For many cases, there is a fast algorithm, and a slow algorithm that
            uses less workspace. Here are sizes for both cases.
    \n
            Optimal lwork (fast algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any                  3*mn + 2*mn*nb
            Path 2:   jobu=O, jobvt=N        mn*mn +     3*mn + 2*mn*nb
                                   or        mn*mn + max(3*mn + 2*mn*nb, mn + mx*mn)
            Path 3:   jobu=O, jobvt=A,S      mn*mn +     3*mn + 2*mn*nb
                                   or        mn*mn + max(3*mn + 2*mn*nb, mn + mx*mn)
            Path 4:   jobu=S, jobvt=N        mn*mn +     3*mn + 2*mn*nb
            Path 5:   jobu=S, jobvt=O      2*mn*mn +     3*mn + 2*mn*nb
            Path 6:   jobu=S, jobvt=A,S      mn*mn +     3*mn + 2*mn*nb
            Path 7:   jobu=A, jobvt=N        mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            Path 8:   jobu=A, jobvt=O      2*mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            Path 9:   jobu=A, jobvt=A,S      mn*mn + max(3*mn + 2*mn*nb, mn + mx*nb)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  3*mn + (mx + mn)*nb
    \n
            Optimal lwork (slow algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any    n/a
            Path 2:   jobu=O, jobvt=N      3*mn + (mx + mn)*nb
            Path 3-9:                      3*mn + max(2*mn*nb, mx*nb)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  n/a
    \n
            MAGMA requires the optimal sizes above, while LAPACK has the same
            optimal sizes but the minimum sizes below.
    \n
            LAPACK minimum lwork (fast algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any              5*mn
            Path 2:   jobu=O, jobvt=N        mn*mn + 5*mn
            Path 3:   jobu=O, jobvt=A,S      mn*mn + 5*mn
            Path 4:   jobu=S, jobvt=N        mn*mn + 5*mn
            Path 5:   jobu=S, jobvt=O      2*mn*mn + 5*mn
            Path 6:   jobu=S, jobvt=A,S      mn*mn + 5*mn
            Path 7:   jobu=A, jobvt=N        mn*mn + max(5*mn, mn + mx)
            Path 8:   jobu=A, jobvt=O      2*mn*mn + max(5*mn, mn + mx)
            Path 9:   jobu=A, jobvt=A,S      mn*mn + max(5*mn, mn + mx)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  max(3*mn + mx, 5*mn)
    \n
            LAPACK minimum lwork (slow algorithm)
            for mx >> mn:
            Path 1:   jobu=N, jobvt=any    n/a
            Path 2-9:                      max(3*mn + mx, 5*mn)
            for mx >= mn, but not mx >> mn:
            Path 10:  jobu=any, jobvt=any  n/a

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  if DBDSQR did not converge, INFO specifies how many
                superdiagonals of an intermediate bidiagonal form B
                did not converge to zero. See the description of WORK
                above for details.
*********************************************************************************/
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
                       size_t ldv) {

  auto iterator = get_iterator(matrix_layout_A);

  // Initializing V = 1
  if (jobv == AllVec) {
    for (size_t i = 0; i < n; ++i) {
      V.elements[iterator(i, i, ldv)] = 1.0;
    }
  } else if (jobv == SomeVec) {
    for (size_t i = 0; i < std::min(m, n); ++i) {
      V.elements[iterator(i, i, ldv)] = 1.0;
    }
  }

#ifdef DEBUG
  // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < m; ++indexRow) {
    for (size_t indexCol = 0; indexCol < n; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < m; ++k_dot){
        value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
#endif
  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t istop = 0;
  size_t stop_condition = n * (n - 1) / 2;
  size_t m_ordering = (n + 1) / 2;
  uint16_t reps = 0;
  uint16_t maxIterations = 30;

  do {
    istop = 0;
    // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
    // 1971
    for (size_t k = 1; k < m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;
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

        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
        for (size_t i = 0; i < m; ++i) {
          double tmp_p = A.elements[iterator(i, p_trans, lda)];
          double tmp_q = A.elements[iterator(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

        if (convergence_value > TOLERANCE) {
          auto [c_schur, s_schur] = non_sym_Schur_non_ordered(iterator, m, n, A, lda, p_trans, q_trans, alpha);

          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = c_schur * A.elements[iterator(i, p_trans, lda)] - s_schur * A.elements[iterator(i, q_trans, lda)];
            tmp_q = s_schur * A.elements[iterator(i, p_trans, lda)] + c_schur * A.elements[iterator(i, q_trans, lda)];
            A.elements[iterator(i, p_trans, lda)] = tmp_p;
            A.elements[iterator(i, q_trans, lda)] = tmp_q;
          }

          if (jobv == AllVec || jobv == SomeVec) {
            for (size_t i = 0; i < n; ++i) {
              tmp_p = c_schur * V.elements[iterator(i, p_trans, ldv)] - s_schur * V.elements[iterator(i, q_trans, ldv)];
              tmp_q = s_schur * V.elements[iterator(i, p_trans, ldv)] + c_schur * V.elements[iterator(i, q_trans, ldv)];
              V.elements[iterator(i, p_trans, ldv)] = tmp_p;
              V.elements[iterator(i, q_trans, ldv)] = tmp_q;
            }
          }
        } else {
          ++istop;
        }

#ifdef DEBUG
        // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
      }
    }

    for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;
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

        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
        for (size_t i = 0; i < m; ++i) {
          double tmp_p = A.elements[iterator(i, p_trans, lda)];
          double tmp_q = A.elements[iterator(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q)
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

        if (convergence_value > TOLERANCE) {
          // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q) > tolerance
          auto [c_schur, s_schur] = non_sym_Schur_non_ordered(iterator, m, n, A, lda, p, q_trans, alpha);
          double tmp_p, tmp_q;
          for (size_t i = 0; i < m; ++i) {
            tmp_p = c_schur * A.elements[iterator(i, p_trans, lda)] - s_schur * A.elements[iterator(i, q_trans, lda)];
            tmp_q = s_schur * A.elements[iterator(i, p_trans, lda)] + c_schur * A.elements[iterator(i, q_trans, lda)];
            A.elements[iterator(i, p_trans, lda)] = tmp_p;
            A.elements[iterator(i, q_trans, lda)] = tmp_q;
          }
          if (jobv == AllVec || jobv == SomeVec) {
            for (size_t i = 0; i < n; ++i) {
              tmp_p = c_schur * V.elements[iterator(i, p_trans, ldv)] - s_schur * V.elements[iterator(i, q_trans, ldv)];
              tmp_q = s_schur * V.elements[iterator(i, p_trans, ldv)] + c_schur * V.elements[iterator(i, q_trans, ldv)];
              V.elements[iterator(i, p_trans, ldv)] = tmp_p;
              V.elements[iterator(i, q_trans, ldv)] = tmp_q;
            }
          }
        } else {
          ++istop;
        }

#ifdef DEBUG
        // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
      }
    }

#ifdef DEBUG
    // Report Matrix A^T * A
    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
    for (size_t indexRow = 0; indexRow < m; ++indexRow) {
      for (size_t indexCol = 0; indexCol < n; ++indexCol) {
        double value = 0.0;
        for(size_t k_dot = 0; k_dot < m; ++k_dot){
          value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
        }
        std::cout << value << " ";
      }
      std::cout << '\n';
    }
#endif
  } while (++reps < maxIterations && istop < stop_condition);

  // Compute \Sigma
  for (size_t k = 0; k < std::min(m, n); ++k) {
    for (size_t i = 0; i < m; ++i) {
      s.elements[k] += A.elements[iterator(i, k, lda)] * A.elements[iterator(i, k, lda)];
    }
    s.elements[k] = sqrt(s.elements[k]);
  }

  //Compute U
  if (jobu == AllVec) {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < m; ++j) {
        U.elements[iterator(j, i, ldu)] = A.elements[iterator(j, i, ldu)] / s.elements[i];
      }
    }
  } else if (jobu == SomeVec) {
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        U.elements[iterator(i, k, ldu)] = A.elements[iterator(i, k, ldu)] / s.elements[k];
      }
    }
  }
}

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
                size_t ldv) {

  auto iterator = get_iterator(matrix_layout_A);

  // Initializing V = 1
  if (jobv == AllVec) {
    for (size_t i = 0; i < n; ++i) {
      V.elements[iterator(i, i, ldv)] = 1.0;
    }
  } else if (jobv == SomeVec) {
    for (size_t i = 0; i < std::min(m, n); ++i) {
      V.elements[iterator(i, i, ldv)] = 1.0;
    }
  }

  // Create scheduler
  size_t m_ordering = (n + 1) / 2;

#ifdef DEBUG
  // Report Matrix A^T * A
  std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
  for (size_t indexRow = 0; indexRow < m; ++indexRow) {
    for (size_t indexCol = 0; indexCol < n; ++indexCol) {
      double value = 0.0;
      for(size_t k_dot = 0; k_dot < m; ++k_dot){
        value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
      }
      std::cout << value << " ";
    }
    std::cout << '\n';
  }
#endif
  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t istop = 0;
  size_t stop_condition = n * (n - 1) / 2;
  uint16_t reps = 0;
  uint16_t maxIterations = 1;

  do {
    istop = 0;
    // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
    // 1971
    for (size_t k = 1; k < m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;
#pragma omp parallel for private(p, p_trans, q_trans)
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

        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
        double tmp_p, tmp_q;
        for (size_t i = 0; i < m; ++i) {
          tmp_p = A.elements[iterator(i, p_trans, lda)];
          tmp_q = A.elements[iterator(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

//        if (convergence_value > tolerance) {

        // Schur
        auto [c_schur, s_schur] = non_sym_Schur_non_ordered(iterator, m, n, A, lda, p_trans, q_trans, alpha);

        double tmp_A_p, tmp_A_q;
        for (size_t i = 0; i < m; ++i) {
          tmp_A_p = A.elements[iterator(i, p_trans, lda)];
          tmp_A_q = A.elements[iterator(i, q_trans, lda)];
          tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
          tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
          A.elements[iterator(i, p_trans, lda)] = tmp_p;
          A.elements[iterator(i, q_trans, lda)] = tmp_q;
        }

        if (jobv == AllVec || jobv == SomeVec) {
          for (size_t i = 0; i < n; ++i) {
            tmp_p = c_schur * V.elements[iterator(i, p_trans, ldv)] - s_schur * V.elements[iterator(i, q_trans, ldv)];
            tmp_q = s_schur * V.elements[iterator(i, p_trans, ldv)] + c_schur * V.elements[iterator(i, q_trans, ldv)];
            V.elements[iterator(i, p_trans, ldv)] = tmp_p;
            V.elements[iterator(i, q_trans, ldv)] = tmp_q;
          }
        }
//        } else {
//          ++istop;
//        }

#ifdef DEBUG
        // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
      }
    }

    for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;
#pragma omp parallel for private(p, p_trans, q_trans)
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

        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
        double tmp_p, tmp_q;
        for (size_t i = 0; i < m; ++i) {
          tmp_p = A.elements[iterator(i, p_trans, lda)];
          tmp_q = A.elements[iterator(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q)
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

//        if (convergence_value > tolerance) {
        // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q) > tolerance
        // Schur
        auto [c_schur, s_schur] = non_sym_Schur_non_ordered(iterator, m, n, A, lda, p_trans, q_trans, alpha);

        double A_tmp_p, A_tmp_q;
        for (size_t i = 0; i < m; ++i) {
          A_tmp_p = A.elements[iterator(i, p_trans, lda)];
          A_tmp_q = A.elements[iterator(i, q_trans, lda)];
          tmp_p = c_schur * A_tmp_p - s_schur * A_tmp_q;
          tmp_q = s_schur * A_tmp_p + c_schur * A_tmp_q;
          A.elements[iterator(i, p_trans, lda)] = tmp_p;
          A.elements[iterator(i, q_trans, lda)] = tmp_q;
        }
        if (jobv == AllVec || jobv == SomeVec) {
          for (size_t i = 0; i < n; ++i) {
            tmp_p = c_schur * V.elements[iterator(i, p_trans, ldv)] - s_schur * V.elements[iterator(i, q_trans, ldv)];
            tmp_q = s_schur * V.elements[iterator(i, p_trans, ldv)] + c_schur * V.elements[iterator(i, q_trans, ldv)];
            V.elements[iterator(i, p_trans, ldv)] = tmp_p;
            V.elements[iterator(i, q_trans, ldv)] = tmp_q;
          }
        }
//        } else {
//          ++istop;
//        }

#ifdef DEBUG
        // Report Matrix A^T * A
        std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
        for (size_t indexRow = 0; indexRow < m; ++indexRow) {
          for (size_t indexCol = 0; indexCol < n; ++indexCol) {
            double value = 0.0;
            for(size_t k_dot = 0; k_dot < m; ++k_dot){
              value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
            }
            std::cout << value << " ";
          }
          std::cout << '\n';
        }
#endif
      }
    }

#ifdef DEBUG
    // Report Matrix A^T * A
    std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
    for (size_t indexRow = 0; indexRow < m; ++indexRow) {
      for (size_t indexCol = 0; indexCol < n; ++indexCol) {
        double value = 0.0;
        for(size_t k_dot = 0; k_dot < m; ++k_dot){
          value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
        }
        std::cout << value << " ";
      }
      std::cout << '\n';
    }
#endif
  } while (++reps < maxIterations);

  std::cout << "How many repetitions?: " << reps << "\n";

  // Compute \Sigma
#pragma omp parallel for
  for (size_t k = 0; k < std::min(m, n); ++k) {
    for (size_t i = 0; i < m; ++i) {
      s.elements[k] += A.elements[iterator(i, k, lda)] * A.elements[iterator(i, k, lda)];
    }
    s.elements[k] = sqrt(s.elements[k]);
  }

  //Compute U
  if (jobu == AllVec) {
#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < m; ++j) {
        U.elements[iterator(j, i, ldu)] = A.elements[iterator(j, i, ldu)] / s.elements[i];
      }
    }
  } else if (jobu == SomeVec) {
#pragma omp parallel for
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        U.elements[iterator(i, k, ldu)] = A.elements[iterator(i, k, ldu)] / s.elements[k];
      }
    }
  }

//  delete []ordering_array;
}

void omp_mpi_dgesvd(SVD_OPTIONS jobu,
                    SVD_OPTIONS jobv,
                    size_t m,
                    size_t n,
                    MatrixMPI &A,
                    size_t lda,
                    MatrixMPI &s,
                    MatrixMPI &V,
                    size_t ldv) {
  // Get iterator
  auto iterator = get_iterator(COL_MAJOR);

  // Get rank of mpi proccess and size of process
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == ROOT_RANK) {
    // Initializing V = 1
    if (jobv == AllVec) {
      #pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        V.elements[iterator(i, i, ldv)] = 1.0;
      }
    } else if (jobv == SomeVec) {
      #pragma omp parallel for
      for (size_t i = 0; i < std::min(m, n); ++i) {
        V.elements[iterator(i, i, ldv)] = 1.0;
      }
    }
  }

  // Create scheduler
  size_t m_ordering = (n + 1) / 2;
  // Stopping condition in Hogben, L. (Ed.). (2013). Handbook of Linear Algebra (2nd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16113
  size_t istop = 0;
  size_t stop_condition = n * (n - 1) / 2;
  uint16_t reps = 0;
  uint16_t maxIterations = 1;

  do {
    istop = 0;
    // Ordering in  A. Sameh. On Jacobi and Jacobi-like algorithms for a parallel computer. Math. Comput., 25:579–590,
    // 1971
    for (size_t k = 1; k < m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;

      std::vector<std::tuple<int, int>> k_ordering;

      if(rank == ROOT_RANK){
        k_ordering = std::vector<std::tuple<int, int>>();
      }

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

        // add to distribution set
        if(rank == ROOT_RANK)
          k_ordering.emplace_back(p_trans, q_trans);

        /*
        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
        double tmp_p, tmp_q;
        for (size_t i = 0; i < m; ++i) {
          tmp_p = A.elements[iterator(i, p_trans, lda)];
          tmp_q = A.elements[iterator(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // abs(a_p^T\cdot a_q) / sqrt((a_p^T\cdot a_p)(a_q^T\cdot a_q))
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

//        if (convergence_value > tolerance) {

        // Schur
        auto [c_schur, s_schur] = non_sym_Schur_non_ordered(iterator, m, n, A, lda, p_trans, q_trans, alpha);

        double tmp_A_p, tmp_A_q;
        for (size_t i = 0; i < m; ++i) {
          tmp_A_p = A.elements[iterator(i, p_trans, lda)];
          tmp_A_q = A.elements[iterator(i, q_trans, lda)];
          tmp_p = c_schur * tmp_A_p - s_schur * tmp_A_q;
          tmp_q = s_schur * tmp_A_p + c_schur * tmp_A_q;
          A.elements[iterator(i, p_trans, lda)] = tmp_p;
          A.elements[iterator(i, q_trans, lda)] = tmp_q;
        }

        if (jobv == AllVec || jobv == SomeVec) {
          for (size_t i = 0; i < n; ++i) {
            tmp_p = c_schur * V.elements[iterator(i, p_trans, ldv)] - s_schur * V.elements[iterator(i, q_trans, ldv)];
            tmp_q = s_schur * V.elements[iterator(i, p_trans, ldv)] + c_schur * V.elements[iterator(i, q_trans, ldv)];
            V.elements[iterator(i, p_trans, ldv)] = tmp_p;
            V.elements[iterator(i, q_trans, ldv)] = tmp_q;
          }
        }
        */
      }

      int k_ordering_len = round((double) n) / 2.0;
      MatrixMPI A_local(k_ordering_len, k_ordering_len);


    }

    for (size_t k = m_ordering; k < 2 * m_ordering; ++k) {
      size_t p = 0;
      size_t p_trans = 0;
      size_t q_trans = 0;
#pragma omp parallel for private(p, p_trans, q_trans)
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

        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        // \alpha = a_p^T\cdot a_q, \beta = a_p^T\cdot a_p, \gamma = a_q^T\cdot a_q
        double tmp_p, tmp_q;
        for (size_t i = 0; i < m; ++i) {
          tmp_p = A.elements[iterator(i, p_trans, lda)];
          tmp_q = A.elements[iterator(i, q_trans, lda)];
          alpha += tmp_p * tmp_q;
          beta += tmp_p * tmp_p;
          gamma += tmp_q * tmp_q;
        }

        // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q)
        double convergence_value = abs(alpha) / sqrt(beta * gamma);

//        if (convergence_value > tolerance) {
        // (a_p^T\cdot a_q)^2 / (a_p^T\cdot a_p)(a_q^T\cdot a_q) > tolerance
        // Schur
        auto [c_schur, s_schur] = non_sym_Schur_non_ordered(iterator, m, n, A, lda, p_trans, q_trans, alpha);

        double A_tmp_p, A_tmp_q;
        for (size_t i = 0; i < m; ++i) {
          A_tmp_p = A.elements[iterator(i, p_trans, lda)];
          A_tmp_q = A.elements[iterator(i, q_trans, lda)];
          tmp_p = c_schur * A_tmp_p - s_schur * A_tmp_q;
          tmp_q = s_schur * A_tmp_p + c_schur * A_tmp_q;
          A.elements[iterator(i, p_trans, lda)] = tmp_p;
          A.elements[iterator(i, q_trans, lda)] = tmp_q;
        }
        if (jobv == AllVec || jobv == SomeVec) {
          for (size_t i = 0; i < n; ++i) {
            tmp_p = c_schur * V.elements[iterator(i, p_trans, ldv)] - s_schur * V.elements[iterator(i, q_trans, ldv)];
            tmp_q = s_schur * V.elements[iterator(i, p_trans, ldv)] + c_schur * V.elements[iterator(i, q_trans, ldv)];
            V.elements[iterator(i, p_trans, ldv)] = tmp_p;
            V.elements[iterator(i, q_trans, ldv)] = tmp_q;
          }
        }
//        } else {
//          ++istop;
//        }

#ifdef DEBUG
        // Report Matrix A^T * A
      std::cout << std::fixed << std::setprecision(3) << "A^T * A: \n";
      for (size_t indexRow = 0; indexRow < m; ++indexRow) {
        for (size_t indexCol = 0; indexCol < n; ++indexCol) {
          double value = 0.0;
          for(size_t k_dot = 0; k_dot < m; ++k_dot){
            value += A.elements[iterator(k_dot, indexRow, lda)] * A.elements[iterator(k_dot, indexCol, lda)];
          }
          std::cout << value << " ";
        }
        std::cout << '\n';
      }
#endif
      }
    }

  } while (++reps < maxIterations);

  if(rank == ROOT_RANK){
    // Compute \Sigma
    #pragma omp parallel for
    for (size_t k = 0; k < std::min(m, n); ++k) {
      for (size_t i = 0; i < m; ++i) {
        s.elements[k] += A.elements[iterator(i, k, lda)] * A.elements[iterator(i, k, lda)];
      }
      s.elements[k] = sqrt(s.elements[k]);
    }

    //Compute U
    if (jobu == AllVec) {
      // We expect here squared matrix
      #pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          A.elements[iterator(j, i, lda)] /= s.elements[i];
        }
      }
    } else if (jobu == SomeVec) {
      // We expect here squared matrix
      #pragma omp parallel for
      for (size_t k = 0; k < std::min(m, n); ++k) {
        for (size_t i = 0; i < m; ++i) {
          A.elements[iterator(j, i, lda)] /= s.elements[i];
        }
      }
    }
  }

}
}