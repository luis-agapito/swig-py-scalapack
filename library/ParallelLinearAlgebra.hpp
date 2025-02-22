#include "mkl_blacs.h"
#include "mkl_pblas.h"
#include "mkl_scalapack.h"

#include "Eigen/Dense"
#include "mpi.h"
#include <array>
#include <cassert>
#include <iostream>
#include <vector>

#ifndef CPP_PARALLELLINEARALGEBRA_HPP
#define CPP_PARALLELLINEARALGEBRA_HPP

// Based on class Scalapack from https://github.com/yohm/scalapack_cpp
// Contains wrapper to scalapack eigensolver routines and clases for describing
// global and local version of distributed matrices.
class ParallelLinearAlgebra {
public:
  static int ICTXT, NPROW, NPCOL, MYROW, MYCOL;

  static void Initialize(const std::array<int, 2> &proc_grid_size) {
    NPROW = proc_grid_size[0];
    NPCOL = proc_grid_size[1];
    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    if (num_proc != NPROW * NPCOL) {
      std::cerr << "Error: invalid number of procs" << std::endl;
      std::cerr << "num_proc, NPROW, NPCOL: " << num_proc << ", " << NPROW
                << ", " << NPCOL << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Luis
    int inegone = -1, ione = 1, izero = 0;
    // sl_init_(&ICTXT, &NPROW, &NPCOL);
    int int1, int2;
    blacs_pinfo(&int1, &int2);
    // std::cout << "blacs_pinfo: my id is: " << int1 << "  . #of procs: " <<
    // int2 << std::endl;
    blacs_get(&inegone, &izero, &ICTXT);
    blacs_gridinit(&ICTXT, "C", &NPROW, &NPCOL);

    blacs_gridinfo(&ICTXT, &NPROW, &NPCOL, &MYROW, &MYCOL);
    // std::cout << "blacs_gridinfo: ICTXT, NPROW, NPCOL, MYROW, MYCOL: " <<
    // ICTXT << ", " << NPROW << ", "
    //           << NPCOL << ", " << MYROW << ", " << MYCOL << std::endl;
  }

  static void Finalize() {
    blacs_gridexit(&ICTXT);
    int blacs_exitcode = 0;
    blacs_exit(&blacs_exitcode);
  }

  /// Global matrix. The data is stored row-wise (C style)
  /// Notation: M,N (dimensions) and i,j (indices) for rows,columns
  class GMatrix {
  public:
    GMatrix(size_t M, size_t N) : M(M), N(N) { A.resize(M * N, 0.0); }

    size_t M, N;
    std::vector<double> A;

    double At(size_t I, size_t J) const { return A.at(I * N + J); }

    void Set(size_t I, size_t J, double val) { A[I * N + J] = val; }

    double *Data() { return A.data(); }

    size_t Size() { return A.size(); }

    friend std::ostream &operator<<(std::ostream &os, const GMatrix &gm) {
      for (size_t i = 0; i < gm.M; i++) {
        for (size_t j = 0; j < gm.N; j++) {
          os << gm.At(i, j) << ' ';
        }
        os << "\n";
      }
      return os;
    }

    void BcastFrom(int root_rank) {
      int my_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
      std::array<uint64_t, 2> sizes = {M, N};
      MPI_Bcast(sizes.data(), 2, MPI_UINT64_T, root_rank, MPI_COMM_WORLD);
      if (my_rank != root_rank) {
        A.resize(M * N);
      }
      MPI_Bcast(A.data(), M * N, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
    }

    void printMemoryData() {
      int my_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
      if (my_rank == 0) {
        std::cout << "vector : ";
        for (auto item : A) {
          std::cout << item << ", ";
        }
        std::cout << std::endl;
      }
    }
  };

  /// Local matrix for scalapack. The data is stored column-wise (Fortran style)
  /// Notation: M,N (dimensions) and i,j (indices) for rows,columns
  class LMatrix {
  public:
    LMatrix(int M, int N, int MB, int NB)
        : M(M), N(N), MB(MB), NB(NB) { // matrix M x N with block NB x MB
      SUB_ROWS = (M / (MB * NPROW)) * MB + std::min(M % (MB * NPROW), MB);
      SUB_COLS = (N / (NB * NPCOL)) * NB + std::min(N % (NB * NPCOL), NB);
      int RSRC = 0, CSRC = 0, INFO;
      descinit(DESC, &M, &N, &MB, &NB, &RSRC, &CSRC, &ICTXT, &SUB_ROWS, &INFO);
      assert(INFO == 0);
      SUB.resize(SUB_ROWS * SUB_COLS, 0.0);
    }

    LMatrix(const GMatrix &gm, int MB, int NB)
        : M(gm.M), N(gm.N), MB(MB), NB(NB) {
      SUB_ROWS = (M / (MB * NPROW)) * MB + std::min(M % (MB * NPROW), MB);
      SUB_COLS = (N / (NB * NPCOL)) * NB + std::min(N % (NB * NPCOL), NB);
      int RSRC = 0, CSRC = 0, INFO;
      descinit(DESC, &M, &N, &MB, &NB, &RSRC, &CSRC, &ICTXT, &SUB_ROWS, &INFO);
      assert(INFO == 0);
      SUB.resize(SUB_ROWS * SUB_COLS, 0.0);
      for (int i = 0; i < SUB_ROWS; i++) {
        for (int j = 0; j < SUB_COLS; j++) {
          auto IJ = ToGlobalCoordinate(i, j);
          size_t I = IJ[0], J = IJ[1];
          if (I < M && J < N) {
            Set(i, j, gm.At(I, J));
          }
        }
      }
    };
    int M, N;               // size of the global matrix
    int MB, NB;             // block sizes
    int SUB_ROWS, SUB_COLS; // size of the local matrix
    int DESC[9];
    std::vector<double> SUB;

    // convert submatrix index (i,j) at process (p_row, p_col) into global
    // coordinate (I,J)
    std::array<size_t, 2> ToGlobalCoordinate(size_t i, size_t j,
                                             int p_row = MYROW,
                                             int p_col = MYCOL) const {
      // block coordinate (bi, bj)
      size_t bi = i / MB;
      size_t bj = j / NB;
      // local coordinate inside the block
      size_t ii = i % MB;
      size_t jj = j % NB;
      // calculate global coordinate
      size_t I = bi * (MB * NPROW) + p_row * MB + ii;
      size_t J = bj * (NB * NPCOL) + p_col * NB + jj;
      return {I, J};
    }

    // convert global matrix index (I,J) to local coordinate (i,j),(p_row,p_col)
    std::pair<std::array<size_t, 2>, std::array<int, 2>>
    ToLocalCoordinate(size_t I, size_t J) const {
      // global block coordinate (BI, BJ)
      size_t BI = I / MB;
      size_t BJ = J / NB;
      // process coordinate (bi, bj)
      int p_row = BI % NPROW;
      int p_col = BJ % NPCOL;
      // local block coordinate (bi, bj)
      size_t bi = BI / NPROW;
      size_t bj = BJ / NPCOL;
      // local coordinate inside the block
      size_t ii = I % MB;
      size_t jj = J % NB;
      // calculate global coordinate
      size_t i = bi * MB + ii;
      size_t j = bj * NB + jj;
      return {{i, j}, {p_row, p_col}};
    }

    double At(size_t i, size_t j) const { // get an element at SUB[ (i,j) ]
      return SUB[i + j * SUB_ROWS];
    }

    void Set(size_t i, size_t j, double val) { SUB[i + j * SUB_ROWS] = val; }

    void SetByGlobalCoordinate(size_t I, size_t J, double val) {
      auto local_pos = ToLocalCoordinate(I, J);
      auto ij = local_pos.first;
      auto proc_grid = local_pos.second;
      if (proc_grid[0] == MYROW && proc_grid[1] == MYCOL) {
        Set(ij[0], ij[1], val);
      }
    }

    void SetAll(double val) {
      for (size_t i = 0; i < SUB_ROWS; i++) {
        for (size_t j = 0; j < SUB_COLS; j++) {
          auto IJ = ToGlobalCoordinate(i, j);
          if (IJ[0] < M && IJ[1] < N)
            Set(i, j, val);
        }
      }
    }

    double *Data() { return SUB.data(); }

    friend std::ostream &operator<<(std::ostream &os, const LMatrix &lm) {
      for (size_t i = 0; i < lm.SUB_ROWS; i++) {
        for (size_t j = 0; j < lm.SUB_COLS; j++) {
          os << lm.At(i, j) << ' ';
        }
        os << "\n";
      }
      return os;
    }

    GMatrix ConstructGlobalMatrix() const {
      GMatrix A(M, N);
      for (size_t i = 0; i < SUB_ROWS; i++) {
        for (size_t j = 0; j < SUB_COLS; j++) {
          auto IJ = ToGlobalCoordinate(i, j);
          size_t I = IJ[0], J = IJ[1];
          if (I < M && J < N) {
            A.Set(I, J, At(i, j));
          }
        }
      }
      GMatrix AA(M, N);
      MPI_Allreduce(A.Data(), AA.Data(), M * N, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      return AA;
    }

    void DebugPrintAtRoot(std::ostream &out) const {
      MPI_Barrier(MPI_COMM_WORLD);
      GMatrix g = ConstructGlobalMatrix();
      if (ParallelLinearAlgebra::MYROW == 0 &&
          ParallelLinearAlgebra::MYCOL == 0) {
        out << g;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    static LMatrix Identity(int N, int M, int NB, int MB) {
      LMatrix lm(N, M, NB, MB);
      for (size_t i = 0; i < lm.SUB_ROWS; i++) {
        for (size_t j = 0; j < lm.SUB_COLS; j++) {
          auto IJ = lm.ToGlobalCoordinate(i, j);
          if (IJ[0] < N && IJ[1] < M && IJ[0] == IJ[1])
            lm.Set(i, j, 1.0);
        }
      }
      return lm;
    }

    void printMemoryData() {
      int my_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
      if (my_rank == 0) {
        std::cout << "memoryData: ";
        for (auto item : SUB) {
          std::cout << item << ", ";
        }
        std::cout << std::endl;
      }
    }
  };

  // computes all eigen vectors/values of a real symmetric matrix
  // the results are stored in C
  static void callPDSYEV(LMatrix &la, GMatrix &w, LMatrix &lz) {
    assert(la.M == lz.M);
    const size_t M = la.M, N = la.N;
    int IA = 1, JA = 1, IZ = 1, JZ = 1;

    const char jobz = 'V';
    const char uplo = 'U';
    double work_query;
    MKL_INT INFO;

    // Query (lwork = -1) and allocate WORK
    MKL_INT lwork = -1;
    pdsyev(&jobz, &uplo, &la.M, la.Data(), &IA, &JA, la.DESC, w.Data(),
           lz.Data(), &IZ, &JZ, lz.DESC, &work_query, &lwork, &INFO);

    lwork = static_cast<MKL_INT>(work_query);
    // work= (double*)malloc(lwork*sizeof(double));
    double *work = new double[lwork];

    // Do the eigendecomposition
    pdsyev(&jobz, &uplo, &(la.M), la.Data(), &IA, &JA, la.DESC, w.Data(),
           lz.Data(), &IZ, &JZ, lz.DESC, work, &lwork, &INFO);

    // Print eigenvects
    ParallelLinearAlgebra::GMatrix z = lz.ConstructGlobalMatrix();

    // Deallocate memory
    delete[] work;

    if (INFO != 0) {
      std::cerr << "Error: INFO of callPDSYEV is not zero but " << INFO
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    assert(INFO == 0);
  }
};

int ParallelLinearAlgebra::ICTXT = -1;
int ParallelLinearAlgebra::NPROW = -1, ParallelLinearAlgebra::NPCOL = -1;
int ParallelLinearAlgebra::MYROW = -1, ParallelLinearAlgebra::MYCOL = -1;

#endif // CPP_PARALLELLINEARALGEBRA_HPP
