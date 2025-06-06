#ifndef CPP_PARALLELLINEARALGEBRA_HPP
#define CPP_PARALLELLINEARALGEBRA_HPP

#define MKL_Complex16 std::complex<double>
#define MKL_Complex8 std::complex<float>

// clang-format off
#include <iostream>
#include <complex> //std::abs() for complex data
// clang-format on
#include "mkl_blacs.h"
#include "mkl_pblas.h"
#include "mkl_scalapack.h"

#include "Eigen/Dense"
#include "mpi.h"
#include <array>
#include <cassert>
#include <cmath> //std::abs() for real data
#include <iostream>
#include <vector>

using dcomplex = std::complex<double>;
using fcomplex = std::complex<float>;

#define assert_macro(x)                                                        \
  do {                                                                         \
    if (!(x)) {                                                                \
      printf("Assertion %s in %s (line %d) failed.\n", #x, __FILE__,           \
             __LINE__);                                                        \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0)

// Helper to check whether type is std::complex
template <typename T> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

template <typename T> MPI_Datatype resolveMPIType();

// class DistributedMatrix
template <typename T> class DistributedMatrix {
  // Adapted from https://github.com/yohm/scalapack_cpp
  // Clases for describing global and local version of distributed matrices.
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
    //
    int inegone = -1, ione = 1, izero = 0;
    // sl_init_(&ICTXT, &NPROW, &NPCOL);
    // int int1, int2;
    // blacs_pinfo(&int1, &int2);
    blacs_get(&inegone, &izero, &ICTXT);
    blacs_gridinit(&ICTXT, "C", &NPROW, &NPCOL);
    blacs_gridinfo(&ICTXT, &NPROW, &NPCOL, &MYROW, &MYCOL);
  }

  static void Finalize() {
    blacs_gridexit(&ICTXT); // Release the context ICTXT
    // int continuation = 1; // 0 stops MPI, 1 keeps MPI alive.
    // blacs_exit(&continuation);
  }

  /// Global matrix. The data is stored row-wise (C style)
  /// Notation: M,N (dimensions) and i,j (indices) for rows,columns
  class GlobalMatrix {
  public:
    GlobalMatrix(size_t M, size_t N) : M(M), N(N) { A.resize(M * N, 0.0); }

    size_t M, N;
    std::vector<T> A;

    T At(size_t I, size_t J) const { return A.at(J * M + I); }

    void Set(size_t I, size_t J, T val) { A[J * M + I] = val; }

    T *data() { return A.data(); }

    size_t Size() { return A.size(); }
  };

  /// Local matrix for scalapack. The data is stored column-wise (Fortran
  /// style) Notation: M,N (dimensions) and i,j (indices) for rows,columns
  class LocalMatrix {
  public:
    LocalMatrix(int M, int N, int MB, int NB)
        : M(M), N(N), MB(MB), NB(NB) { // matrix M x N with block NB x MB
      SUB_ROWS = (M / (MB * NPROW)) * MB + std::min(M % (MB * NPROW), MB);
      SUB_COLS = (N / (NB * NPCOL)) * NB + std::min(N % (NB * NPCOL), NB);
      int RSRC = 0, CSRC = 0, INFO;
      descinit(DESC, &M, &N, &MB, &NB, &RSRC, &CSRC, &ICTXT, &SUB_ROWS, &INFO);
      assert(INFO == 0);
      SUB.resize(SUB_ROWS * SUB_COLS, 0.0);
    }

    // Constructor: initialize local matrix SUB with data from input GMatrix gm.
    LocalMatrix(const GlobalMatrix &gm, int MB, int NB)
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
            Set(i, j, gm.At(I, J)); // SUB[i,j] <-- gm[I,J]
          }
        }
      }
    };

    // Constructor: initialized local matrix SUB with data from input global
    // buffer 'global_buffer' of size MxN
    LocalMatrix(T *global_buffer, int M, int N, int MB, int NB)
        : M(M), N(N), MB(MB), NB(NB) { // matrix M x N with block NB x MB
      SUB_ROWS = (M / (MB * NPROW)) * MB + std::min(M % (MB * NPROW), MB);
      SUB_COLS = (N / (NB * NPCOL)) * NB + std::min(N % (NB * NPCOL), NB);
      int RSRC = 0, CSRC = 0, INFO;
      descinit(DESC, &M, &N, &MB, &NB, &RSRC, &CSRC, &ICTXT, &SUB_ROWS, &INFO);
      assert(INFO == 0);
      // SUB.reserve(SUB_ROWS * SUB_COLS);
      // SUB.insert(SUB.begin(), global_buffer, global_buffer + M * N); // does
      // not copy memory
      SUB.resize(SUB_ROWS * SUB_COLS, 0.0);
      for (int i = 0; i < SUB_ROWS; i++) {
        for (int j = 0; j < SUB_COLS; j++) {
          auto IJ = ToGlobalCoordinate(i, j);
          size_t I = IJ[0], J = IJ[1];
          if (I < M && J < N) {
            Set(i, j, global_buffer[J * M + I]); // SUB[i,j] <-- gm[I,J]
          }
        }
      }
    }

    int M, N;               // size of the global matrix
    int MB, NB;             // block sizes
    int SUB_ROWS, SUB_COLS; // size of the local matrix
    int DESC[9];
    std::vector<T> SUB; // data of local matrix

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

    // convert global matrix index (I,J) to local coordinate
    // (i,j),(p_row,p_col)
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

    T At(size_t i, size_t j) const { // get an element at SUB[ (i,j) ]
      return SUB[j * SUB_ROWS + i];
    }

    void Set(size_t i, size_t j, T val) { SUB[j * SUB_ROWS + i] = val; }

    void SetByGlobalCoordinate(size_t I, size_t J, T val) {
      auto local_pos = ToLocalCoordinate(I, J);
      auto ij = local_pos.first;
      auto proc_grid = local_pos.second;
      if (proc_grid[0] == MYROW && proc_grid[1] == MYCOL) {
        Set(ij[0], ij[1], val);
      }
    }

    T *data() { return SUB.data(); }

    GlobalMatrix constructGlobalMatrix() const {
      GlobalMatrix A(M, N);
      for (size_t j = 0; j < SUB_COLS; j++) {
        for (size_t i = 0; i < SUB_ROWS; i++) {
          auto IJ = ToGlobalCoordinate(i, j);
          size_t I = IJ[0], J = IJ[1];
          if (I < M && J < N) {
            A.Set(I, J, At(i, j));
          }
        }
      }
      GlobalMatrix AA(M, N);
      MPI_Datatype data_type = resolveMPIType<T>();
      MPI_Allreduce(A.data(), AA.data(), M * N, data_type, MPI_SUM,
                    MPI_COMM_WORLD);
      return AA;
    }

    // ConstructGlobalMatrix in place
    void constructGlobalMatrix(T *global_buffer) const {
      GlobalMatrix A(M, N);
      for (size_t n = 0; n < M * N; n++) {
        global_buffer[n] = 0;
      }
      for (size_t j = 0; j < SUB_COLS; j++) {
        for (size_t i = 0; i < SUB_ROWS; i++) {
          auto IJ = ToGlobalCoordinate(i, j);
          size_t I = IJ[0], J = IJ[1];
          if (I < M && J < N) {
            global_buffer[J * M + I] =
                At(i, j); // global_buffer[I,J] <-- local SUB[i,j]
          }
        }
      }
      GlobalMatrix AA(M, N);
      MPI_Datatype data_type = resolveMPIType<T>();
      MPI_Allreduce(MPI_IN_PLACE, global_buffer, M * N, data_type, MPI_SUM,
                    MPI_COMM_WORLD);
    }
  };
};

// Class to Scalapack
template <typename T> class Scalapack {
  // Contains wrappers to ScaLAPACK routines
private:
  // ------------------------------------------------------
  // p_sd_syev/p_cz_heev  Eigensolver for Symmetric/Hermitian matrix
  template <typename T1>
  static typename std::enable_if<
      !(is_complex<T1>::value) && std::is_same<T1, float>::value, void>::type
  p_sd_syev(char *jobz, char *uplo, MKL_INT *n, T1 *a, MKL_INT *ia, MKL_INT *ja,
            MKL_INT *desca, T1 *w, T1 *z, MKL_INT *iz, MKL_INT *jz,
            MKL_INT *descz, T1 *work, MKL_INT *lwork) {
    MKL_INT info;
    pssyev(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work, lwork,
           &info);
    assert_macro(info == 0);
  }
  template <typename T1>
  static typename std::enable_if<
      !(is_complex<T1>::value) && std::is_same<T1, double>::value, void>::type
  p_sd_syev(char *jobz, char *uplo, MKL_INT *n, T1 *a, MKL_INT *ia, MKL_INT *ja,
            MKL_INT *desca, T1 *w, T1 *z, MKL_INT *iz, MKL_INT *jz,
            MKL_INT *descz, T1 *work, MKL_INT *lwork) {
    MKL_INT info;
    pdsyev(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work, lwork,
           &info);
    assert_macro(info == 0);
  }
  template <typename T1>
  static typename std::enable_if<
      (is_complex<T1>::value) &&
          std::is_same<typename T1::value_type, float>::value,
      void>::type
  p_cz_heev(char *jobz, char *uplo, MKL_INT *n, T1 *a, MKL_INT *ia, MKL_INT *ja,
            MKL_INT *desca, typename T1::value_type *w, T1 *z, MKL_INT *iz,
            MKL_INT *jz, MKL_INT *descz, T1 *work, MKL_INT *lwork,
            typename T1::value_type *rwork, MKL_INT *lrwork) {
    MKL_INT info;
    pcheev(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work, lwork,
           rwork, lrwork, &info);
    assert_macro(info == 0);
  }
  template <typename T1>
  static typename std::enable_if<
      (is_complex<T1>::value) &&
          std::is_same<typename T1::value_type, double>::value,
      void>::type
  p_cz_heev(char *jobz, char *uplo, MKL_INT *n, T1 *a, MKL_INT *ia, MKL_INT *ja,
            MKL_INT *desca, typename T1::value_type *w, T1 *z, MKL_INT *iz,
            MKL_INT *jz, MKL_INT *descz, T1 *work, MKL_INT *lwork,
            typename T1::value_type *rwork, MKL_INT *lrwork) {
    MKL_INT info;
    pzheev(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work, lwork,
           rwork, lrwork, &info);
    assert_macro(info == 0);
  }

  // p_sdcz_getrf ------------------------------------------------------
  template <typename T1>
  static typename std::enable_if<
      !(is_complex<T1>::value) && std::is_same<T1, float>::value, void>::type
  p_sdcz_getrf(MKL_INT *m, MKL_INT *n, T1 *local_a, MKL_INT *ia, MKL_INT *ja,
               MKL_INT *desca, MKL_INT *local_ipiv) {
    MKL_INT info = 1;
    psgetrf(m, n, local_a, ia, ja, desca, local_ipiv, &info);
    assert_macro(info == 0);
  }
  template <typename T1>
  static typename std::enable_if<
      !(is_complex<T1>::value) && std::is_same<T1, double>::value, void>::type
  p_sdcz_getrf(MKL_INT *m, MKL_INT *n, T1 *local_a, MKL_INT *ia, MKL_INT *ja,
               MKL_INT *desca, MKL_INT *local_ipiv) {
    MKL_INT info = 1;
    pdgetrf(m, n, local_a, ia, ja, desca, local_ipiv, &info);
    assert_macro(info == 0);
  }
  template <typename T1>
  static typename std::enable_if<
      (is_complex<T1>::value) &&
          std::is_same<typename T1::value_type, float>::value,
      void>::type
  p_sdcz_getrf(MKL_INT *m, MKL_INT *n, T1 *local_a, MKL_INT *ia, MKL_INT *ja,
               MKL_INT *desca, MKL_INT *local_ipiv) {
    MKL_INT info = 1;
    pcgetrf(m, n, local_a, ia, ja, desca, local_ipiv, &info);
    assert_macro(info == 0);
  }
  template <typename T1>
  static typename std::enable_if<
      (is_complex<T1>::value) &&
          std::is_same<typename T1::value_type, double>::value,
      void>::type
  p_sdcz_getrf(MKL_INT *m, MKL_INT *n, T1 *local_a, MKL_INT *ia, MKL_INT *ja,
               MKL_INT *desca, MKL_INT *local_ipiv) {
    MKL_INT info = 1;
    pzgetrf(m, n, local_a, ia, ja, desca, local_ipiv, &info);
    assert_macro(info == 0);
  }

  // p_sdcz_getri ------------------------------------------------------
  template <typename T1>
  static typename std::enable_if<
      !(is_complex<T1>::value) && std::is_same<T1, float>::value, void>::type
  p_sdcz_getri(MKL_INT *n, T1 *local_a, MKL_INT *ia, MKL_INT *ja,
               MKL_INT *desca, MKL_INT *local_ipiv, T1 *local_work,
               MKL_INT *local_lwork, MKL_INT *local_iwork, MKL_INT *liwork) {
    MKL_INT info = 1;
    psgetri(n, local_a, ia, ja, desca, local_ipiv, local_work, local_lwork,
            local_iwork, liwork, &info);
    assert_macro(info == 0);
  }
  template <typename T1>
  static typename std::enable_if<
      !(is_complex<T1>::value) && std::is_same<T1, double>::value, void>::type
  p_sdcz_getri(MKL_INT *n, T1 *local_a, MKL_INT *ia, MKL_INT *ja,
               MKL_INT *desca, MKL_INT *local_ipiv, T1 *local_work,
               MKL_INT *local_lwork, MKL_INT *local_iwork, MKL_INT *liwork) {
    MKL_INT info = 1;
    pdgetri(n, local_a, ia, ja, desca, local_ipiv, local_work, local_lwork,
            local_iwork, liwork, &info);
    assert_macro(info == 0);
  }
  template <typename T1>
  static typename std::enable_if<
      (is_complex<T1>::value) &&
          std::is_same<typename T1::value_type, float>::value,
      void>::type
  p_sdcz_getri(MKL_INT *n, T1 *local_a, MKL_INT *ia, MKL_INT *ja,
               MKL_INT *desca, MKL_INT *local_ipiv, T1 *local_work,
               MKL_INT *local_lwork, MKL_INT *local_iwork, MKL_INT *liwork) {
    MKL_INT info = 1;
    pcgetri(n, local_a, ia, ja, desca, local_ipiv, local_work, local_lwork,
            local_iwork, liwork, &info);
    assert_macro(info == 0);
  }
  template <typename T1>
  static typename std::enable_if<
      (is_complex<T1>::value) &&
          std::is_same<typename T1::value_type, double>::value,
      void>::type
  p_sdcz_getri(MKL_INT *n, T1 *local_a, MKL_INT *ia, MKL_INT *ja,
               MKL_INT *desca, MKL_INT *local_ipiv, T1 *local_work,
               MKL_INT *local_lwork, MKL_INT *local_iwork, MKL_INT *liwork) {
    MKL_INT info = 1;
    pzgetri(n, local_a, ia, ja, desca, local_ipiv, local_work, local_lwork,
            local_iwork, liwork, &info);
    assert_macro(info == 0);
  }

  // p_sdcz_gemm ------------------------------------------------------
  template <typename T1>
  static typename std::enable_if<
      !(is_complex<T1>::value) && std::is_same<T1, float>::value, void>::type
  p_sdcz_gemm(const char *transa, const char *transb, const MKL_INT *m,
              const MKL_INT *n, const MKL_INT *k, const T1 *alpha, const T1 *a,
              const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca,
              const T1 *b, const MKL_INT *ib, const MKL_INT *jb,
              const MKL_INT *descb, const T1 *beta, T1 *c, const MKL_INT *ic,
              const MKL_INT *jc, const MKL_INT *descc) {

    psgemm(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb,
           beta, c, ic, jc, descc);
  }
  template <typename T1>
  static typename std::enable_if<
      !(is_complex<T1>::value) && std::is_same<T1, double>::value, void>::type
  p_sdcz_gemm(const char *transa, const char *transb, const MKL_INT *m,
              const MKL_INT *n, const MKL_INT *k, const T1 *alpha, const T1 *a,
              const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca,
              const T1 *b, const MKL_INT *ib, const MKL_INT *jb,
              const MKL_INT *descb, const T1 *beta, T1 *c, const MKL_INT *ic,
              const MKL_INT *jc, const MKL_INT *descc) {

    pdgemm(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb,
           beta, c, ic, jc, descc);
  }
  template <typename T1>
  static typename std::enable_if<
      (is_complex<T1>::value) &&
          std::is_same<typename T1::value_type, float>::value,
      void>::type
  p_sdcz_gemm(const char *transa, const char *transb, const MKL_INT *m,
              const MKL_INT *n, const MKL_INT *k, const T1 *alpha, const T1 *a,
              const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca,
              const T1 *b, const MKL_INT *ib, const MKL_INT *jb,
              const MKL_INT *descb, const T1 *beta, T1 *c, const MKL_INT *ic,
              const MKL_INT *jc, const MKL_INT *descc) {

    pcgemm(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb,
           beta, c, ic, jc, descc);
  }
  template <typename T1>
  static typename std::enable_if<
      (is_complex<T1>::value) &&
          std::is_same<typename T1::value_type, double>::value,
      void>::type
  p_sdcz_gemm(const char *transa, const char *transb, const MKL_INT *m,
              const MKL_INT *n, const MKL_INT *k, const T1 *alpha, const T1 *a,
              const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca,
              const T1 *b, const MKL_INT *ib, const MKL_INT *jb,
              const MKL_INT *descb, const T1 *beta, T1 *c, const MKL_INT *ic,
              const MKL_INT *jc, const MKL_INT *descc) {

    pzgemm(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb,
           beta, c, ic, jc, descc);
  }

public:
  // matrixInversion() for real or complex data
  static void matrixInversion(MPI_Comm comm, T *global_A, int M, int N,
                              const int MB, const int NB, const int NPROW,
                              const int NPCOL) {

    int ia = 1, ja = 1;
    DistributedMatrix<T>::Initialize({NPROW, NPCOL});

    typename DistributedMatrix<T>::LocalMatrix local_A(global_A, M, N, MB, NB);

    MKL_INT *local_ipiv = new MKL_INT[local_A.SUB_ROWS + MB];

    //-----------------------------------
    p_sdcz_getrf<T>(&M, &N, local_A.data(), &ia, &ja, local_A.DESC, local_ipiv);

    // Query call
    T query_local_work[1];
    MKL_INT local_lwork = -1;
    MKL_INT query_local_iwork[1];
    MKL_INT liwork = -1;
    p_sdcz_getri<T>(&N, local_A.data(), &ia, &ja, local_A.DESC, local_ipiv,
                    query_local_work, &local_lwork, query_local_iwork, &liwork);

    // Computation call
    local_lwork = static_cast<MKL_INT>(std::abs(query_local_work[0]));
    liwork = static_cast<MKL_INT>(std::abs(query_local_iwork[0]));
    T *local_work = new T[local_lwork];
    MKL_INT *local_iwork = new MKL_INT[local_lwork];

    p_sdcz_getri<T>(&N, local_A.data(), &ia, &ja, local_A.DESC, local_ipiv,
                    local_work, &local_lwork, local_iwork, &liwork);

    // Deallocate memory
    delete[] local_work;
    delete[] local_iwork;
    //-----------------------------------

    delete[] local_ipiv;
    local_A.constructGlobalMatrix(global_A);

    DistributedMatrix<T>::Finalize();
  }

  // EigensolverForHermitian() for real data
  template <typename T1>
  static
      typename std::enable_if<!(is_complex<T1>::value), std::vector<T1>>::type
      EigensolverForHermitian(MPI_Comm comm, T1 *global_A, int M, int N, int MB,
                              int NB, const int NPROW, const int NPCOL) {

    DistributedMatrix<T1>::Initialize({NPROW, NPCOL});

    typename DistributedMatrix<T1>::LocalMatrix local_A(global_A, M, N, MB, NB);
    typename DistributedMatrix<T1>::LocalMatrix local_eigvecs(M, N, MB, NB);
    std::vector<T1> global_eigvals(M);

    //-----------------------------
    int IA = 1, JA = 1, IZ = 1, JZ = 1;
    char jobz = 'V';
    char uplo = 'U';
    T1 work_query;
    // Query (lwork = -1) and allocate WORK
    MKL_INT lwork = -1;
    p_sd_syev(&jobz, &uplo, &M, local_A.data(), &IA, &JA, local_A.DESC,
              global_eigvals.data(), local_eigvecs.data(), &IZ, &JZ,
              local_eigvecs.DESC, &work_query, &lwork);
    lwork = static_cast<MKL_INT>(work_query);
    T1 *work = new T1[lwork];
    // Do the eigendecomposition
    p_sd_syev(&jobz, &uplo, &M, local_A.data(), &IA, &JA, local_A.DESC,
              global_eigvals.data(), local_eigvecs.data(), &IZ, &JZ,
              local_eigvecs.DESC, work, &lwork);
    // Deallocate memory
    delete[] work;
    //-------------------------------

    local_eigvecs.constructGlobalMatrix(global_A); // overwrites A with
                                                   // eigvectors

    DistributedMatrix<T1>::Finalize();
    return global_eigvals;
  }

  // EigensolverForHermitian() for complex data
  template <typename T1>
  static typename std::enable_if<(is_complex<T1>::value),
                                 std::vector<typename T1::value_type>>::type
  EigensolverForHermitian(MPI_Comm comm, T1 *global_A, int M, int N, int MB,
                          int NB, const int NPROW, const int NPCOL) {

    DistributedMatrix<T1>::Initialize({NPROW, NPCOL});

    typename DistributedMatrix<T1>::LocalMatrix local_A(global_A, M, N, MB, NB);
    typename DistributedMatrix<T1>::LocalMatrix local_eigvecs(M, N, MB, NB);

    using T2 = typename T1::value_type;
    std::vector<T2> global_eigvals(M);

    //--------------------------------------
    int IA = 1, JA = 1, IZ = 1, JZ = 1;
    char jobz = 'V';
    char uplo = 'U';
    T1 work_query;
    T2 rwork_query;
    // Query (lwork = -1) and allocate WORK
    MKL_INT lwork = -1;
    MKL_INT lrwork = -1;
    p_cz_heev(&jobz, &uplo, &M, local_A.data(), &IA, &JA, local_A.DESC,
              global_eigvals.data(), local_eigvecs.data(), &IZ, &JZ,
              local_eigvecs.DESC, &work_query, &lwork, &rwork_query, &lrwork);
    lwork = static_cast<MKL_INT>(work_query.real());
    lrwork = static_cast<MKL_INT>(rwork_query);
    T1 *work = new T1[lwork];
    T2 *rwork = new T2[lrwork];
    // Do the eigendecomposition
    p_cz_heev(&jobz, &uplo, &M, local_A.data(), &IA, &JA, local_A.DESC,
              global_eigvals.data(), local_eigvecs.data(), &IZ, &JZ,
              local_eigvecs.DESC, work, &lwork, rwork, &lrwork);
    // Deallocate memory
    delete[] work;
    delete[] rwork;
    //--------------------------------------

    local_eigvecs.constructGlobalMatrix(global_A); // overwrites A with
                                                   // eigvectors

    DistributedMatrix<T1>::Finalize();
    return global_eigvals;
  }
};

// Definition and initialization of the static member variables
template <typename T> int DistributedMatrix<T>::ICTXT = -1;
template <typename T> int DistributedMatrix<T>::NPROW = -1;
template <typename T> int DistributedMatrix<T>::NPCOL = -1;
template <typename T> int DistributedMatrix<T>::MYROW = -1;
template <typename T> int DistributedMatrix<T>::MYCOL = -1;

// Explicit instantiation
// template class Scalapack<float>;
// template class Scalapack<double>;

/*
template void Scalapack<float>::matrixInversion(MPI_Comm comm, float *global_A,
                                                int M, int N, const int MB,
                                                const int NB, const int NPROW,
                                                const int NPCOL);

template void Scalapack<double>::matrixInversion(MPI_Comm comm,
                                                 double *global_A, int M, int N,
                                                 const int MB, const int NB,
                                                 const int NPROW,
                                                 const int NPCOL);
//-----
template std::vector<float>
Scalapack<float>::EigensolverForHermitian(MPI_Comm comm, float *global_A, int M,
                                          int N, int MB, int NB,
                                          const int NPROW, const int NPCOL);

template std::vector<float>
Scalapack<fcomplex>::EigensolverForHermitian(MPI_Comm comm, fcomplex *global_A,
                                             int M, int N, int MB, int NB,
                                             const int NPROW, const int NPCOL);

template std::vector<double>
Scalapack<double>::EigensolverForHermitian(MPI_Comm comm, double *global_A,
                                           int M, int N, int MB, int NB,
                                           const int NPROW, const int NPCOL);

template std::vector<typename dcomplex::value_type>
Scalapack<dcomplex>::EigensolverForHermitian(MPI_Comm comm, dcomplex *global_A,
                                             int M, int N, int MB, int NB,
                                             const int NPROW, const int NPCOL);
*/

#endif // CPP_PARALLELLINEARALGEBRA_HPP
