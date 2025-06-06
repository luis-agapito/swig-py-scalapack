#include "cpp_functions.h"
#include "ParallelLinearAlgebra.hpp"
#include "mpi.h"
#include <Eigen/Core>

// pdsyev()
Eigen::VectorXd pdsyev(MPI_Comm comm, double *array, int array_m, int array_n,
                       const int MB, const int NB, const int NPROW,
                       const int NPCOL) {
  int size;
  int my_rank;
  // MPI_Init();
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &my_rank);

  const int M = array_m, N = array_n;

  DistributedMatrix<double>::Initialize({NPROW, NPCOL});

  DistributedMatrix<double>::GlobalMatrix A(M, N);
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      A.Set(i, j, array[j * M + i]);
    }
  }

  DistributedMatrix<double>::LocalMatrix lA(A, MB, NB); // the original matrix
  DistributedMatrix<double>::LocalMatrix lZ(M, N, MB, NB); // the eigenvectors

  DistributedMatrix<double>::GlobalMatrix W(M, 1); // eigenvalues

  //----------------------------------
  // eigenvectors/values of a real symmetric matrix
  int IA = 1, JA = 1, IZ = 1, JZ = 1;

  const char jobz = 'V';
  const char uplo = 'U';
  double work_query;
  MKL_INT INFO;

  // Query (lwork = -1) and allocate WORK
  MKL_INT lwork = -1;
  pdsyev(&jobz, &uplo, &lA.M, lA.data(), &IA, &JA, lA.DESC, W.data(), lZ.data(),
         &IZ, &JZ, lZ.DESC, &work_query, &lwork, &INFO);

  lwork = static_cast<MKL_INT>(work_query);
  double *work = new double[lwork];

  // Do the eigendecomposition
  pdsyev(&jobz, &uplo, &(lA.M), lA.data(), &IA, &JA, lA.DESC, W.data(),
         lZ.data(), &IZ, &JZ, lZ.DESC, work, &lwork, &INFO);

  // Deallocate memory
  delete[] work;

  if (INFO != 0) {
    std::cerr << "Error: INFO of callPDSYEV is not zero but " << INFO
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  assert(INFO == 0);
  //-----------------------------

  DistributedMatrix<double>::GlobalMatrix Z = lZ.constructGlobalMatrix();

  Eigen::Map<Eigen::VectorXd> aux(W.data(), M);

  // m2d = &(Z.Data()[0]); //these two lines lock m2d
  // m2d = aux.data();
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      array[j * M + i] = Z.data()[j * M + i];
    }
  }
  return aux;
}

// MyClass::matrixInversion()
template <typename T1, typename T2>
void MyClass<T1, T2>::matrixInversion(T2 *global_A, int M, int N) {
  Scalapack<T2>::matrixInversion(comm_, global_A, M, N, MB_, NB_, NPROW_,
                                 NPCOL_);
};

// MyClass::eigensolverSymmetricHermitian()
template <typename T1, typename T2>
std::vector<T1>
MyClass<T1, T2>::eigensolverForSymmetricOrHermitian(T2 *global_A, int M,
                                                    int N) {
  std::vector<T1> out = Scalapack<T2>::EigensolverForHermitian(
      comm_, global_A, M, N, MB_, NB_, NPROW_, NPCOL_);
  return out;
};

// Explicit class instantiation
template class MyClass<float, float>;
template class MyClass<double, double>;
template class MyClass<float, fcomplex>;
template class MyClass<double, dcomplex>;
