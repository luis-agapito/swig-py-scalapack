#include "mpi.h"
#include <Eigen/Core>
#include <vector>

// pdsyev()
Eigen::VectorXd pdsyev(MPI_Comm comm, double *array, int array_m, int array_n,
                       const int MB, const int NB, const int NPROW,
                       const int NPCOL);
//
// Class MyClass()
template <typename T1, typename T2> class MyClass {
private:
  int MB_, NB_, NPROW_, NPCOL_;
  MPI_Comm comm_;

public:
  MyClass(MPI_Comm comm, int MB, int NB, int NPROW, int NPCOL)
      : comm_(comm), MB_(MB), NB_(NB), NPROW_(NPROW), NPCOL_(NPCOL) {};

  void matrixInversion(T2 *global_A, int M, int N);

  std::vector<T1> eigensolverForSymmetricOrHermitian(T2 *global_A, int M,
                                                     int N);
};
