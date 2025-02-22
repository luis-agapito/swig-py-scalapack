#include "cpp_functions.h"
#include "mpi.h"
#include <iostream>
#include <string>

#include "ParallelLinearAlgebra.hpp"
#include "mpi.h"
#include <Eigen/Core>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

Eigen::VectorXd pdsyev(MPI_Comm comm, const std::string &text, double *m2d,
                       int m2d_m, int m2d_n) {
  int size;
  int my_rank;
  // MPI_Init();
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &my_rank);

  // printf("Rank=%d; Size=%d; PID=%d\n", my_rank, size, getpid());
  // std::cout << text << std::endl;

  const int M = m2d_m, N = m2d_n;
  const int MB = 2, NB = 2; // hardwired for now.

  ParallelLinearAlgebra::Initialize({MB, NB});

  ParallelLinearAlgebra::GMatrix A(M, N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      A.Set(i, j, m2d[i * N + j]);
    }
  }

  ParallelLinearAlgebra::LMatrix lA(A, MB, NB);    // the original matrix
  ParallelLinearAlgebra::LMatrix lZ(M, N, MB, NB); // the eigenvectors

  ParallelLinearAlgebra::GMatrix W(M, 1); // eigenvalues

  ParallelLinearAlgebra::callPDSYEV(lA, W, lZ);
  ParallelLinearAlgebra::GMatrix Z = lZ.ConstructGlobalMatrix();

  // if (my_rank == 0) {
  //   std::cout << "W\n" << W << std::endl;
  //   std::cout << "Z\n" << Z << std::endl;
  // }

  // Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> aux(z.Data(), N,
  // M);

  // Eigen::Map<
  //     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  //     aux(Z.Data(), M, N);

  Eigen::Map<Eigen::VectorXd> aux(W.Data(), M);

  // m2d = &(Z.Data()[0]); //these two lines lock m2d
  // m2d = aux.data();
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      m2d[i * N + j] = Z.Data()[i * N + j];
    }
  }

  // MPI_Finalize(); Don't close the MPI environment. Let Python's main do that.

  return aux;
}
