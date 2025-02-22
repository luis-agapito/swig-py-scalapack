#include "mpi.h"
#include <Eigen/Core>
#include <string>

Eigen::VectorXd pdsyev(MPI_Comm comm, const std::string &text, double *m2d,
                       int m2d_m, int m2d_n);
