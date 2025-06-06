#include "ParallelLinearAlgebra.hpp"

// Specialization of a function template
template <> MPI_Datatype resolveMPIType<float>() { return MPI_FLOAT; }
template <> MPI_Datatype resolveMPIType<double>() { return MPI_DOUBLE; }
template <> MPI_Datatype resolveMPIType<std::complex<float>>() {
  return MPI_COMPLEX;
}
template <> MPI_Datatype resolveMPIType<dcomplex>() {
  return MPI_DOUBLE_COMPLEX;
}
