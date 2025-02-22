%module mymodule
%{
#define SWIG_FILE_WITH_INIT
#include "cpp_functions.h"
%}

%include "std_string.i"
%include "eigen.i"
%include "numpy.i"
%include "mpi4py.i"
%mpi4py_typemap(Comm, MPI_Comm)
%eigen_typemaps(Eigen::MatrixXd)
%eigen_typemaps(Eigen::VectorXd)

//%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* seq, int m, int n)};
//%apply (double ARGOUT_ARRAY2[ANY][ANY]) {(double out_seq[3][2])};

%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* m2d, int m2d_m, int m2d_n)};
%feature("autodoc", "2");
%include "cpp_functions.h"
