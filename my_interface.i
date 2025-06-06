%module mymodule
%{
#define SWIG_FILE_WITH_INIT
#include "cpp_functions.h"
%}

%include "std_string.i"
%include "std_vector.i"
%include "eigen.i"
%include "numpy.i"
%include "mpi4py.i"
%mpi4py_typemap(Comm, MPI_Comm)
%eigen_typemaps(Eigen::MatrixXd)
%eigen_typemaps(Eigen::VectorXd)

%typemap(in) (double* INPLACE_ARRAY2_F, int DIM1, int DIM2) {
    PyArrayObject *array = (PyArrayObject *)PyArray_FromAny($input, PyArray_DescrFromType(NPY_DOUBLE), 2, 2, NPY_ARRAY_ALIGNED, NULL);
    if (!array || !(PyArray_IS_C_CONTIGUOUS(array) || PyArray_IS_F_CONTIGUOUS(array))) {
        PyErr_SetString(PyExc_TypeError, "Array must be C- or Fortran-contiguous.");
        return NULL;
    }
    $1 = (double *)PyArray_DATA(array);
    $2 = PyArray_DIMS(array)[0];
    $3 = PyArray_DIMS(array)[1];
    Py_DECREF(array);
}

%init %{
import_array();
%}

namespace std {
%template(DoubleVector) vector<double>;
%template(FloatVector) vector<float>;
}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* array, int array_m, int array_n)};
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* global_A, int M, int N)};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* global_A, int M, int N)};
%apply (std::complex<float>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<float>* global_A, int M, int N)};
%apply (std::complex<double>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<double>* global_A, int M, int N)};

%feature("autodoc", "2");
%include "cpp_functions.h"

%template(MyClassFloat) MyClass<float,float>;
%template(MyClassDouble) MyClass<double,double>;
%template(MyClassComplexDouble) MyClass<double,std::complex<double>>;
%template(MyClassComplexFloat) MyClass<float,std::complex<float>>;

