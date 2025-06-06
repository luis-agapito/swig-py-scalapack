# Performing linear algebra in parallel  (`ScaLAPACK`) from Python

Diagonalization of matrices is one of the most common and time-consuming linear algebra operations in scientific computing applications. There are well-established mathematical libraries, such as `LAPACK` (serial) and `ScaLAPACK` (parallel), that provide optimized solutions for this problem, often tailored for use on supercomputers. Although these libraries are implemented in Fortran, they also offer APIs for calling them from languages like C and C++.

- In this setup, we use Python with the `mpi4py` module for the front-end. Python, being an interpreted language, is highly suitable for quick prototyping and testing new ideas. However, it is slower compared to compiled languages like Fortran, C, or C++, which are converted into machine code for faster execution.

- For the backend, we opt for C++ over Fortran. While both are fast compiled languages, C++ offers greater flexibility for modern application development. We will utilize `ScaLAPACK`'s C bindings to access its functionality within C++.

- Parallelizing matrix computations with MPI is necessary, as real-world matrices are often too large for serial computation. Although GPU parallelization could yield even better performance, we focus on CPU parallelization for now.

- To integrate everything, we will use `SWIG` to bundle the C++ routines into Python modules.

This approach allows us to combine the speed of compiled parallel C++/`ScaLAPACK` routines with the flexibility of Python, creating a streamlined and efficient solution for large-scale linear algebra operations.

<p align="center">
<img src="human_vs_machine_time.png" alt=" " width="250"/>
</p>
<p align="center">
<p align="center"><sup>https://www.cgl.ucsf.edu/Outreach/bmi219/slides/swc/lec/py01.html</sup></p>

## Examples

<details>
<summary>1. Parallel eigensolver for a fixed data type: Basic workflow</summary>

### Parallel eigensolver for matrices in `double` precision ###

1. The goal is to create a Python function (`pdsyev`) that calls ScaLAPACK to compute the eigenvalues and eigenvectors of a symmetric matrix `d`:

   ```python
   w = mymodule.pdsyev(comm, d, MB, NB, NPROW, NPCOL) 
   ```

   On output, the eigenvalues and eigenvectors are stored in `w` and `d`, respectively. This example uses 4 CPUs arranged in a 2x2 (`NPROW`x`NPCOL`) MPI grid of processors. ScaLAPACK is configured to split the matrix's rows and columns across processors using blocks of size `MB=1` and `NB=1`, respectively.

  Using `SWIG`, this Python function calls the following C++ routine, defined in `cpp_functions.cpp`:

   ```cpp
   Eigen::VectorXd pdsyev(MPI_Comm comm, double *array, int array_m, int array_n,
                          const int MB, const int NB, const int NPROW,
                          const int NPCOL)
   ```

   This C++ function relies on custom classes defined in `library/ParallelLinearAlgebra.h` (used more extensively in later examples). To build the dynamic library (`libplinalg.so`),  use the following commands:

   ```bash
   cd library/
   make clean
   make
   ```

   Next, compile the C++ code into a Python-callable module using `SWIG`:

   ```bash
   sh ./compile.sh
   ```

   This generates a dynamic library `_mymodule.so` that can be loaded from Python.

2. To test the functionality, use the following script (`test_driver.py`):

   ```python
   from mpi4py import MPI
   import mymodule
   import numpy
 
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
 
   d = numpy.array([[3,5,2,1],
                    [5,1,0,3],
                    [2,0,1,2],
                    [1,3,2,1]
                    ], dtype=numpy.double)
 
   # size of blocks in columns and rows
   MB = NB = 1
   # number of rows (NPROW) and columns (NPCOL) of the two dimensional process grid
   NPROW = NPCOL = 2
 
   w = mymodule.pdsyev(comm, d, MB, NB, NPROW, NPCOL)
 
   if rank == 0:
       print("Eigenvectors:")
       print(d.T)
       print("\nEigenvalues:")
       print(w)
   ```

   Since `mymodule.pdsyev()` is an MPI-parallel routine, run the script using MPI:

   ```bash
   mpiexec -np 4 test_driver.py
   ```

   The script successfully prints the eigenvalues and eigenvectors of matrix `d`, confirming that the parallel routine works as expected:

   ```
   Eigenvectors:
   [[-0.49718297  0.44119429  0.33641001  0.66707196]
    [ 0.68625837 -0.32691264  0.29372946  0.5795693 ]
    [ 0.33524331  0.5263284  -0.73279059  0.27130847]
    [-0.4116679  -0.649195   -0.51338997  0.38145266]]
   
   Eigenvalues:
   [[-4.42203738]
    [ 0.20961624]
    [ 1.48303009]
    [ 8.72939105]]
   ```

   To save on RAM, the input matrix `d` is overwritten with the eigenvectors upon output. Note that the matrix is transposed (`d.T`) before printing the eigenvectors. This is because the `SWIG` template (see `INPLACE_ARRAY2` in `my_interface.i`) that converts the C++ output variable (`double *array`, which is stored column-major) to a numpy array assumes that the data is stored row-major. Similarly, on input `SWIG` is assuming that `d` is stored row-major whereas ScaLAPACK uses column-major storage. Thus, the ScaLAPACK backend is acting on `d.T` instead of `d`. However, here we don't need to take the transpose because the matrix is symmetric.

</details>

<details>
<summary>2. Parallel matrix inversion for multiple data type: Using C++ templates</summary>

### Templates in C++ and `SWIG` ###

The previous example was hard-coded to work with real matrices in double precision (`double`  in C++ or `numpy.float64` in Python) and will crash for other types of data. However, in practical applications, we need to compute not only with real data but also with complex data. Moreover, by using different precisions (i.e., single instead of double precision), we can achieve shorter computational and reduced memory requirements, at the expense of some accuracy.

1. The goal here is to implement a Python function to invert a matrix in parallel, where the matrix can have any of the following C++ data types: `float`, `double`, `std::complex<single>`, or `std::complex<double>`, which correspond to `numpy.single`, `numpy.double`, `numpy.csingle`, and `numpy.cdouble`, respectively. Below is a usage example for a complex-double matrix:

   ```python
   from mpi4py import MPI
   import mymodule

   comm = MPI.COMM_WORLD

   # Holds the information required by ScaLAPACK.
   Parallel = mymodule.MyClassComplexDouble(comm, MB, NB, NPROW, NPCOL)
 
   A = numpy.array(
       [[3+1j, 5+2j, 2+0j, 1+1j], 
        [5-2j, 3-1j, 0+0j, 1+2j], 
        [2-0j, 0-0j, 1+2j, 2+3j], 
        [1-1j, 1-2j, 2-3j, 1+4j]],
       dtype=numpy.cdouble)

   # Compute the inverse
   Parallel.matrixInversion(A)

   ```

2. First, we implement a C++ class `MyClass` (in `cpp_functions.{h,cpp}`) as a container for the member functions `matrixInversion()` and `eigensolverForSymmetricOrHermitian()`, as well as the parallel setup information required by ScaLAPACK (i.e., `comm_`, `MB_`, `NB_`, `NPROW_`, and `NPCOL`).

   ```cpp
   // Class MyClass()
   template <typename T1, typename T2> class MyClass {
   private:
     int MB_, NB_, NPROW_, NPCOL_;
     MPI_Comm comm_;
   
   public:
     MyClass(MPI_Comm comm, int MB, int NB, int NPROW, int NPCOL)
         : comm_(comm), MB_(MB), NB_(NB), NPROW_(NPROW), NPCOL_(NPCOL) { ... };
   
     void matrixInversion(T2 *global_A, int M, int N) { ... };
   
     std::vector<T1> eigensolverForSymmetricOrHermitian(T2 *global_A, int M, int N) { ... };
   };

   //Explicit class instantiation
   template class MyClass<float, float>;
   template class MyClass<double, double>;
   template class MyClass<float, std::complex<float>>;
   template class MyClass<double, std::complex<double>>;

   ```

   Strictly speaking, the implementation shown above is not a class but a **class template**, which requires two template parameters: `T1` and `T2`. This template is used by the compiler to generate multiple classes depending on the parameters passed. With the last four lines, the compiler defines four different classes, each supporting a different combination of data types. For instance, one could directly call matrix inversion for complex data in single precision with the following C++ code:

   ```cpp
   std::complex<float> array[16] = {{1,1}, {2,2}, ... };
   MyClass<float, std::complex<float>> Parallel(comm, MB, NB, NPROW, NPCOL);
   Parallel::matrixInversion(array, 4, 4);
   ```

   For the specific case of the `matrixInversion()` function, we only need a single template  parameter (`T1`). However, this class template uses two parameters (`T1`, `T2`) to support the other member function.

3. Next, we create Python-accessible wrapper classes for the instantiated C++ templates. This is done in the `SWIG` interface file `my_interface.i`.

   ```swig
   %template(MyClassFloat) MyClass<float,float>;
   %template(MyClassDouble) MyClass<double,double>;
   %template(MyClassComplexDouble) MyClass<double,std::complex<double>>;
   %template(MyClassComplexFloat) MyClass<float,std::complex<float>>;
   ```

   With that, we can, for example, use the Python class `MyClassDouble()` to access the C++ class `MyClass<double, double>()`.

4. Finally, we test that the parallel matrix inversion works for the following data types: `float`, `double`, `std::complex<single>`, and `std::complex<double>`. Run the test script:

   ```bash
   mpiexec -np 4 test_inverse.py
   ```

   It produces the following output, demonstrating the correctness of the implementation.

   ```
   ========== A
   [[3. 4. 3. 1.]
    [5. 3. 2. 3.]
    [2. 0. 1. 2.]
    [1. 1. 0. 1.]]

   ========== numpy inverse(A)
   [[-0.4  0.9 -0.6 -1.1]
    [ 0.2 -0.2 -0.2  0.8]
    [ 0.4 -0.4  0.6 -0.4]
    [ 0.2 -0.7  0.8  1.3]]

   ========== ScaLAPACK inverse(A) with data type= float
   [[-0.4000001   0.9000002  -0.60000014 -1.1000004 ]
    [ 0.20000002 -0.20000005 -0.19999997  0.80000013]
    [ 0.40000004 -0.40000004  0.6        -0.39999992]
    [ 0.20000006 -0.70000017  0.80000013  1.3000003 ]]

   ========== ScaLAPACK inverse(A) with data type= double
   [[-0.4  0.9 -0.6 -1.1]
    [ 0.2 -0.2 -0.2  0.8]
    [ 0.4 -0.4  0.6 -0.4]
    [ 0.2 -0.7  0.8  1.3]]

   ========== ScaLAPACK inverse(A) with data type= std::complex<single>
   [[-0.39999995+0.j  0.89999986+0.j -0.5999999 +0.j -1.0999999 +0.j]
    [ 0.19999999+0.j -0.19999996+0.j -0.20000002+0.j  0.79999995+0.j]
    [ 0.39999995+0.j -0.3999999 +0.j  0.5999999 +0.j -0.4000001 +0.j]
    [ 0.1999999 +0.j -0.6999998 +0.j  0.79999983+0.j  1.2999998 +0.j]]

   ========== ScaLAPACK inverse(A) with data type= std::complex<double>
   [[-0.4+0.j  0.9+0.j -0.6+0.j -1.1+0.j]
    [ 0.2+0.j -0.2+0.j -0.2+0.j  0.8+0.j]
    [ 0.4+0.j -0.4+0.j  0.6+0.j -0.4+0.j]
    [ 0.2+0.j -0.7+0.j  0.8+0.j  1.3+0.j]]

   ```

</details>

<details>
<summary>3. Parallel eigensolver for multiple data types</summary>

### Parallel eigensolver for multiple data types ###

The method `eigensolverForSymmetricOrHermitian()` of the class template `MyClass<T1, T2>` (defined in `cpp_functions.{h,cpp}`) implements a parallel eigensolver that supports multiple data types. This method internally calls `EigensolverForHermitian()`, a member function of the class template `Scalapack<T>` (found in `library/ParallelLinearAlgebra.h`). This class template is designed as a C++ library that provides parallel routines for eigensolver, matrix inversion, and matrix multiplication. The following script test the eigensolver:

   ```python
   from mpi4py import MPI
   import mymodule

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   
   # size of blocks in columns and rows
   MB = NB = 1
   # number of rows (NPROW) and columns (NPCOL) of the two dimensional process grid
   NPROW = NPCOL = 2

   # Holds the information required by ScaLAPACK.
   Parallel = mymodule.MyClassComplexDouble(comm, MB, NB, NPROW, NPCOL)
 
   A = numpy.array(
       [[3 + 0j, 5 + 1j, 2 + 2j, 1 + 3j],
        [5 - 1j, 1 + 0j, 0 + 0j, 3 + 0j],
        [2 - 2j, 0 + 0j, 1 + 0j, 2 + 0j],
        [1 - 3j, 3 + 0j, 2 + 0j, 1 + 0j]],
       dtype=numpy.cdouble)

   eigvals = Parallel.eigensolverForSymmetricOrHermitian(A)

  if rank == 0:
      print("\n========== matrix A")
      print(A.T)
      print("\n========== ScaLAPACK eigenvectors(A)")
      print(A.T)
      print("           ScaLAPACK eigenvalues(A)")
      print(eigvals)
   ```

Note that the matrix is transposed `(A.T)` before being displayed. This is required because the `SWIG` interface and ScaLAPACK use different memory layouts: row-major and column-major, respectively.

Run the following command to execute the test:

   ```bash
   mpiexec -np 4 test_eigensolver.py
   ```

The output confirms that the Python implementation works correctly for multiple data types:

```
========== matrix A
[[3.+0.j 5.-1.j 2.-2.j 1.-3.j]
 [5.+1.j 1.+0.j 0.+0.j 3.+0.j]
 [2.+2.j 0.+0.j 1.+0.j 2.+0.j]
 [1.+3.j 3.+0.j 2.+0.j 1.+0.j]]

========== numpy eigenvectors(A):
[[-0.5575604 +0.j          0.42647246+0.j         -0.2279303 +0.j         -0.6747558 +0.j        ]
 [ 0.63716435-0.04343621j -0.04587614-0.08101396j -0.25537   +0.5124243j  -0.46923044-0.1884073j ]
 [ 0.2993389 +0.09647986j  0.3585622 -0.44315562j  0.5679991 -0.39662963j -0.21259072-0.22583443j]
 [-0.3294669 +0.2716628j  -0.6884423 -0.10269763j  0.34484187+0.1462918j  -0.27936608-0.33880475j]]
           numpy eigenvalues(A):
[-4.9265757  -0.46122402  1.411661    9.976138  ]

========== ScaLAPACK eigenvectors(A) with data type= std::complex<single> :
[[ 0.43018156+0.35470736j  0.42180514-0.06292233j  0.2098296 -0.08901569j -0.42926782+0.52059996j]
 [-0.51923263-0.37183666j -0.0573273 -0.07335877j  0.03496893-0.57146275j -0.44387966+0.24216783j]
 [-0.16957432-0.2648709j   0.2892546 -0.4912081j  -0.36799353+0.58695716j -0.30948627+0.02035002j]
 [ 0.4270236 +0.j         -0.69606   +0.j         -0.37458932+0.j         -0.43912885+0.j        ]]
           ScaLAPACK eigenvalues(A) with data type= std::complex<single> :
(-4.926576137542725, -0.46122482419013977, 1.4116613864898682, 9.976139068603516)

========== ScaLAPACK eigenvectors(A) with data type= std::complex<double> :
[[ 0.43018165+0.35470741j  0.42180509-0.06292232j  0.20982952-0.0890157j  -0.42926793+0.52060011j]
 [-0.51923265-0.37183677j -0.05732698-0.0733587j   0.03496841-0.57146276j -0.44387974+0.2421679j ]
 [-0.16957442-0.26487102j  0.28925429-0.49120842j -0.36799286+0.58695771j -0.30948649+0.02034999j]
 [ 0.42702356+0.j         -0.69606007+0.j         -0.37458939+0.j         -0.43912875+0.j        ]]
           ScaLAPACK eigenvalues(A) with data type= std::complex<double> :
(-4.926575528924668, -0.46122402739986995, 1.4116610770365932, 9.976138479287947)
```

</details>
