from mpi4py import MPI
import mymodule
import numpy

numpy.set_printoptions(linewidth=120)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# size of blocks in columns and rows
MB = NB = 1
# number of rows (NPROW) and columns (NPCOL) of the two dimensional process grid
NPROW = NPCOL = 2

ctypes = ["float", "double", "std::complex<single>", "std::complex<double>"]
types = [numpy.single, numpy.double, numpy.csingle, numpy.cdouble]
modules = [
    mymodule.MyClassFloat(comm, MB, NB, NPROW, NPCOL),
    mymodule.MyClassDouble(comm, MB, NB, NPROW, NPCOL),
    mymodule.MyClassComplexFloat(comm, MB, NB, NPROW, NPCOL),
    mymodule.MyClassComplexDouble(comm, MB, NB, NPROW, NPCOL),
]

for i in [2, 3]:
    A = numpy.array(
        [
            [3 + 0j, 5 + 1j, 2 + 2j, 1 + 3j],
            [5 - 1j, 1 + 0j, 0 + 0j, 3 + 0j],
            [2 - 2j, 0 + 0j, 1 + 0j, 2 + 0j],
            [1 - 3j, 3 + 0j, 2 + 0j, 1 + 0j],
        ],
        dtype=types[i],
    )

    if i == 2 and rank == 0:
        print("\n========== matrix A")
        print(A.T)
        eigvals, eigvecs = numpy.linalg.eigh(A.T)
        print("\n========== numpy eigenvectors(A):")
        print(eigvecs)
        # print(eigvecs * eigvecs.conj())
        print("           numpy eigenvalues(A):")
        print(eigvals)

    eigvals = modules[i].eigensolverForSymmetricOrHermitian(A)

    if rank == 0:
        print("\n========== ScaLAPACK eigenvectors(A) with data type=", ctypes[i], ":")
        print(A.T)
        # print((A * A.conj()).T)
        print("           ScaLAPACK eigenvalues(A) with data type=", ctypes[i], ":")
        print(eigvals)
