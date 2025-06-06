from mpi4py import MPI
import mymodule
import numpy

numpy.set_printoptions(linewidth=100)

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

for i in range(4):
    A = numpy.array(
        [[3, 5, 2, 1], [4, 3, 0, 1], [3, 2, 1, 0], [1, 3, 2, 1]],
        dtype=types[i],
    )
    if i == 0 and rank == 0:
        print("\n\n========== A")
        print(A.T)
        print("\n\n========== numpy inverse(A)")
        print(numpy.linalg.inv(A.T))
    modules[i].matrixInversion(A)
    if rank == 0:
        print("\n\n========== ScaLAPACK inverse(A) with data type=", ctypes[i])
        print(A.T)
