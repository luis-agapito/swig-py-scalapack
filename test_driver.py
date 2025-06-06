from mpi4py import MPI
import mymodule
import numpy

numpy.set_printoptions(linewidth=100)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

d = numpy.array(
    [[3, 5, 2, 1], [5, 1, 0, 3], [2, 0, 1, 2], [1, 3, 2, 1]],
    dtype=numpy.double,
)

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
    # -------------------------
    # dd = numpy.array(
    #    [[3, 5, 2, 1], [5, 1, 0, 3], [2, 0, 1, 2], [1, 3, 2, 1]],
    #    dtype=numpy.double,
    # )
    # eigvals, eigvecs = numpy.linalg.eig(dd.T)
    # print("\nnumpy eigenvecs\n", eigvecs)
