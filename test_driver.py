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

w = mymodule.pdsyev(comm, "input_string", d) 

if rank==0:
    print('Eigenvectors:')
    print(d)
    print('\nEigenvalues:')
    print(w)

