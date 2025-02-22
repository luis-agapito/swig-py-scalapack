rm -f *.o
rm -f *.so
rm -f *.cxx

MKLROOT=/home/beto/local_data/libraries/intel_mkl/mkl/2025.0/
EIGENROOT=/home/beto/local_data/libraries/eigen-3.4.0/

#generates my_interface_wrap.cxx and mymodule.py
swig -I${EIGENROOT} -I./ifiles -c++ -python my_interface.i   


#compiles .cpp and .cxx to .o
mpiCC -O2 -std=c++11 -fPIC -c ./cpp_functions.cpp \
  -I /home/beto/docs/macbook/projects/learning/libraries/ \
  -I ${MKLROOT}/include/ \
  -I ${EIGENROOT}


VIRTENV=/home/beto/local_data/env_beaver/
mpiCC -std=c++11 -fPIC -c ./my_interface_wrap.cxx \
  -I /usr/include/python3.12/  \
  -I ${VIRTENV}/lib/python3.12/site-packages/numpy/_core/include/  \
  -I ${VIRTENV}/lib/python3.12/site-packages/mpi4py/include/ \
  -I ${EIGENROOT} \
  -I ./library/ #ParallelLinearAlgebra.hpp
 
#build shared library. Name from 'mymodule.py' with '_' and '.so'
mpiCC -fPIC -shared -m64 \
  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi_lp64 \
  -lpthread -lm -ldl \
  -L ./library/ -lplinalg \
  cpp_functions.o my_interface_wrap.o -o _mymodule.so \
  -Wl,-rpath,./library/ \
  -Wl,-rpath,${MKLROOT}/lib

