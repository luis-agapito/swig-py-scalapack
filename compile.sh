rm -f *.o
rm -f *.so
rm -f *.cxx

MKLROOT=/home/pop/local_data/libraries/intel_mkl/mkl/latest/
EIGENROOT=/home/pop/local_data/libraries/eigen-3.4.0/
MPICC=/home/pop/local_data/libraries/mpich-4.3.0/bin/mpicxx

echo "swig ---------------------------------"
#generates my_interface_wrap.cxx and mymodule.py
swig -I${EIGENROOT} -I./ifiles -c++ -python my_interface.i

echo "MPICC1---------------------------------"
#compiles .cpp and .cxx to .o
${MPICC} -O2 -std=c++11 -fPIC -c ./cpp_functions.cpp \
  -I ./library \
  -I ${MKLROOT}/include/ \
  -I ${EIGENROOT}

echo "MPICC2--------------------------------"
VIRTENV=/home/pop/local_data/envs/env_beaver/ #A Python env., with numpy and mpi4py installed.
${MPICC} -std=c++11 -fPIC -c ./my_interface_wrap.cxx \
  -I /usr/include/python3.10/ \
  -I ${VIRTENV}/lib/python3.10/site-packages/numpy/_core/include/ \
  -I ${VIRTENV}/lib/python3.10/site-packages/mpi4py/include/ \
  -I ${EIGENROOT} \
  -I ${MKLROOT}/include \
  -I ./library/ #ParallelLinearAlgebra.hpp

echo "MPICC3--------------------------------"
#build shared library. Name from 'mymodule.py' with '_' and '.so'
${MPICC} -fPIC -shared -m64 \
  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 \
  -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 \
  -lpthread -lm -ldl \
  -L ./library/ -lplinalg \
  cpp_functions.o my_interface_wrap.o -o _mymodule.so \
  -Wl,-rpath,./library/ \
  -Wl,-rpath,${MKLROOT}/lib
