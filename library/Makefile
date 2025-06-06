CXX = /home/pop/local_data/libraries/mpich-4.3.0/bin/mpicxx
CXXFLAGS = -std=c++11 -pedantic -g -m64 -fPIC #-Wall -Werror

MKLROOT=/home/pop/local_data/libraries/intel_mkl/mkl/latest/
EIGENROOT=/home/pop/local_data/libraries/eigen-3.4.0/

##compiler options
#mpiCC -std=c++11 -c -DMKL_ILP64  -m64  -I"${MKLROOT}/include" lapacke_dgeev.cpp 
##Linking line:
#mpiCC -std=c++11 -m64  -L${MKLROOT}/lib -lmkl_scalapack_ilp64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64 -lgomp -lpthread -lm -ldl lapacke_dgeev.o -o lapacke_dgeev.bin 
#without cluster libraries:
#-m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldlc
INCLUDES = -I$(MKLROOT)/include/ -I$(EIGENROOT)
#INCLUDES = -I$(MKLROOT)/include/

LDFLAGS = -L$(MKLROOT)/lib

#with cluster libraries. using mpicc
#LDLIBS = -lmkl_scalapack_ilp64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64 -lgomp -lpthread -lm -ldl -Wl,-rpath,$(MKLROOT)/lib

#without cluster libraries. using g++
#LDLIBS = -m64 -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -Wl,-rpath,$(MKLROOT)/lib
#with cluster, without openmpi
LDLIBS = -m64 -L${MKLROOT}/lib -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl -Wl,-rpath,$(MKLROOT)/lib
SOURCES = ParallelLinearAlgebra.cpp
OBJS = $(SOURCES:.cpp=.o)
TARGET = libplinalg.so 

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -shared $(CXXFLAGS) -o $(TARGET) $(OBJS) $(INCLUDES) $(LDFLAGS) $(LDLIBS)

run: all
	@./$(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: clean
clean:
	#rm -f $(TARGET).bin $(OBJS)
	rm -f *.so *.o
