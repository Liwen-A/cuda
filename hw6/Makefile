# Add the following command to your .bashrc:
# ml course/cme213/nvhpc/24.1

# Compiler settings
CXX=nvc++
# If needed you can set CXXFLAGS from the command line as such:
# $ CXXFLAGS=-DNDEBUG make -j
# $ CXXFLAGS=-DNDEBUG srun -p gpu-turing make -j
#
# You may set DOUBLE_FLAG to -DUSE_DOUBLE
# Example:
# $ DOUBLE_FLAG=-DUSE_DOUBLE make -j
# $ DOUBLE_FLAG=-DUSE_DOUBLE srun -p gpu-turing make -j
CXXFLAGS+=-O2 -tp=px -gpu=cuda12.3,cc75,sm_75 -cuda --diag_suppress unrecognized_attribute $(DOUBLE_FLAG)
LDFLAGS=-gpu=cuda12.3,cc75,sm_75 -cuda -lblas -lcublas -L$(NVHPC_ROOT)/math_libs/12.3/targets/x86_64-linux/lib

# Include directories
ARMADILLO_INC=./armadillo-12.8.2/include
CUDA_HELPER_INC=./cuda_helper
MPI_INC=$(NVHPC_ROOT)/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi/include
INCFLAGS=-I$(ARMADILLO_INC) -I$(CUDA_HELPER_INC) -I$(MPI_INC)

# Google Test
# Root directory
GTEST_ROOT=./googletest-main
GTEST_DIR=$(GTEST_ROOT)/googletest
GTEST_INC=$(GTEST_DIR)/include
GTEST_SRC=$(GTEST_DIR)/src

GTEST_FLAGS = -isystem $(GTEST_INC) -O2
GTEST_HEADERS = $(GTEST_INC)/gtest/*.h \
                $(GTEST_INC)/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_SRC)/*.cc $(GTEST_SRC)/*.h $(GTEST_HEADERS)

CPPFLAGS=-isystem $(GTEST_INC)

default: main_q1 main_q2

gtest: gtest.a gtest_main.a
gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $(GTEST_DIR)/src/gtest-all.cc
gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $(GTEST_DIR)/src/gtest_main.cc
gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^
gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

gpu_func.o: gpu_func.cu gpu_func.h common.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c gpu_func.cu

neural_network.o: neural_network.cpp neural_network.h common.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c neural_network.cpp

mnist.o: mnist.cpp mnist.h common.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c mnist.cpp

SRC1=main_q1.cpp
HDR1=gpu_func.h neural_network.h mnist.h common.h
main_q1.o : $(SRC1) $(HDR1) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c $(SRC1)
main_q1: main_q1.o gpu_func.o neural_network.o mnist.o gtest_main.a
	mpic++ $^ -o $@ $(LDFLAGS)

SRC2=main_q2.cpp
HDR2=gpu_func.h neural_network.h mnist.h common.h
main_q2.o : $(SRC2) $(HDR2) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c $(SRC2)
main_q2: main_q2.o gpu_func.o neural_network.o mnist.o gtest_main.a
	mpic++ $^ -o $@ $(LDFLAGS)

clean:
	rm -f main_q1 main_q2 *.o *.a
