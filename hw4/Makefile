# Add the following command to your .bashrc:
# ml course/cme213/nvhpc/24.1

# Compiler settings
CXX=nvc++
CPPFLAGS=-isystem $(GTEST_INC)
CXXFLAGS=-O2 -tp=px -gpu=cuda12.3,cc75,sm_75 -cuda --diag_suppress unrecognized_attribute
LDFLAGS=-gpu=cuda12.3,cc75,sm_75 -cuda -lblas -L$(NVHPC_ROOT)/math_libs/12.3/targets/x86_64-linux/lib -lcublas

# Include directories
CUDA_HELPER_INC=./cuda_helper
ARMADILLO_INC=./armadillo-12.8.2/include
INCFLAGS=-I$(ARMADILLO_INC) -I$(CUDA_HELPER_INC)

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

default: main_q1 main_q2 main_q3

gtest: gtest.a gtest_main.a
gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $(GTEST_DIR)/src/gtest-all.cc
gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $(GTEST_DIR)/src/gtest_main.cc
gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^
gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

SRC1=main_q1.cu
CUDA_SRC=gpu_func.cu 
HDR1=util.cuh gpu_func.h

main_q1.o : $(SRC1) $(HDR1) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c $(SRC1)
gpu_func.o: $(CUDA_SRC) $(HDR1)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c $(CUDA_SRC)
gemm_test.o: gemm_test.cpp gemm_test.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c gemm_test.cpp
main_q1: main_q1.o gpu_func.o gtest_main.a gemm_test.o
	$(CXX) $^ -o $@ $(LDFLAGS)

SRC2=main_q2.cu
HDR2=util.cuh gpu_func.h

main_q2.o : $(SRC2) $(HDR2) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c $(SRC2)
main_q2: main_q2.o gpu_func.o gtest_main.a gemm_test.o
	$(CXX) $^ -o $@ $(LDFLAGS)

SRC3=main_q3.cu
HDR3=util.cuh gpu_func.h

main_q3.o : $(SRC3) $(HDR3) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c $(SRC3)
main_q3: main_q3.o gpu_func.o gtest_main.a gemm_test.o
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm -f main_q1 main_q2 main_q3 *.o *.a
