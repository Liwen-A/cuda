# Add the following command to your .bashrc:
# ml course/cme213/nvhpc/24.1

# Root directories
GTEST_ROOT=./googletest-main
HPC_SDK_ROOT=/home/cme213/software/nvidia-hpc-sdk/2024_24.1/Linux_x86_64/24.1

# Google Test
GTEST_DIR=$(GTEST_ROOT)/googletest
GTEST_INC=$(GTEST_DIR)/include
GTEST_SRC=$(GTEST_DIR)/src

GTEST_FLAGS = -isystem $(GTEST_INC) -O2
GTEST_HEADERS = $(GTEST_INC)/gtest/*.h \
                $(GTEST_INC)/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_SRC)/*.cc $(GTEST_SRC)/*.h $(GTEST_HEADERS)

# Compiler settings
CXX = nvcc
CPPFLAGS += -isystem $(GTEST_INC)
CXXFLAGS=-O2 -std=c++14 -arch=compute_75 -code=sm_75

# Include directories
CUDA_HELPER_INC=./cuda_helper
ARMADILLO_INC=./armadillo-12.8.2/include
CUDA_INC=-I$(HPC_SDK_ROOT)/cuda/12.3/targets/x86_64-linux/include
INCFLAGS=-I$(ARMADILLO_INC) -I$(CUDA_INC) -I$(CUDA_HELPER_INC)

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
HDR1=recurrence.cuh test_recurrence.h
main_q1.o : $(SRC1) $(HDR1) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC1)
main_q1: main_q1.o gtest_main.a 
	$(CXX) $^ -o $@ $(LDFLAGS)

SRC2=main_q2.cu
HDR2=util.cuh benchmark.cuh
main_q2.o : $(SRC2) $(HDR2) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC2)
main_q2: main_q2.o gtest_main.a
	$(CXX) $^ -o $@ $(LDFLAGS)

SRC3=main_q3.cpp
CUDA_SRC3=gpu_func.cu
HDR3=util.cuh gpu_func.h
main_q3.o : $(SRC3) $(HDR3) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c $(SRC3)
gpu_func.o: $(CUDA_SRC3) $(HDR3)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) -c $(CUDA_SRC3)
main_q3: main_q3.o gpu_func.o gtest_main.a
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm -f *.o main_q1 main_q2 main_q3 gtest-all.o gtest.a gtest_main.a gtest_main.o