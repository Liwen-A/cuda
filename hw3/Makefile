# Add the following command to your .bashrc:
# ml course/cme213/nvhpc/24.1

# nvc++ option
CXX=nvc++
CXXFLAGS=-O2 -tp=px -gpu=cuda12.3,cc75,sm_75 -cuda
# nvc++ linking; optional
# LDFLAGS=-cuda

# Choose your option
# fma = fused multiply–add optimization
# -gpu=nofma will mean lower performance 
# but the results will match the CPU results.
# No option (empty) means that the code runs faster 
# but there will be roundoff errors compared to the CPU.
# Make sure to run:
# $ make clean && make
# when changing this option so all the input files recompile correctly.

# FMAD_FLAG=-gpu=fma
FMAD_FLAG=-gpu=nofma

# nvcc option 
# CXX=nvcc
# CXXFLAGS=-O2 -arch=compute_75 -code=sm_75
# FMAD_FLAG=--fmad=false
# FMAD_FLAG=--fmad=true

# Compiler settings
CUDFLAGS=$(FMAD_FLAG) $(CXXFLAGS)

# Root directories
GTEST_ROOT=./googletest-main

# Google Test
GTEST_DIR=$(GTEST_ROOT)/googletest
GTEST_INC=$(GTEST_DIR)/include
GTEST_SRC=$(GTEST_DIR)/src

GTEST_FLAGS = -isystem $(GTEST_INC) -O2
GTEST_HEADERS = $(GTEST_INC)/gtest/*.h \
                $(GTEST_INC)/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_SRC)/*.cc $(GTEST_SRC)/*.h $(GTEST_HEADERS)

HDR=main.cu mp1-util.h simParams.h Grid.h gpuStencil.cu BC.h \
	CPUComputation.h Errors.h

CPPFLAGS += -isystem $(GTEST_INC)

GTEST_HEADERS = $(GTEST_INC)/gtest/*.h \
                $(GTEST_INC)/gtest/internal/*.h

GTEST_SRCS_ = $(GTEST_SRC)/*.cc $(GTEST_SRC)/*.h $(GTEST_HEADERS)

.PHONY: default clean

default: main

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_SRC)/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_SRC)/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

main: main.o simParams.o Grid.o gtest_main.a
	$(CXX) $^ $(CUDFLAGS) $(CPPFLAGS) -o $@ $(LDFLAGS)

main.o: $(HDR) $(GTEST_HEADERS)
	$(CXX) -c $< $(CUDFLAGS) $(CPPFLAGS)

simParams.o: simParams.cpp simParams.h
	$(CXX) -c $< $(CXXFLAGS)

Grid.o: Grid.cu Grid.h
	$(CXX) -c $< $(CUDFLAGS)

clean:
	rm -f *.o *.a *~ *~ *Errors.txt main