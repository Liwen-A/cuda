#ifndef COMMON_H_
#define COMMON_H_

#include <cassert>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

using std::string;
using std::to_string;

#ifndef USE_DOUBLE

#define TOL 1e-6 // Tolerance for tests

typedef float nn_real;
#define MPI_FP MPI_FLOAT
#define cublas_gemm cublasSgemm
#define Log(value) (logf(value))
#define Exp(value) (expf(value))

#else

#define TOL 1e-14 // Tolerance for tests

typedef double nn_real;
#define MPI_FP MPI_DOUBLE
#define cublas_gemm cublasDgemm
#define Log(value) (log(value))
#define Exp(value) (exp(value))

#endif

#define CUDA_DEBUG

#ifdef CUDA_DEBUG
#define CHECK_LAUNCH(msg) getLastCudaError(msg)
#else
#define CHECK_LAUNCH(msg)
#endif

#endif // COMMON_H_