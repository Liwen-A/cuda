#include "common.h"
#include "cublas_v2.h"
#include "gemm_test.h"
#include "gpu_func.h"
#include "util.cuh"
#include <armadillo>
#include <chrono>

TEST(SharedGEMMTest, deviceWarmup) {
  DWarmup(); // For accurate test timing
}

TEST(SharedGEMMTest, small) {
  int M = 37, N = 41, K = 43;
  BenchmarkGEMM(M, N, K, sharedMemoryGEMM);
}

TEST(SharedGEMMTest, medium) {
  int M = 599, N = 433, K = 751;
  BenchmarkGEMM(M, N, K, sharedMemoryGEMM);
}

TEST(SharedGEMMTest, large) {
  int M = 7817, N = 7919, K = 6869;
  BenchmarkGEMM(M, N, K, sharedMemoryGEMM);
}
