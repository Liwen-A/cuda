#include "common.h"
#include "cublas_v2.h"
#include "gemm_test.h"
#include "gpu_func.h"
#include "util.cuh"
#include <armadillo>
#include <chrono>

TEST(testWarmup, deviceWarmup) {
  DWarmup(); // For accurate test timing
}

TEST(basicGEMMRowMajor, small) {
  std::cout << "Testing row major basicGEMM" << std::endl;
  int M = 37, N = 41, K = 43;
  BenchmarkGEMM(M, N, K, basicGEMMRowMajor);
}

TEST(basicGEMMRowMajor, medium) {
  std::cout << "Testing row major basicGEMM" << std::endl;
  int M = 599, N = 433, K = 751;
  BenchmarkGEMM(M, N, K, basicGEMMRowMajor);
}

TEST(basicGEMMRowMajor, large) {
  std::cout << "Testing row major basicGEMM" << std::endl;
  int M = 7817, N = 7919, K = 6869;
  BenchmarkGEMM(M, N, K, basicGEMMRowMajor);
}

TEST(basicGEMMColumnMajor, small) {
  std::cout << "Testing column major basicGEMM" << std::endl;
  int M = 37, N = 41, K = 43;
  BenchmarkGEMM(M, N, K, basicGEMMColumnMajor);
}

TEST(basicGEMMColumnMajor, medium) {
  std::cout << "Testing column major basicGEMM" << std::endl;
  int M = 599, N = 433, K = 751;
  BenchmarkGEMM(M, N, K, basicGEMMColumnMajor);
}

TEST(basicGEMMColumnMajor, large) {
  std::cout << "Testing column major basicGEMM" << std::endl;
  int M = 7817, N = 7919, K = 6869;
  BenchmarkGEMM(M, N, K, basicGEMMColumnMajor);
}
