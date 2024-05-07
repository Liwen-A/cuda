#include "common.h"
#include "cublas_v2.h"
#include "gemm_test.h"
#include "gpu_func.h"
#include "util.cuh"
#include <armadillo>
#include <chrono>

TEST(TiledGEMMTest, deviceWarmup) {
  DWarmup(); // For accurate test timing
}

TEST(TiledGEMMTest, small) {
  int M = 37, N = 41, K = 43;
  BenchmarkGEMM(M, N, K, tiledGEMM);
}

TEST(TiledGEMMTest, medium) {
  int M = 599, N = 433, K = 751;
  BenchmarkGEMM(M, N, K, tiledGEMM);
}

TEST(TiledGEMMTest, large) {
  int M = 7817, N = 7919, K = 6869;
  BenchmarkGEMM(M, N, K, tiledGEMM);
}
