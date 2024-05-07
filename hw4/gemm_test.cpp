#include "common.h"
#include "cublas_v2.h"
#include "gpu_func.h"
#include "util.cuh"
#include "gtest/gtest.h"
#include <armadillo>

void createMATS(nn_real *A, nn_real *B, nn_real *C, int NI, int NJ, int NK) {
  int i, j;

  for (j = 0; j < NK; j++) {
    for (i = 0; i < NI; i++) {
      A[i + j * NI] = (i * j) % 17 - 8.;
    }
  }

  for (j = 0; j < NJ; j++) {
    for (i = 0; i < NK; i++) {
      B[i + j * NK] = (i * (1 + j)) % 17 - 8.;
    }
  }

  for (j = 0; j < NJ; j++) {
    for (i = 0; i < NI; i++) {
      C[i + j * NI] = ((2 + i) * j) % 17 - 8.;
    }
  }
}

void compareGEMMResults(arma::Mat<nn_real> A, arma::Mat<nn_real> B) {
  nn_real reldiff = arma::norm(A - B, "inf") / arma::norm(B, "inf");
  // Print results
  std::cout << "Relative Inf error = " << reldiff << std::endl;

  ASSERT_EQ(reldiff, 0.);
}

void TestGEMM(int M, int N, int K,
              void (*myGEMM)(DeviceMatrix, DeviceMatrix, DeviceMatrix, nn_real,
                             nn_real)) {
  arma::Mat<nn_real> A(M, K);
  arma::Mat<nn_real> B(K, N);
  arma::Mat<nn_real> C(M, N);

  createMATS(A.memptr(), B.memptr(), C.memptr(), M, N, K);

  arma::Mat<nn_real> C_cublas(C);
  arma::Mat<nn_real> C_myGEMM(C);

  DeviceMatrix dA(A);
  DeviceMatrix dB(B);
  DeviceMatrix dummy(C);
  DeviceMatrix dC_myGEMM(C_myGEMM);

  nn_real alpha = 2.0;
  nn_real beta = 5.0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);

  /* Warm up GPU before we run. We run one extra CuBlas */
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS initialization failed!" << std::endl;
    return;
  }
  stat = cublas_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                     dA.memptr(), M, dB.memptr(), K, &beta, dummy.memptr(), M);

  /* Compute reference solution and time cuBLAS */
  int repeats = 10;
  auto total_time_cublas = 0.0;
  float milliseconds = 0;
  for (int i = 0; i < repeats; i++) {
    DeviceMatrix dC_cublas(C);

    cudaEventRecord(start);
    stat = cublas_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                       dA.memptr(), M, dB.memptr(), K, &beta,
                       dC_cublas.memptr(), M);
    cudaEventRecord(stop);
    check_launch("Reference GEMM");

    cudaEventElapsedTime(&milliseconds, start, stop);

    // std::cout << "Trial " << i << " CUBLAS gemm time: " << milliseconds /
    // 1000
    //           << " seconds" << std::endl;
    total_time_cublas += milliseconds / 1000;

    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "CUBLAS gemm error at " << __FILE__ << ":" << __LINE__
                << std::endl;
    }

    dC_cublas.to_cpu(C_cublas);
  }

  /* We are calling your GEMM function here */
  /* We will make one dummy call and check_launch here */
  myGEMM(dA, dB, dummy, alpha, beta);
  check_launch("myGEMM dummy");

  auto total_time_myGEMM = 0.0;
  for (int i = 0; i < repeats; i++) {
    DeviceMatrix dC_myGEMM(C);

    cudaEventRecord(start);
    myGEMM(dA, dB, dC_myGEMM, alpha, beta);
    cudaEventRecord(stop);

    check_launch("myGEMM");

    cudaEventElapsedTime(&milliseconds, start, stop);

    // std::cout << "Trial " << i << " myGemm time: " << milliseconds / 1000
    //           << " seconds" << std::endl;
    total_time_myGEMM += milliseconds / 1000;

    dC_myGEMM.to_cpu(C_myGEMM);
  }

  compareGEMMResults(C_myGEMM, C_cublas);

  auto avg_time_cublas = total_time_cublas / repeats;
  std::cout << "Average of " << repeats << " runs for cublas:\t"
            << avg_time_cublas << " seconds" << std::endl;

  auto avg_time_myGEMM = total_time_myGEMM / repeats;
  std::cout << "Average of " << repeats << " runs for myGEMM:\t"
            << avg_time_myGEMM << " seconds" << std::endl;

  std::cout << "Reference GEMM is\t\t" << avg_time_myGEMM / avg_time_cublas
            << " times faster" << std::endl;
}

void BenchmarkGEMM(int M, int N, int K,
                   void (*myGEMM)(DeviceMatrix, DeviceMatrix, DeviceMatrix,
                                  nn_real, nn_real)) {
  std::cout << std::endl
            << "GEMM: "
            << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  TestGEMM(M, N, K, myGEMM);
  return;
}
