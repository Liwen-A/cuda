#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "benchmark.cuh"
#include "gtest/gtest.h"
#include "util.cuh"

constexpr int MAX_STRIDE = 32;
constexpr int X_FILL = 0;
constexpr int Y_FILL = 1;
constexpr int Z_FILL = -1;

void checkErrors(int *z, unsigned int stride, unsigned int N) {
  for (size_t i = 0; i < MAX_STRIDE * N; ++i) {
    if (i % stride == 0 && i < stride * N) {
      EXPECT_EQ(z[i], 0x01010101)
          << "Mismatch with stride " << stride << ". " << std::endl
          << "z[" << i << "] != x[" << i << "] + "
          << "y[" << i << "]" << std::endl;
    } else {
      EXPECT_EQ(z[i], Z_FILL)
          << "Mismatch with stride " << stride << ". " << std::endl
          << "z[" << i << "] != x[" << i << "] + "
          << "y[" << i << "]" << std::endl;
    }
  }
}

TEST(testQ3, StrideTest) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) FAIL() << "Failed to get CUDA device name";
  std::cout << "# Using device: " << prop.name << std::endl;

  std::size_t N = 10000000;

  const int nbytes_data = sizeof(int) * MAX_STRIDE * N;

  // Allocate GPU memory
  int *x, *y, *z;
  err = cudaMalloc(&x, nbytes_data);
  if (err != cudaSuccess) FAIL() << "Failed to allocate CUDA memory for x";
  err = cudaMalloc(&y, nbytes_data);
  if (err != cudaSuccess) FAIL() << "Failed to allocate CUDA memory for y";
  err = cudaMalloc(&z, nbytes_data);
  if (err != cudaSuccess) FAIL() << "Failed to allocate CUDA memory for z";

  // Allocate CPU memory
  int *host_z = new int[MAX_STRIDE * N];

  // Warmup calculation
  elementwise_add<<<72, 1024>>>(x, y, z, static_cast<unsigned int>(1),
                                static_cast<unsigned int>(N));
  check_launch("warm up");

  // Benchmark runs
  const int n_repeat = 5;
  printf("# stride     time [ms]   GB/sec\n");
  for (int stride = 1; stride <= MAX_STRIDE; ++stride) {
    // Testing implementation
    cudaMemset(x, X_FILL, nbytes_data);
    cudaMemset(y, Y_FILL, nbytes_data);
    cudaMemset(z, Z_FILL, nbytes_data);

    elementwise_add<<<72, 1024>>>(x, y, z, static_cast<unsigned int>(stride),
                                  static_cast<unsigned int>(N));
    check_launch("testing");

    cudaMemcpy(host_z, z, nbytes_data, cudaMemcpyDeviceToHost);
    checkErrors(host_z, stride, N);

    // Benchmark
    event_pair timer;

    start_timer(&timer);
    // repeat calculation several times, then average
    for (int num_runs = 0; num_runs < n_repeat; ++num_runs) {
      elementwise_add<<<72, 1024>>>(x, y, z, static_cast<unsigned int>(stride),
                                    static_cast<unsigned int>(N));
    }
    double exec_time = stop_timer(&timer);

    check_launch("elementwise_add");

    printf("   %5d    %8.4f   %7.1f\n", stride, exec_time,
           n_repeat * 3.0 * sizeof(int) * N / exec_time * 1e-6);
  }
}
