#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <cuda_runtime.h>

#include "common.h"
#include <memory>

__global__ void Warmup();

void DWarmup();

/**
 * A class that manages memory for the GPU data.
 */
class DeviceAllocator
{
private:
  nn_real *data = nullptr; // pointer to the GPU data
  int nbytes = 0;          // number of bytes that were allocated

public:
  DeviceAllocator() = default;

  DeviceAllocator(nn_real *cpu_data, int n);
  DeviceAllocator(int n);

  ~DeviceAllocator();

  int total_bytes();
  nn_real *memptr();
  void to_cpu(nn_real *cpu_data);
};

/**
 * Matrix on the GPU
 */
class DeviceMatrix
{
  /**
   * A pointer to the matrix data.
   */
private:
  std::shared_ptr<DeviceAllocator> allocator;
  nn_real *data = nullptr;

public:
  int n_rows = 0;
  int n_cols = 0;

  DeviceMatrix() = default;
  DeviceMatrix(int n_rows, int n_cols);
  DeviceMatrix(arma::Mat<nn_real> &cpu_mat);

  int total_bytes();
  nn_real *memptr();
  void to_cpu(arma::Mat<nn_real> &cpu_mat);

  /**
   * A device function that returns the element at the specified
   * row and column of the matrix.
   *
   * @param row The row index of the element.
   * @param col The column index of the element.
   * @return The element at the specified row and column of the matrix.
   */
  __device__ nn_real &operator()(int row, int col, bool transpose = false);
};

__global__ void BasicMatMulColumnMajor(DeviceMatrix A, DeviceMatrix B,
                                       DeviceMatrix C, nn_real alpha,
                                       nn_real beta);

void basicGEMMColumnMajor(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                          nn_real alpha, nn_real beta);

__global__ void BasicMatMulRowMajor(DeviceMatrix A, DeviceMatrix B,
                                    DeviceMatrix C, nn_real alpha,
                                    nn_real beta);

void basicGEMMRowMajor(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                       nn_real alpha, nn_real beta);

// Basic sharedMemoryGEMM
template <int num_thread, int blockItemsM, int blockItemsN, int blockItemsK>
__global__ void SharedMemoryMatMul(DeviceMatrix A, bool transa, DeviceMatrix B,
                                 bool transb, DeviceMatrix C, nn_real alpha,
                                 nn_real beta);

void sharedMemoryGEMM(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                      nn_real alpha, nn_real beta);

// 32x32 Hierarchical Tiling
template <int num_thread, int blockItemsM, int blockItemsN, int blockItemsK>
__global__ void TiledMatMul(DeviceMatrix A, bool transa, DeviceMatrix B,
                                 bool transb, DeviceMatrix C, nn_real alpha,
                                 nn_real beta);

void tiledGEMM(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                      nn_real alpha, nn_real beta);
#endif
