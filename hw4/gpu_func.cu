#define ARMA_ALLOW_FAKE_GCC
#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <memory>

#include "gpu_func.h"
#include "util.cuh"

__global__ void Warmup() {}

void DWarmup() { Warmup<<<1, 1>>>(); }

/**
 * DeviceAllocator and DeviceMatrix
 */

DeviceAllocator::DeviceAllocator(nn_real *cpu_data, int n) {
  assert(n >= 0);
  assert(cpu_data != nullptr);
  nbytes = n * sizeof(nn_real);
  cudaMalloc(&data, nbytes);
  cudaMemcpy(data, cpu_data, nbytes, cudaMemcpyHostToDevice);
}

DeviceAllocator::DeviceAllocator(int n) {
  assert(n >= 0);
  nbytes = n * sizeof(nn_real);
  cudaMalloc(&data, nbytes);
}

DeviceAllocator::~DeviceAllocator() {
  if (data != nullptr)
    cudaFree(data);
}

int DeviceAllocator::total_bytes() { return nbytes; }

nn_real *DeviceAllocator::memptr() { return data; }

void DeviceAllocator::to_cpu(nn_real *cpu_data) {
  assert(data != nullptr && cpu_data != nullptr);
  cudaMemcpy(cpu_data, data, nbytes, cudaMemcpyDeviceToHost);
}

DeviceMatrix::DeviceMatrix(int n_rows, int n_cols) {
  assert(n_rows >= 0 && n_cols >= 0);
  this->allocator = std::make_shared<DeviceAllocator>(n_rows * n_cols);
  this->data = this->allocator->memptr();
  this->n_rows = n_rows;
  this->n_cols = n_cols;
}

DeviceMatrix::DeviceMatrix(arma::Mat<nn_real> &cpu_mat) {
  this->allocator = std::make_shared<DeviceAllocator>(
      cpu_mat.memptr(), cpu_mat.n_rows * cpu_mat.n_cols);
  this->data = this->allocator->memptr();
  this->n_rows = cpu_mat.n_rows;
  this->n_cols = cpu_mat.n_cols;
}

int DeviceMatrix::total_bytes() { return allocator->total_bytes(); }

nn_real *DeviceMatrix::memptr() { return data; }

void DeviceMatrix::to_cpu(arma::Mat<nn_real> &cpu_mat) {
  allocator->to_cpu(cpu_mat.memptr());
}

__device__ nn_real &DeviceMatrix::operator()(int row, int col, bool transpose) {
  assert(data != nullptr && row >= 0 && row < n_rows && col >= 0 &&
         col < n_cols);
  return transpose ? data[row * n_cols + col] : data[col * n_rows + row];
}
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//                           GEMM kernels                           //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

__global__ void BasicMatMulColumnMajor(DeviceMatrix A, DeviceMatrix B,
                                       DeviceMatrix C, nn_real alpha,
                                       nn_real beta) {
  // TODO: Implement this kernel
  int id = blockDim.x * blockIdx.x + threadIdx.x; 
  if (id < C.n_cols * C.n_rows){
    int i = id % C.n_rows;
    int j = id / C.n_rows;
    float cum_sum = 0.0;
    for (int k = 0; k < A.n_cols; k++){
      cum_sum += A(i,k) * B(k,j) ;
    }
    C(i,j) = alpha * cum_sum + beta * C(i,j);
  }
}

void basicGEMMColumnMajor(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                          nn_real alpha, nn_real beta) {
  // TODO: Implement this kernel wrapper
  // Remember that column major means that consecutive threads compute
  // consecutive elements in a column of the output matrix
  int numThread = 32*32;
  int numBlock = C.n_cols * C.n_rows / numThread+1;
  dim3 blockPerGrid(numBlock);
  dim3 threadPerBlock(numThread);
  BasicMatMulColumnMajor<<<blockPerGrid,threadPerBlock>>>(A,B,C,alpha,beta);
  check_launch("basicGEMMColumnMajor");
}

__global__ void BasicMatMulRowMajor(DeviceMatrix A, DeviceMatrix B,
                                    DeviceMatrix C, nn_real alpha,
                                    nn_real beta) {
  // TODO: Implement this kernel
  int id = blockDim.x * blockIdx.x + threadIdx.x; 
  if (id < C.n_cols * C.n_rows){
    int i = id / C.n_cols;
    int j = id % C.n_cols;
    float cum_sum = 0.0;
    for (int k = 0; k < A.n_cols; k++){
      cum_sum += A(i,k) * B(k,j) ;
    }
    C(i,j) = alpha * cum_sum + beta * C(i,j);
  }
}

void basicGEMMRowMajor(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                       nn_real alpha, nn_real beta) {
  // TODO: Implement this kernel wrapper
  // Remember that row major means that consecutive threads compute
  // consecutive elements in a row of the output matrix
  int numThread = 32*32;
  int numBlock = C.n_cols * C.n_rows / numThread+1;
  dim3 blockPerGrid(numBlock);
  dim3 threadPerBlock(numThread);
  BasicMatMulColumnMajor<<<blockPerGrid,threadPerBlock>>>(A,B,C,alpha,beta);
  check_launch("basicGEMMRowMajor");
}

template <int blockSizeX, int blockSizeY>
__global__ void SharedMemoryMatMul(DeviceMatrix A, DeviceMatrix B,
                                   DeviceMatrix C, nn_real alpha,
                                   nn_real beta) {

  // TODO: Implement this kernel
    float CValue = 0;

    int Row = blockIdx.y*blockSizeY + threadIdx.y;
    int Col = blockIdx.x*blockSizeX+ threadIdx.x;

    __shared__ float As[blockSizeX][blockSizeY];
    __shared__ float Bs[blockSizeX][blockSizeY];
    for (int k = 0; k < (blockSizeX + A.n_cols - 1)/blockSizeX; k++) {

      if (k*blockSizeX + threadIdx.x < A.n_cols && Row < A.n_rows)
          As[threadIdx.y][threadIdx.x] = A(Row , k*blockSizeX+threadIdx.x); 
      else
          As[threadIdx.y][threadIdx.x] = 0.0;

      if (k*blockSizeY + threadIdx.y < B.n_rows && Col < B.n_cols)
          Bs[threadIdx.y][threadIdx.x] =  B(k*blockSizeY+threadIdx.y, Col);
      else
          Bs[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();

      for (int n = 0; n < blockSizeY; ++n)
          CValue += alpha*As[threadIdx.y][n] * Bs[n][threadIdx.x];

      __syncthreads();
    }

    if (Row < C.n_rows && Col < C.n_cols){
      C(Row,Col) = CValue + beta * C(Row,Col);
    }
}

void sharedMemoryGEMM(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                      nn_real alpha, nn_real beta) {
  // TODO: Implement this wrapper
  int blockX = C.n_cols/32+1;
  int blockY = C.n_rows/32+1;
  dim3 blockPerGrid(blockX,blockY);
  dim3 threadPerBlock(32,32);
  SharedMemoryMatMul<32,32><<<blockPerGrid,threadPerBlock>>>(A,B,C,alpha,beta);
  check_launch("sharedMemoryGEMM");
}

// 32x32 Hierarchical Tiling
// num_thread: number of threads per block
// blockItemsM: number of rows of A in each submatrix of A
// blockItemsN: number of columns of B in each submatrix of B
// blockItemsK: number of columns in submatrix of A and rows in submatrix of B
template <int num_thread, int blockItemsM, int blockItemsN, int blockItemsK>
__global__ void TiledMatMul(DeviceMatrix A, bool transa, DeviceMatrix B,
                            bool transb, DeviceMatrix C, nn_real alpha,
                            nn_real beta) {
  // TODO: Implement this kernel
}

// wrapper for MatMulTile_32_32
void tiledGEMM(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C, nn_real alpha,
               nn_real beta) {
  assert((A.n_cols) == (B.n_rows));
  assert(C.n_rows == (A.n_rows) && C.n_cols == (B.n_cols));

  constexpr int block_m = 32;
  constexpr int block_n = 32;
  constexpr int block_k = 32;
  constexpr int num_thread = 128;
  dim3 grid((C.n_rows + block_m - 1) / block_m,
            (C.n_cols + block_n - 1) / block_n);
  TiledMatMul<num_thread, block_m, block_n, block_k>
      <<<grid, num_thread>>>(A, false, B, false, C, alpha, beta);

  check_launch("tiledGEMM");
}
