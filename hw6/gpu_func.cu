#define ARMA_ALLOW_FAKE_GCC
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <algorithm>
#include <armadillo>
#include <cassert>
#include <iostream>
#include <memory>

#include "gpu_func.h"
#include "util.cuh"

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                      DeviceAllocator and DeviceMatrix                      //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

DeviceAllocator::DeviceAllocator(int n) {
  assert(n >= 0);
  nbytes = n * sizeof(nn_real);
  checkCudaErrors(cudaMalloc(&data, nbytes));
}

DeviceAllocator::DeviceAllocator(nn_real *cpu_data, int n) {
  assert(n >= 0);
  assert(cpu_data != nullptr);
  nbytes = n * sizeof(nn_real);
  checkCudaErrors(cudaMalloc(&data, nbytes));
  checkCudaErrors(cudaMemcpy(data, cpu_data, nbytes, cudaMemcpyHostToDevice));
}

DeviceAllocator::~DeviceAllocator() {
  if (data != nullptr) checkCudaErrors(cudaFree(data));
}

int DeviceAllocator::total_bytes() { return nbytes; }

nn_real *DeviceAllocator::memptr() { return data; }

DeviceMatrix::DeviceMatrix(int n_rows, int n_cols) {
  assert(n_rows >= 0 && n_cols >= 0);
  this->allocator = std::make_shared<DeviceAllocator>(n_rows * n_cols);
  this->data = this->allocator->memptr();
  this->nbytes = this->allocator->total_bytes();
  this->n_allocated_bytes = this->allocator->total_bytes();
  this->n_rows = n_rows;
  this->n_cols = n_cols;
}

DeviceMatrix::DeviceMatrix(arma::Mat<nn_real> &cpu_mat) {
  assert(cpu_mat.memptr() != nullptr);
  this->allocator = std::make_shared<DeviceAllocator>(
      cpu_mat.memptr(), cpu_mat.n_rows * cpu_mat.n_cols);
  this->data = this->allocator->memptr();
  this->nbytes = this->allocator->total_bytes();
  this->n_allocated_bytes = this->allocator->total_bytes();
  this->n_rows = cpu_mat.n_rows;
  this->n_cols = cpu_mat.n_cols;
}

DeviceMatrix::DeviceMatrix(nn_real *gpu_mat, int n_rows, int n_cols) {
  assert(n_rows >= 0 && n_cols >= 0 && gpu_mat != nullptr);
  this->allocator = nullptr;
  this->data = gpu_mat;
  this->nbytes = n_rows * n_cols * sizeof(nn_real);
  this->n_allocated_bytes = n_rows * n_cols * sizeof(nn_real);
  this->n_rows = n_rows;
  this->n_cols = n_cols;
}

int DeviceMatrix::total_bytes() { return nbytes; }

nn_real *DeviceMatrix::memptr() { return data; }

void DeviceMatrix::to_cpu(arma::Mat<nn_real> &cpu_mat) {
  assert(data != nullptr && cpu_mat.memptr() != nullptr);
  assert(n_rows == cpu_mat.n_rows && n_cols == cpu_mat.n_cols);
  checkCudaErrors(
      cudaMemcpy(cpu_mat.memptr(), data, nbytes, cudaMemcpyDeviceToHost));
}

void DeviceMatrix::to_gpu(DeviceMatrix &gpu_mat) const {
  assert(data != nullptr && gpu_mat.data != nullptr);
  assert(n_rows == gpu_mat.n_rows && n_cols == gpu_mat.n_cols);
  checkCudaErrors(
      cudaMemcpy(gpu_mat.memptr(), data, nbytes, cudaMemcpyDeviceToDevice));
}

void DeviceMatrix::set_n_cols(int n_cols) {
  assert(data != nullptr);
  assert(n_rows * n_cols * sizeof(nn_real) <= n_allocated_bytes);
  this->n_cols = n_cols;
  this->nbytes = n_rows * n_cols * sizeof(nn_real);
}

__device__ nn_real &DeviceMatrix::operator()(int row, int col, bool trans) {
  assert(data != nullptr && row >= 0 && row < trans ? n_cols
         : n_rows && col >= 0 && col < trans        ? n_rows
                                                    : n_cols);
  return trans ? data[row * n_rows + col] : data[col * n_rows + row];
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                            Element-wise kernels                            //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

__global__ void Warmup() {}

void DWarmup() {
  Warmup<<<1, 1>>>();
  CHECK_LAUNCH("Warmup");
}

__global__ void PrintMatrix(DeviceMatrix mat, char name, bool trans) {
  printf("\nDeviceMatrix %c\n", name);
  for (int i = 0; i < (trans ? mat.n_cols : mat.n_rows); i++) {
    for (int j = 0; j < (trans ? mat.n_rows : mat.n_cols); j++) {
      printf("%e ", mat(i, j, trans));
    }
    printf("\n");
  }
}

void DevicePrintMatrix(DeviceMatrix mat, char name, bool trans) {
  PrintMatrix<<<1, 1>>>(mat, name, trans);
  CHECK_LAUNCH("DevicePrintMatrix");
}

__global__ void MatSigmoid(DeviceMatrix src, DeviceMatrix dst) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i = idx / src.n_cols;
  int j = idx % src.n_cols;
  if (idx < src.n_rows * src.n_cols) {
    dst(i, j) = 1.0f / (1.0f + Exp(-src(i, j)));
  }
}

void DSigmoid(DeviceMatrix src, DeviceMatrix dst) {
  assert(src.n_rows == dst.n_rows && src.n_cols == dst.n_cols);

  int block = 256;
  int grid = (src.n_rows * src.n_cols + block - 1) / block;
  MatSigmoid<<<grid, block>>>(src, dst);
  CHECK_LAUNCH("DSigmoid");
}

__global__ void MatRepeatColVec(DeviceMatrix src, DeviceMatrix dst,
                                int repeat) {
  // For column-major matrix, use x-axis for row and y-axis for col
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < src.n_rows) {
    nn_real value = src(row, 0);
    for (int i = 0; i < repeat; i++) {
      dst(row, i) = value;
    }
  }
}

void DRepeatColVec(DeviceMatrix src, DeviceMatrix dst, int repeat) {
  assert(src.n_cols == 1 && dst.n_cols == repeat);
  int block = 128;
  int grid = (src.n_rows + block - 1) / block;
  MatRepeatColVec<<<grid, block>>>(src, dst, repeat);
  CHECK_LAUNCH("DRepeatColVec");
}

__global__ void MatSum(DeviceMatrix src, DeviceMatrix dst, nn_real alpha,
                       int axis) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ((axis == 0 && idx < dst.n_cols) || (axis == 1 && idx < dst.n_rows)) {
    nn_real sum = 0;
    int len = axis == 0 ? src.n_rows : src.n_cols;
    for (int i = 0; i < len; i++) {
      int row = axis == 0 ? i : idx;
      int col = axis == 0 ? idx : i;
      sum += src(row, col);
    }
    int target_row = axis == 0 ? 0 : idx;
    int target_col = axis == 0 ? idx : 0;
    dst(target_row, target_col) = alpha * sum;
  }
}

void DSum(DeviceMatrix src, DeviceMatrix dst, nn_real alpha, int axis) {
  assert(axis == 0 || axis == 1);
  if (axis == 0) {
    assert(src.n_cols == dst.n_cols && dst.n_rows == 1);
  } else {
    assert(src.n_rows == dst.n_rows && dst.n_cols == 1);
  }
  int block = 256;
  int len = axis == 0 ? src.n_cols : src.n_rows;
  int grid = (len + block - 1) / block;
  MatSum<<<grid, block>>>(src, dst, alpha, axis);
  CHECK_LAUNCH("DSum");
}

__global__ void MatSoftmax(DeviceMatrix src, DeviceMatrix dst, int axis) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (axis == 0 && idx < src.n_cols || axis == 1 && idx < src.n_rows) {
    nn_real exp_sum = 0;
    int len = axis == 0 ? src.n_rows : src.n_cols;
    for (int i = 0; i < len; i++) {
      int row = axis == 0 ? i : idx;
      int col = axis == 0 ? idx : i;
      exp_sum += Exp(src(row, col));
    }
    for (int i = 0; i < len; i++) {
      int row = axis == 0 ? i : idx;
      int col = axis == 0 ? idx : i;
      dst(row, col) = Exp(src(row, col)) / exp_sum;
    }
  }
}

void DSoftmax(DeviceMatrix src, DeviceMatrix dst, int axis) {
  assert(src.n_rows == dst.n_rows && src.n_cols == dst.n_cols);
  assert(axis == 0 || axis == 1);

  int block = 256;
  int len = axis == 0 ? src.n_cols : src.n_rows;
  int grid = (len + block - 1) / block;
  MatSoftmax<<<grid, block>>>(src, dst, axis);
  CHECK_LAUNCH("DSoftmax");
}

__global__ void MatCrossEntropyLoss(DeviceMatrix y_pred, DeviceMatrix y,
                                    DeviceMatrix loss) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < loss.n_rows && col < loss.n_cols) {
    loss(row, col) = -y(row, col) * Log(y_pred(row, col));
  }
}

void DCELoss(DeviceMatrix y_pred, DeviceMatrix y, DeviceMatrix loss) {
  assert(y.n_rows == y_pred.n_rows && y.n_cols == y_pred.n_cols);

  constexpr int block_size = 32;
  dim3 block(block_size, block_size);
  dim3 grid((y.n_rows + block_size - 1) / block_size,
            (y.n_cols + block_size - 1) / block_size);
  DeviceMatrix loss_mat(y.n_rows, y.n_cols);
  MatCrossEntropyLoss<<<grid, block>>>(y_pred, y, loss_mat);

  DeviceMatrix loss_vec(y.n_rows, 1);
  DSum(loss_mat, loss_vec, static_cast<nn_real>(1), 1);
  DSum(loss_vec, loss, static_cast<nn_real>(1), 0);
  CHECK_LAUNCH("DCELoss");
}

__global__ void MatElemArith(DeviceMatrix A, DeviceMatrix B, nn_real alpha,
                             nn_real beta) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i = idx / A.n_cols;
  int j = idx % A.n_cols;
  if (idx < A.n_rows * A.n_cols) A(i, j) = alpha * (A(i, j) + beta * B(i, j));
}

void DElemArith(DeviceMatrix A, DeviceMatrix B, nn_real alpha, nn_real beta) {
  assert(A.n_rows == B.n_rows && A.n_cols == B.n_cols);
  int block = 256;
  int grid = (A.n_rows * A.n_cols + block - 1) / block;
  MatElemArith<<<grid, block>>>(A, B, alpha, beta);
  CHECK_LAUNCH("DElemArith");
}

__global__ void MatSquare(DeviceMatrix src, DeviceMatrix dst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / src.n_cols;
  int j = idx % src.n_cols;
  if (idx < src.n_rows * src.n_cols) {
    nn_real value = src(i, j);
    dst(i, j) = value * value;
  }
}

void DSquare(DeviceMatrix src, DeviceMatrix dst) {
  assert(src.n_rows == dst.n_rows && src.n_cols == dst.n_cols);
  int block = 256;
  int grid = (src.n_rows * src.n_cols + block - 1) / block;
  MatSquare<<<grid, block>>>(src, dst);
  CHECK_LAUNCH("DSquare");
}

__global__ void MatSigmoidBackProp(DeviceMatrix da1, DeviceMatrix a1,
                                   DeviceMatrix dz1) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / da1.n_cols;
  int j = idx % da1.n_cols;
  if (idx < dz1.n_rows * dz1.n_cols) {
    dz1(i, j) = da1(i, j) * a1(i, j) * (1 - a1(i, j));
  }
}

void DSigmoidBackprop(DeviceMatrix da1, DeviceMatrix a1, DeviceMatrix dz1) {
  assert(da1.n_rows == a1.n_rows && da1.n_cols == a1.n_cols);
  assert(da1.n_rows == dz1.n_rows && da1.n_cols == dz1.n_cols);

  int block = 256;
  int grid = (da1.n_rows * da1.n_cols + block - 1) / block;
  MatSigmoidBackProp<<<grid, block>>>(da1, a1, dz1);
  CHECK_LAUNCH("DSigmoidBackprop");
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                                GEMM kernels                                //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

__global__ void BasicMatMulColumnMajor(DeviceMatrix A, DeviceMatrix B,
                                       DeviceMatrix C, nn_real alpha,
                                       nn_real beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < C.n_rows && j < C.n_cols) {
    nn_real sum = 0;
    for (int k = 0; k < A.n_cols; k++) {
      sum += A(i, k) * B(k, j);
    }
    C(i, j) = alpha * sum + beta * C(i, j);
  }
}

void basicGEMMColumnMajor(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                          nn_real alpha, nn_real beta) {
  assert((A.n_cols) == (B.n_rows));
  assert(C.n_rows == (A.n_rows) && C.n_cols == (B.n_cols));

  dim3 threads(16, 16);
  dim3 grid((C.n_rows + threads.x - 1) / threads.x,
            (C.n_cols + threads.y - 1) / threads.y);
  BasicMatMulColumnMajor<<<grid, threads>>>(A, B, C, alpha, beta);

  CHECK_LAUNCH("basicGEMMColumnMajor");
}

__global__ void BasicMatMulRowMajor(DeviceMatrix A, DeviceMatrix B,
                                    DeviceMatrix C, nn_real alpha,
                                    nn_real beta) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < C.n_rows && j < C.n_cols) {
    nn_real sum = 0;
    for (int k = 0; k < A.n_cols; k++) {
      sum += A(i, k) * B(k, j);
    }
    C(i, j) = alpha * sum + beta * C(i, j);
  }
}

void basicGEMMRowMajor(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                       nn_real alpha, nn_real beta) {
  assert((A.n_cols) == (B.n_rows));
  assert(C.n_rows == (A.n_rows) && C.n_cols == (B.n_cols));

  dim3 threads(16, 16);
  dim3 grid((C.n_cols + threads.x - 1) / threads.x,
            (C.n_rows + threads.y - 1) / threads.y);
  BasicMatMulRowMajor<<<grid, threads>>>(A, B, C, alpha, beta);

  CHECK_LAUNCH("basicGEMMRowMajor");
}

template <int blockSizeX, int blockSizeY>
__global__ void SharedMemoryMatMul(DeviceMatrix A, DeviceMatrix B,
                                   DeviceMatrix C, nn_real alpha,
                                   nn_real beta) {
  // row,col of element this thread is responsible for
  int row = threadIdx.y;
  int col = threadIdx.x;

  nn_real sum = 0;  // accumulator for the value of the current element
  __shared__ nn_real As[blockSizeX][blockSizeY];  // shared memory for A
  __shared__ nn_real Bs[blockSizeX][blockSizeY];  // shared memory for B
  int numProds = (A.n_cols + blockSizeX - 1) / blockSizeX;

  for (int k = 0; k < numProds; k++) {
    // load the current block of A and B into shared memory
    // x goes along cols for A
    if (k * blockSizeX + col < A.n_cols &&
        row + blockIdx.y * blockSizeY < A.n_rows) {
      As[row][col] = A(row + blockIdx.y * blockSizeY, k * blockSizeX + col);
    } else {
      As[row][col] = 0;
    }
    // x goes along rows for B
    if (k * blockSizeX + row < B.n_rows &&
        col + blockIdx.x * blockSizeY < B.n_cols) {
      Bs[row][col] = B(k * blockSizeX + row, col + blockIdx.x * blockSizeY);
    } else {
      Bs[row][col] = 0;
    }
    __syncthreads();
    // compute the value of the current element
    for (int i = 0; i < blockSizeX; i++) {
      sum += As[row][i] * Bs[i][col];
    }
    __syncthreads();
  }
  // write the value of the current element to C
  int writeRow = row + blockIdx.y * blockSizeY;
  int writeCol = col + blockIdx.x * blockSizeX;
  if (writeRow < C.n_rows && writeCol < C.n_cols) {
    C(writeRow, writeCol) = alpha * sum + beta * C(writeRow, writeCol);
  }
}

void sharedMemoryGEMM(DeviceMatrix A, DeviceMatrix B, DeviceMatrix C,
                      nn_real alpha, nn_real beta) {
  assert((A.n_cols) == (B.n_rows));
  assert(C.n_rows == (A.n_rows) && C.n_cols == (B.n_cols));

  constexpr int blockSize = 16;
  dim3 threads(blockSize, blockSize);
  dim3 grid((C.n_cols + threads.x - 1) / threads.x,
            (C.n_rows + threads.y - 1) / threads.y);
  SharedMemoryMatMul<blockSize, blockSize>
      <<<grid, threads>>>(A, B, C, alpha, beta);

  CHECK_LAUNCH("sharedMemoryGEMM");
}

// 32x32 Hierarchical Tiling
template <int num_thread, int blockItemsM, int blockItemsN, int blockItemsK>
__global__ void TiledMatMul(DeviceMatrix A, bool transa, DeviceMatrix B,
                            bool transb, DeviceMatrix C, nn_real alpha,
                            nn_real beta) {
  nn_real frag_a[2] = {0, 0};
  nn_real frag_b[4] = {0, 0, 0, 0};
  nn_real accumulator[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

  const int warp_tile_dimx = 16;
  const int warp_tile_dimy = 16;
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int block_tile_dimy = blockItemsM / warp_tile_dimy;
  const int wx = (warp_id / block_tile_dimy) * warp_tile_dimx;
  const int wy = (warp_id % block_tile_dimy) * warp_tile_dimy;
  const int tx = lane_id / 8;
  const int ty = lane_id % 8;

  const int K = transa ? A.n_rows : A.n_cols;
  const int A_max_row = transa ? A.n_cols : A.n_rows;
  const int A_max_col = transa ? A.n_rows : A.n_cols;
  const int B_max_row = transb ? B.n_cols : B.n_rows;
  const int B_max_col = transb ? B.n_rows : B.n_cols;

  __shared__ nn_real SMEMA[blockItemsK][blockItemsM + 1];
  __shared__ nn_real SMEMB[blockItemsK][blockItemsN + 1];

  for (int m = 0; m < (K + blockItemsK - 1) / blockItemsK; m++) {
    int by = threadIdx.x % blockItemsM;
    for (int bx = threadIdx.x / blockItemsM; bx < blockItemsK;
         bx += num_thread / blockItemsM) {
      if (blockItemsM * blockIdx.x + by < A_max_row &&
          m * blockItemsK + bx < A_max_col) {
        SMEMA[bx][by] =
            A(blockItemsM * blockIdx.x + by, m * blockItemsK + bx, transa);
      } else {
        SMEMA[bx][by] = 0;
      }
    }
    by = threadIdx.x % blockItemsK;
    for (int bx = threadIdx.x / blockItemsK; bx < blockItemsN;
         bx += num_thread / blockItemsK) {
      if (m * blockItemsK + by < B_max_row &&
          blockItemsN * blockIdx.y + bx < B_max_col) {
        SMEMB[by][bx] =
            B(m * blockItemsK + by, blockItemsN * blockIdx.y + bx, transb);
      } else {
        SMEMB[by][bx] = 0;
      }
    }
    __syncthreads();

    for (int k = 0; k < blockItemsK; k++) {
      int base_y = (warp_id % block_tile_dimy) * warp_tile_dimy + ty;
      int base_x = (warp_id / block_tile_dimy) * warp_tile_dimx + tx * 4;

      frag_a[0] = SMEMA[k][base_y];
      frag_a[1] = SMEMA[k][base_y + 8];
      for (int i = 0; i < 4; i++) {
        frag_b[i] = SMEMB[k][base_x + i];
        frag_b[i + 4] = SMEMB[k][base_x + i + warp_tile_dimx / 2];
      }

      for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 4; x++) {
          accumulator[y][x] += frag_a[y] * frag_b[x];
        }
      }
    }
    __syncthreads();
  }

  for (int x = 0; x < 4; x++) {
    for (int y = 0; y < 2; y++) {
      const int Cx = blockItemsN * blockIdx.y + wx + tx * 4 + x;
      const int Cy = blockItemsM * blockIdx.x + wy + ty + y * 8;
      if (Cy < C.n_rows && Cx < C.n_cols) {
        C(Cy, Cx) = alpha * accumulator[y][x] + beta * C(Cy, Cx);
      }
    }
  }
}

void tiledGEMM(DeviceMatrix A, bool transa, DeviceMatrix B, bool transb,
               DeviceMatrix C, nn_real alpha, nn_real beta) {
  int A_rows = transa ? A.n_cols : A.n_rows;
  int A_cols = transa ? A.n_rows : A.n_cols;
  int B_rows = transb ? B.n_cols : B.n_rows;
  int B_cols = transb ? B.n_rows : B.n_cols;
  assert((A_cols == B_rows) && (C.n_rows == A_rows) && (C.n_cols == B_cols));

  constexpr int block_m = 32;
  constexpr int block_n = 32;
  constexpr int block_k = 32;
  constexpr int num_thread = 128;
  dim3 grid((C.n_rows + block_m - 1) / block_m,
            (C.n_cols + block_n - 1) / block_n);
  TiledMatMul<num_thread, block_m, block_n, block_k>
      <<<grid, num_thread>>>(A, transa, B, transb, C, alpha, beta);

  CHECK_LAUNCH("tiledGEMM");
}