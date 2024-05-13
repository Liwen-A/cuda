#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#define ARMA_ALLOW_FAKE_GCC
#include <cuda_runtime.h>

#include <armadillo>
#include <memory>

#include "common.h"

/**
 * A class that manages memory for the GPU data.
 */
class DeviceAllocator {
 private:
  nn_real *data = nullptr;  // pointer to the GPU data
  int nbytes = 0;           // number of bytes that were allocated

 public:
  DeviceAllocator() = default;
  DeviceAllocator(int n);
  DeviceAllocator(nn_real *cpu_data, int n);
  ~DeviceAllocator();

  int total_bytes();
  nn_real *memptr();
};

/**
 * Matrix on the GPU.
 */
class DeviceMatrix {
  /**
   * A pointer to the matrix data.
   */
 private:
  std::shared_ptr<DeviceAllocator> allocator = nullptr;
  nn_real *data = nullptr;
  int nbytes = 0;

 public:
  int n_rows = 0;
  int n_cols = 0;

  DeviceMatrix() = default;
  DeviceMatrix(int n_rows, int n_cols);
  DeviceMatrix(arma::Mat<nn_real> &cpu_mat);
  DeviceMatrix(nn_real *gpu_mat, int n_rows, int n_cols);

  int total_bytes();
  nn_real *memptr();
  void to_cpu(arma::Mat<nn_real> &cpu_mat);
  void to_gpu(DeviceMatrix &gpu_mat);

  /**
   * A device function that returns the element at the specified
   * row and column of the matrix.
   *
   * @param row The row index of the element.
   * @param col The column index of the element.
   * @param transpose Is the matrix being accessed transposed?
   * @return The element at the specified row and column of the matrix.
   */
  __device__ nn_real &operator()(int row, int col, bool transpose = false);
};

/**
 * A dummy CUDA kernel to warm up the GPU for accurate timing of tests.
 */
__global__ void Warmup();

/**
 * Calls a dummy CUDA kernel to warm up the GPU.
 */
void DWarmup();

/**
 * A CUDA kernel that prints a matrix on the GPU to the standard output.
 *
 * @param mat The input matrix.
 * @param name The name of the matrix.
 * @param trans Whether to print the matrix in transposed form.
 */
__global__ void PrintMatrix(DeviceMatrix mat);

/**
 * Prints a matrix on the GPU to the standard output.
 */
void DevicePrintMatrix(DeviceMatrix mat);

/**
 * A CUDA kernel that applies the sigmoid function element-wise to a matrix on
 * the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
__global__ void MatSigmoid(DeviceMatrix src, DeviceMatrix dst);

/**
 * Applies the sigmoid function element-wise to a matrix on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
void DSigmoid(DeviceMatrix src, DeviceMatrix dst);

/**
 * A CUDA kernel that repeats each column of the source matrix `repeat` times
 * and stores the result in the destination matrix.
 *
 * @param src The source matrix to repeat.
 * @param dst The destination matrix to store the repeated columns.
 * @param repeat The number of times to repeat each column.
 */
__global__ void MatRepeatColVec(DeviceMatrix src, DeviceMatrix dst, int repeat);

/**
 * Repeats each column of the source matrix `repeat` times and stores the result
 * in the destination matrix.
 *
 * @param src The source matrix to repeat.
 * @param dst The destination matrix to store the repeated columns.
 * @param repeat The number of times to repeat each column.
 */
void DRepeatColVec(DeviceMatrix src, DeviceMatrix dst, int repeat);

/**
 * A CUDA kernel that computes the sum of a matrix along a specified axis on the
 * GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 * @param alpha The scaling factor for the sum.
 * @param axis The axis along which to compute the sum (0 for rows, 1 for
 * columns).
 */
__global__ void MatSum(DeviceMatrix src, DeviceMatrix dst, nn_real alpha,
                       int axis);

/**
 * Computes the sum of a matrix along the specified axis on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 * @param alpha The scaling factor for the output matrix.
 * @param axis The axis along which to compute the sum.
 */
void DSum(DeviceMatrix src, DeviceMatrix dst, nn_real alpha, int axis);

/**
 * A CUDA kernel that applies the softmax function along a specified axis to a
 * matrix on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 * @param axis The axis along which to apply the softmax function (0 for rows, 1
 * for columns).
 */
__global__ void MatSoftmax(DeviceMatrix src, DeviceMatrix dst, int axis);

/**
 * Computes the softmax function along the specified axis on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 * @param axis The axis along which to compute the softmax.
 */
void DSoftmax(DeviceMatrix src, DeviceMatrix dst, int axis);

/**
 * A CUDA kernel that computes the cross-entropy loss between predicted and true
 * labels on the GPU.
 *
 * @param y_pred The predicted label matrix.
 * @param y The true label matrix.
 * @param loss The output loss matrix.
 */
__global__ void MatCrossEntropyLoss(DeviceMatrix y_pred, DeviceMatrix y,
                                    DeviceMatrix loss);

/**
 * Computes the cross-entropy loss on the GPU.
 *
 * @param y_pred The predicted labels.
 * @param y The ground truth labels.
 * @param loss The output loss.
 */
void DCELoss(DeviceMatrix y_pred, DeviceMatrix y, DeviceMatrix loss);

/**
 * A CUDA kernel that performs element-wise arithmetic operations on two
 * matrices on the GPU. A = alpha * (A + beta * B)
 *
 * @param A The first input matrix.
 * @param B The second input matrix.
 * @param alpha The scaling factor for the first input matrix.
 * @param beta The scaling factor for the second input matrix.
 */
__global__ void MatElemArith(DeviceMatrix A, DeviceMatrix B, nn_real alpha,
                             nn_real beta);

/**
 * Performs element-wise arithmetic on two matrices on the GPU.
 * A = alpha * (A + beta * B)
 *
 * @param A The first matrix.
 * @param B The second matrix.
 * @param alpha The scaling factor for the first matrix
 * @param beta The scaling factor for the second matrix.
 */
void DElemArith(DeviceMatrix A, DeviceMatrix B, nn_real alpha, nn_real beta);

/**
 * A CUDA kernel that computes the element-wise square of a matrix on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
__global__ void MatSquare(DeviceMatrix src, DeviceMatrix dst);

/**
 * Computes the square of a matrix element-wise on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
void DSquare(DeviceMatrix src, DeviceMatrix dst);

// __global__ void MatSquareSum(DeviceMatrix src, nn_real *sum);

// /**
//  * Computes the sum of the squares of a matrix on the GPU and stores the
//  result
//  * in a scalar variable.
//  *
//  * @param src The input matrix.
//  * @param sum The reduction sum.
//  */
// void DSquareSum(DeviceMatrix src, nn_real *sum);

/**
 * A CUDA kernel that computes backpropagation for sigmoid function on the GPU.
 * dz1 = da1 * a1 * (1 - a1)
 *
 * @param da1 The upstream derivative matrix.
 * @param a1 The activation matrix.
 * @param dz1 The output derivative matrix.
 */
__global__ void MatSigmoidBackProp(DeviceMatrix da1, DeviceMatrix a1,
                                   DeviceMatrix dz1);

/**
 * Computes backpropagation for the sigmoid function on the GPU.
 * dz1 = da1 * a1 * (1 - a1)
 *
 * @param da1 The upstream gradient.
 * @param a1 The input activation.
 * @param dz1 The output derivative.
 */
void DSigmoidBackprop(DeviceMatrix da1, DeviceMatrix a1, DeviceMatrix dz1);

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

void tiledGEMM(DeviceMatrix A, bool transa, DeviceMatrix B, bool transb,
               DeviceMatrix C, nn_real alpha, nn_real beta);
#endif