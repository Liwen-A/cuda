#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <cuda_runtime.h>
#include "common.h"
#include <memory>

/**
 * A class that manages memory for GPU data.
 * The memory is allocated on the GPU when the object is created
 * and deallocated when the object is destroyed.
 * If the object is created with a pointer to CPU data, the data
 * is also copied to the GPU memory when the object is created.
 */
class DeviceAllocator
{
public:
  nn_real *data = nullptr; // pointer to the GPU data
  int nbytes = 0;          // number of bytes that were allocated

  DeviceAllocator() = default;

  // Allocate memory on the GPU and copy the CPU data to the GPU.
  DeviceAllocator(nn_real *cpu_data, int n);

  // Only allocate memory on the GPU.
  DeviceAllocator(int n);

  // Deallocate the memory on the GPU.
  ~DeviceAllocator();

  // Copy the GPU data to the CPU pointer.
  void to_cpu(nn_real *cpu_data);
};

/**
 * Matrix on the GPU.
 *
 * Create a new memory allocator for the matrix and store the shared pointer to this
 * allocator whenever a new DeviceMatrix is constructed. Use the memory allocator to
 * allocate and deallocate memory for the matrix.
 *
 * A shared_ptr is used to ensure that the memory is deallocated only when the last
 * DeviceMatrix object that references the memory is destroyed.
 *
 * This allows shallow copies of the DeviceMatrix object to be created without
 * duplicating the memory, thereby avoiding potential memory leaks.
 */
struct DeviceMatrix
{
private:
  std::shared_ptr<DeviceAllocator> allocator; // shared pointer to the memory allocator
  // Note that the allocator is made private because we don't need to expose it to the
  // user of the DeviceMatrix class. The user only needs to interact with the DeviceMatrix
  // object and not the memory allocator. All memory allocation and deallocation is handled
  // internally by this DeviceAllocator.
  nn_real *data = nullptr;

public:
  int n_rows = 0;
  int n_cols = 0;

  int total_bytes();

  DeviceMatrix() = default;

  // Initialize the matrix with the given dimensions.
  DeviceMatrix(int n_rows, int n_cols);

  // Initialize the matrix with the given CPU matrix.
  DeviceMatrix(arma::Mat<nn_real> &cpu_mat);

  // This function copies the data from the GPU to the given arma matrix.
  void to_cpu(arma::Mat<nn_real> &cpu_mat);

  /**
   * A device function that returns the element at the specified
   * row and column of the matrix.
   *
   * @param row The row index of the element.
   * @param col The column index of the element.
   * @return The element at the specified row and column of the matrix.
   */
  __device__ nn_real &operator()(int row, int col);
};

/**
 * Applies the sigmoid function element-wise to a matrix on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
void DSigmoid(DeviceMatrix src, DeviceMatrix dst);

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
 * Computes the sum of a matrix along the specified axis on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 * @param alpha The scaling factor for the output matrix.
 * @param axis The axis along which to compute the sum.
 */
void DSum(DeviceMatrix src, DeviceMatrix dst, nn_real alpha, int axis);

/**
 * Computes the softmax function along the specified axis on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 * @param axis The axis along which to compute the softmax.
 */
void DSoftmax(DeviceMatrix src, DeviceMatrix dst, int axis);

/**
 * Computes the cross-entropy loss on the GPU.
 *
 * @param y_pred The predicted labels.
 * @param y The ground truth labels.
 * @param loss The output loss (note that this is treated as a 1x1 matrix).
 */
void DCELoss(DeviceMatrix y_pred, DeviceMatrix y, DeviceMatrix loss);

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
 * Computes the square of a matrix element-wise on the GPU.
 *
 * @param src The input matrix.
 * @param dst The output matrix.
 */
void DSquare(DeviceMatrix src, DeviceMatrix dst);

/**
 * Computes backpropagation for the sigmoid function on the GPU.
 * dz1 = da1 * a1 * (1 - a1)
 *
 * @param da1 The upstream gradient.
 * @param a1 The input activation.
 * @param dz1 The output derivative.
 */
void DSigmoidBackprop(DeviceMatrix da1, DeviceMatrix a1, DeviceMatrix dz1);

void DWarmup();

#endif
