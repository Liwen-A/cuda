#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <sstream>

#include "common.h"
#include "gpu_func.h"
#include "gtest/gtest.h"

#define ASSERT_MAT_SAME_SIZE(mat1, mat2)                                       \
  assert(mat1.n_rows == mat2.n_rows && mat1.n_cols == mat2.n_cols)

/*
 * Follow these CPU implementations and the code in the tests below to write
 * your GPU implementations. 
 * See documentation for functions such as arma::exp, arma::sum, repmat, etc.
 * from the header <armadillo> at https://arma.sourceforge.net/docs.html.
 */

// Sigmoid activation
void sigmoid(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2) {
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1. / (1. + arma::exp(-mat));
}

// Softmax activation
void softmax(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2) {
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
}

/*
 * Test GPU implementations
 */

TEST(testMatrix, deviceWarmup) {
  DWarmup(); // For accurate test timing
}

TEST(testMatrix, deviceSigmoid) {
  int n_rows = 5, n_cols = 5;

  arma::Mat<nn_real> X_in(n_rows, n_cols, arma::fill::randn);
  arma::Mat<nn_real> X_out;
  arma::Mat<nn_real> X_out_gpu(n_rows, n_cols);

  sigmoid(X_in, X_out);

  DeviceMatrix d_X_in(X_in);
  DeviceMatrix d_X_out(n_rows, n_cols);

  DSigmoid(d_X_in, d_X_out);
  d_X_out.to_cpu(X_out_gpu);

  EXPECT_EQ(true, arma::approx_equal(X_out, X_out_gpu, "absdiff", TOL));
  EXPECT_EQ(true, arma::approx_equal(X_out, X_out_gpu, "reldiff", TOL));
}

TEST(testMatrix, deviceRepeatColVec) {
  int n_rows = 5, n_cols = 5;
  arma::Mat<nn_real> X_in(n_rows, 1, arma::fill::randn);
  arma::Mat<nn_real> X_out = arma::repmat(X_in, 1, n_cols);
  arma::Mat<nn_real> X_out_gpu_cpu(n_rows, n_cols);

  DeviceMatrix X_in_gpu(X_in);
  DeviceMatrix X_out_gpu(n_rows, n_cols);
  DRepeatColVec(X_in_gpu, X_out_gpu, n_cols);
  X_out_gpu.to_cpu(X_out_gpu_cpu);

  EXPECT_EQ(true, arma::approx_equal(X_out, X_out_gpu_cpu, "absdiff", TOL));
  EXPECT_EQ(true, arma::approx_equal(X_out, X_out_gpu_cpu, "reldiff", TOL));
}

TEST(testMatrix, deviceSum) {
  int n_rows = 5, n_cols = 5;
  arma::Mat<nn_real> X_in(n_rows, n_cols, arma::fill::randn);
  arma::Mat<nn_real> X_out_r = arma::sum(X_in, 0);
  arma::Mat<nn_real> X_out_c = arma::sum(X_in, 1);
  arma::Mat<nn_real> X_out_gpu_cpu_r(1, n_cols);
  arma::Mat<nn_real> X_out_gpu_cpu_c(n_rows, 1);

  DeviceMatrix X_in_gpu(X_in);
  DeviceMatrix X_out_gpu_r(1, n_cols);
  DeviceMatrix X_out_gpu_c(n_rows, 1);
  DSum(X_in_gpu, X_out_gpu_r, static_cast<nn_real>(1.0), 0);
  DSum(X_in_gpu, X_out_gpu_c, static_cast<nn_real>(1.0), 1);

  X_out_gpu_r.to_cpu(X_out_gpu_cpu_r);
  X_out_gpu_c.to_cpu(X_out_gpu_cpu_c);

  EXPECT_EQ(true, arma::approx_equal(X_out_r, X_out_gpu_cpu_r, "absdiff", TOL));
  EXPECT_EQ(true, arma::approx_equal(X_out_r, X_out_gpu_cpu_r, "reldiff", TOL));
  EXPECT_EQ(true, arma::approx_equal(X_out_c, X_out_gpu_cpu_c, "absdiff", TOL));
  EXPECT_EQ(true, arma::approx_equal(X_out_c, X_out_gpu_cpu_c, "reldiff", TOL));
}

TEST(testMatrix, deviceSoftmax) {
  int n_rows = 5, n_cols = 5;
  arma::Mat<nn_real> X_in(n_rows, n_cols, arma::fill::randn);
  arma::Mat<nn_real> X_out;
  arma::Mat<nn_real> X_out_gpu(n_rows, n_cols);
  softmax(X_in, X_out);

  DeviceMatrix d_X_in(X_in);
  DeviceMatrix d_X_out(n_rows, n_cols);

  DSoftmax(d_X_in, d_X_out, 0);
  d_X_out.to_cpu(X_out_gpu);

  EXPECT_EQ(true, arma::approx_equal(X_out, X_out_gpu, "absdiff", TOL));
  EXPECT_EQ(true, arma::approx_equal(X_out, X_out_gpu, "reldiff", TOL));
}

TEST(testMatrix, deviceCELoss) {
  int n_rows = 5, n_cols = 5;
  arma::Col<nn_real> tmp1(n_rows, arma::fill::ones);
  arma::Mat<nn_real> tmp2(n_rows, n_cols - 1, arma::fill::zeros);
  arma::Mat<nn_real> y = arma::join_horiz(tmp1, tmp2);
  arma::Mat<nn_real> y_pred(n_rows, n_cols, arma::fill::randu);
  nn_real loss_cpu = -arma::accu(arma::log(y_pred.elem(arma::find(y == 1))));
  /**
   * Note that above we sum over only those elements of y_pred that correspond
   * to a 1 in y. In the function DCELoss, we sum over all elements of the
   * temporary matrix T used to store the loss (see hint in DCELoss in 
   * gpu_func.h), but this is equivalent since elements of T that correspond
   * to a 0 in y will be equal to 0 if the kernel MatCrossEntropyLoss is 
   * implemented correctly.
   */

  DeviceMatrix y_pred_gpu(y_pred);
  DeviceMatrix y_gpu(y);
  DeviceMatrix loss_gpu(1, 1);
  DCELoss(y_pred_gpu, y_gpu, loss_gpu);

  arma::Mat<nn_real> loss_gpu_cpu(1, 1);
  loss_gpu.to_cpu(loss_gpu_cpu);

  #ifndef USE_DOUBLE
  EXPECT_FLOAT_EQ(loss_cpu, loss_gpu_cpu(0, 0));
  #else
  EXPECT_DOUBLE_EQ(loss_cpu, loss_gpu_cpu(0, 0));
  #endif
}

TEST(testMatrix, deviceElemArith) {
  int n_rows = 5, n_cols = 5;
  arma::Mat<nn_real> A(n_rows, n_cols, arma::fill::randn);
  arma::Mat<nn_real> B(n_rows, n_cols, arma::fill::randn);
  arma::Mat<nn_real> A_gpu_cpu(n_rows, n_cols);
  nn_real alpha = arma::randn();
  nn_real beta = arma::randn();
  DeviceMatrix A_gpu(A);
  DeviceMatrix B_gpu(B);

  A = alpha * (A + beta * B);
  DElemArith(A_gpu, B_gpu, alpha, beta);
  A_gpu.to_cpu(A_gpu_cpu);
  EXPECT_EQ(true, arma::approx_equal(A, A_gpu_cpu, "absdiff", TOL));
  EXPECT_EQ(true, arma::approx_equal(A, A_gpu_cpu, "reldiff", TOL));
}

TEST(testMatrix, deviceSquare) {
  int n_rows = 5, n_cols = 5;
  arma::Mat<nn_real> X_in(n_rows, n_cols, arma::fill::randn);
  arma::Mat<nn_real> X_out = arma::pow(X_in, 2);
  arma::Mat<nn_real> X_out_gpu_cpu(n_rows, n_cols);

  DeviceMatrix X_in_gpu(X_in);
  DeviceMatrix X_out_gpu(n_rows, n_cols);
  DSquare(X_in_gpu, X_out_gpu);

  X_out_gpu.to_cpu(X_out_gpu_cpu);

  EXPECT_EQ(true, arma::approx_equal(X_out, X_out_gpu_cpu, "absdiff", TOL));
  EXPECT_EQ(true, arma::approx_equal(X_out, X_out_gpu_cpu, "reldiff", TOL));
}

TEST(testMatrix, deviceSigmoidBackprop) {
  int n_rows = 5, n_cols = 5;
  arma::Mat<nn_real> da1(n_rows, n_cols, arma::fill::randn);
  arma::Mat<nn_real> a1(n_rows, n_cols, arma::fill::randn);
  arma::Mat<nn_real> dz1 = da1 % a1 % (1 - a1);
  arma::Mat<nn_real> dz1_gpu_cpu(n_rows, n_cols);

  DeviceMatrix da1_gpu(da1);
  DeviceMatrix a1_gpu(a1);
  DeviceMatrix dz1_gpu(n_rows, n_cols);
  DSigmoidBackprop(da1_gpu, a1_gpu, dz1_gpu);

  dz1_gpu.to_cpu(dz1_gpu_cpu);

  EXPECT_EQ(true, arma::approx_equal(dz1, dz1_gpu_cpu, "absdiff", TOL));
  EXPECT_EQ(true, arma::approx_equal(dz1, dz1_gpu_cpu, "reldiff", TOL));
}
