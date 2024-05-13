#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <chrono>
#include "cublas_v2.h"
#include <mpi.h>

#include "common.h"
#include "gpu_func.h"
#include "gtest/gtest.h"
#include "neural_network.h"
#include "util.cuh"

void testForward(hyperparameters &hparams) {
  // Initialize data
  arma::Mat<nn_real> X(hparams.input_size, hparams.batch_size, 
                       arma::fill::randu);

  // Initialize two identical two-layer neural networks on the CPU
  std::vector<int> H = {hparams.input_size, hparams.hidden_size, 
                        hparams.num_classes};
  NeuralNetwork nn_seq(H);
  NeuralNetwork nn_par(H);

  // Initialize a neural network on the GPU from a neural network on the CPU
  DataParallelNeuralNetwork dpnn(nn_par, hparams.reg, hparams.learning_rate, 
                                 hparams.batch_size, 1);

  // Forward pass on the CPU
  struct cache cache_seq;
  forward(nn_seq, X, cache_seq);

  // Forward pass on the GPU
  DeviceMatrix dX = DeviceMatrix(X);
  dpnn.forward(dX);

  // Create CPU copies of members of GPUCache
  arma::Mat<nn_real> dpnn_cache_X(dpnn.cache.X.n_rows, dpnn.cache.X.n_cols);
  dpnn.cache.X.to_cpu(dpnn_cache_X);
  std::vector<arma::Mat<nn_real>> dpnn_cache_z;
  for (int i = 0; i < nn_par.num_layers; i++) {
    arma::Mat<nn_real> temp(dpnn.cache.z[i].n_rows, dpnn.cache.z[i].n_cols);
    dpnn.cache.z[i].to_cpu(temp);
    dpnn_cache_z.push_back(temp);
  }
  std::vector<arma::Mat<nn_real>> dpnn_cache_a;
  for (int i = 0; i < nn_par.num_layers; i++) {
    arma::Mat<nn_real> temp(dpnn.cache.a[i].n_rows, dpnn.cache.a[i].n_cols);
    dpnn.cache.a[i].to_cpu(temp);
    dpnn_cache_a.push_back(temp);
  }
  arma::Mat<nn_real> dpnn_cache_yc(dpnn.cache.yc.n_rows, dpnn.cache.yc.n_cols);
  dpnn.cache.yc.to_cpu(dpnn_cache_yc);

  // Check that items in both caches are identical
  EXPECT_EQ(true, arma::approx_equal(X, cache_seq.X, "both", TOL, TOL));
  EXPECT_EQ(true, arma::approx_equal(X, dpnn_cache_X, "both", TOL, TOL));
  for (int i = 0; i < nn_par.num_layers; i++) {
    EXPECT_EQ(true, arma::approx_equal(cache_seq.z[i], dpnn_cache_z[i], "both", 
                                       TOL, TOL));
    EXPECT_EQ(true, arma::approx_equal(cache_seq.a[i], dpnn_cache_a[i], "both", 
                                       TOL, TOL));
  }
  EXPECT_EQ(true, arma::approx_equal(cache_seq.yc, dpnn_cache_yc, "both", TOL, 
                                     TOL));
}

TEST(gtestForward, deviceWarmup) {
  DWarmup();
}

TEST(gtestForward, small) {
  hyperparameters hparams_small = hyperparameters{
    9, 4, 10, 
    1e-4, 0.01, 
    8, 1, 8
  };
  testForward(hparams_small);
}

TEST(gtestForward, medium) {
  hyperparameters hparams_medium = hyperparameters{
    100, 16, 10, 
    1e-4, 0.01, 
    80, 1, 80
  };
  testForward(hparams_medium);
}

TEST(gtestForward, large) {
  hyperparameters hparams_large = hyperparameters{
    784, 64, 10, 
    1e-4, 0.01, 
    800, 1, 800
  };
  testForward(hparams_large);
}