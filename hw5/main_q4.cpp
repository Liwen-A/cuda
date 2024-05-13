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

void testBackward(hyperparameters &hparams) {
  // Initialize data
  arma::Mat<nn_real> X(hparams.input_size, hparams.batch_size, 
                       arma::fill::randu);
  arma::Row<nn_real> label = arma::randi<arma::Row<nn_real>>(
    hparams.batch_size, arma::distr_param(0, hparams.num_classes - 1));
  arma::Mat<nn_real> y(hparams.num_classes, hparams.batch_size);
  label_to_y(label, hparams.num_classes, y);

  // Initialize two identical two-layer neural networks on the CPU
  std::vector<int> H = {hparams.input_size, hparams.hidden_size, 
                        hparams.num_classes};
  NeuralNetwork nn_seq(H);
  NeuralNetwork nn_par(H);

  // Initialize a neural network on the GPU from a neural network on the CPU
  DataParallelNeuralNetwork dpnn(nn_par, hparams.reg, hparams.learning_rate, 
                                 hparams.batch_size, 1);

  // Forward and backward pass on the CPU
  struct cache cache_seq;
  struct grads grads_seq;
  forward(nn_seq, X, cache_seq);
  backward(nn_seq, y, hparams.reg, cache_seq, grads_seq);

  // Forward and backward pass on the GPU
  DeviceMatrix dX = DeviceMatrix(X);
  DeviceMatrix dy = DeviceMatrix(y);
  dpnn.forward(dX);
  dpnn.backward(dy, 1.0 / hparams.batch_size);

  // Create CPU copies of members of GPUGrads
  std::vector<arma::Mat<nn_real>> dpnn_grads_dW;
  for (int i = 0; i < nn_par.num_layers; i++) {
    arma::Mat<nn_real> temp(dpnn.grads.dW[i].n_rows, dpnn.grads.dW[i].n_cols);
    dpnn.grads.dW[i].to_cpu(temp);
    dpnn_grads_dW.push_back(temp);
  }
  std::vector<arma::Mat<nn_real>> dpnn_grads_db;
  for (int i = 0; i < nn_par.num_layers; i++) {
    arma::Mat<nn_real> temp(dpnn.grads.db[i].n_rows, dpnn.grads.db[i].n_cols);
    dpnn.grads.db[i].to_cpu(temp);
    dpnn_grads_db.push_back(temp);
  }

  // Check that items in both caches are identical
  for (int i = 0; i < nn_par.num_layers; i++) {
    EXPECT_EQ(true, arma::approx_equal(grads_seq.dW[i], dpnn_grads_dW[i], 
                                       "both", TOL, TOL));
    EXPECT_EQ(true, arma::approx_equal(grads_seq.db[i], dpnn_grads_db[i], 
                                       "both", TOL, TOL));
  }
}

TEST(gtestBackward, deviceWarmup) {
  DWarmup();
}

TEST(gtestBackward, small) {
  hyperparameters hparams_small = hyperparameters{
    9, 4, 10, 
    1e-4, 0.01, 
    8, 1, 8
  };
  testBackward(hparams_small);
}

TEST(gtestBackward, medium) {
hyperparameters hparams_medium = hyperparameters{
    100, 16, 10, 
    1e-4, 0.01, 
    80, 1, 80
  };
  testBackward(hparams_medium);
}

TEST(gtestBackward, large) {
hyperparameters hparams_large = hyperparameters{
    784, 64, 10, 
    1e-4, 0.01, 
    800, 1, 800
  };
  testBackward(hparams_large);
}