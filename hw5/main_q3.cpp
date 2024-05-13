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

void testLoss(hyperparameters &hparams) {
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

  // Forward pass and loss computation on the CPU
  struct cache cache_seq;
  forward(nn_seq, X, cache_seq);
  nn_real loss_seq = loss(nn_seq, cache_seq.yc, y, hparams.reg);

  // Forward pass and loss computation on the GPU
  DeviceMatrix dX = DeviceMatrix(X);
  DeviceMatrix dY = DeviceMatrix(y);
  dpnn.forward(dX);
  nn_real loss_par = dpnn.loss(y, 1.0 / hparams.batch_size);

  // Check that both losses are identical
  EXPECT_NEAR(loss_seq, loss_par, TOL);
}

TEST(gtestLoss, deviceWarmup) {
  DWarmup();
}

TEST(gtestLoss, small) {
  hyperparameters hparams_small = hyperparameters{
    9, 4, 10, 
    1e-4, 0.01, 
    8, 1, 8
  };
  testLoss(hparams_small);
}

TEST(gtestLoss, medium) {
  hyperparameters hparams_medium = hyperparameters{
    100, 16, 10, 
    1e-4, 0.01, 
    80, 1, 80
  };
  testLoss(hparams_medium);
}

TEST(gtestLoss, large) {
  hyperparameters hparams_large = hyperparameters{
    784, 64, 10, 
    1e-4, 0.01, 
    800, 1, 800
  };
  testLoss(hparams_large);
}