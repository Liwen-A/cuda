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

void testStep(hyperparameters &hparams) {
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

  // Forward, backward, and optimizer step on the CPU
  struct cache cache_seq;
  struct grads grads_seq;
  forward(nn_seq, X, cache_seq);
  backward(nn_seq, y, hparams.reg, cache_seq, grads_seq);
  for (int i = 0; i < nn_seq.W.size(); ++i)
    nn_seq.W[i] -= hparams.learning_rate * grads_seq.dW[i];
  for (int i = 0; i < nn_seq.b.size(); ++i)
    nn_seq.b[i] -= hparams.learning_rate * grads_seq.db[i];

  // Forward, backward, and optimizer step on the GPU
  DeviceMatrix dX = DeviceMatrix(X);
  DeviceMatrix dy = DeviceMatrix(y);
  dpnn.forward(dX);
  dpnn.backward(dy, 1.0 / hparams.batch_size);
  dpnn.step();

  // Create CPU copies of updated parameters
  dpnn.to_cpu(nn_par);

  // Check that both sets of updated parameters are identical
  for (int i = 0; i < nn_par.num_layers; i++) {
    EXPECT_EQ(true, arma::approx_equal(nn_seq.W[i], nn_par.W[i], "both", TOL,
                                       TOL));
    EXPECT_EQ(true, arma::approx_equal(nn_seq.b[i], nn_par.b[i], "both", TOL,
                                       TOL));
  }
}

TEST(gtestStep, deviceWarmup) {
  DWarmup();
}

TEST(gtestStep, small) {
  hyperparameters hparams_small = hyperparameters{
    9, 4, 10, 
    1e-4, 0.01, 
    8, 1, 8
  };
  testStep(hparams_small);
}

TEST(gtestStep, medium) {
  hyperparameters hparams_medium = hyperparameters{
    100, 16, 10, 
    1e-4, 0.01, 
    80, 1, 80
  };
  testStep(hparams_medium);
}

TEST(gtestStep, large) {
hyperparameters hparams_large = hyperparameters{
    784, 64, 10, 
    1e-4, 0.01, 
    800, 1, 800
  };
  testStep(hparams_large);
}