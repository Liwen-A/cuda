#define ARMA_ALLOW_FAKE_GCC
#include <mpi.h>

#include <armadillo>
#include <chrono>

#include "common.h"
#include "cublas_v2.h"
#include "gpu_func.h"
#include "gtest/gtest.h"
#include <helper_cuda.h>
#include "neural_network.h"
#include "mnist.h"
#include "util.cuh"

using namespace std::chrono;

string file_train_images = "./MNIST_DATA/train-images.idx3-ubyte";
string file_train_labels = "./MNIST_DATA/train-labels.idx1-ubyte";
string file_test_images = "./MNIST_DATA/t10k-images.idx3-ubyte";
string file_test_labels = "./MNIST_DATA/t10k-labels.idx1-ubyte";

void testTrain(hyperparameters &hparams)
{
  // Continue MPI setup
  int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));

  // Get MPI hostname
  int resultlen;
  char name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(name, &resultlen);

  // Assign a GPU device to each MPI process
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  assert(num_devices > 0);
  const int my_device = rank % num_devices;
  checkCudaErrors(cudaSetDevice(my_device));
  printf("Rank %1d/%1d started on node %s [myGPU = %1d, nGPUs = %1d]\n", rank,
         num_procs, name, my_device, num_devices);

  // Initialize data
  assert(hparams.input_size == 784 && hparams.num_classes == 10);
  arma::Mat<nn_real> X(hparams.input_size, hparams.N);
  arma::Row<nn_real> label = arma::zeros<arma::Row<nn_real>>(hparams.N);
  arma::Mat<nn_real> y(hparams.num_classes, hparams.N);
  if (rank == 0)
  {
    read_mnist(file_train_images, X, hparams.N);
    read_mnist_label(file_train_labels, label, hparams.N);
    label_to_y(label, hparams.num_classes, y);

    assert(X.n_cols == hparams.N && X.n_rows == 784);
    assert(label.size() == hparams.N);
  }

  // Initialize two identical two-layer neural networks on the CPU
  std::vector<int> H = {hparams.input_size, hparams.hidden_size,
                        hparams.num_classes};
  NeuralNetwork nn_seq(H);
  NeuralNetwork nn_par(H);

  // Initialize a neural network on the GPU from a neural network on the CPU
  DataParallelNeuralNetwork dpnn(nn_par, hparams.reg, hparams.learning_rate,
                                 hparams.batch_size, num_procs, rank);

  // Train on the CPU
  if (rank == 0)
  {
    high_resolution_clock::time_point t0 = high_resolution_clock::now();
    train(nn_seq, X, y, hparams);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    duration<double> dt = duration_cast<duration<double>>(t1 - t0);
    std::cout << "Time for Sequential Training: " << dt.count()
              << " seconds" << std::endl;
  }

  // Train on the GPU
  MPI_Barrier(MPI_COMM_WORLD);
  high_resolution_clock::time_point t0 = high_resolution_clock::now();
  dpnn.train(nn_par, X, y, hparams);
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  duration<double> dt = duration_cast<duration<double>>(t1 - t0);
  if (rank == 0)
    std::cout << "Time for Parallel Training: " << dt.count()
              << " seconds" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  // Check that both sets of updated parameters are identical
  if (rank == 0)
  {
    dpnn.to_cpu(nn_par);
    for (int i = 0; i < nn_par.num_layers; i++)
    {
      EXPECT_EQ(true,
                arma::approx_equal(nn_seq.W[i], nn_par.W[i], "both", TOL, TOL));
      EXPECT_EQ(true,
                arma::approx_equal(nn_seq.b[i], nn_par.b[i], "both", TOL, TOL));
    }
  }
}

TEST(gtestTrain, small1)
{
  hyperparameters hparams_small =
      hyperparameters{784, 32, 10, 1e-4, 0.01, 32, 1, 32, 0};
  testTrain(hparams_small);
}

TEST(gtestTrain, small2)
{
  hyperparameters hparams_small =
      hyperparameters{784, 32, 10, 1e-4, 0.005, 512, 2, 32, 0};
  testTrain(hparams_small);
}

TEST(gtestTrain, small3)
{
  hyperparameters hparams_small =
      hyperparameters{784, 32, 10, 1e-4, 0.002, 512, 16, 32, 0};
  testTrain(hparams_small);
}

TEST(gtestTrain, medium)
{
  hyperparameters hparams_medium =
      hyperparameters{784, 128, 10, 1e-4, 0.004, 1087, 4, 64, 0};
  testTrain(hparams_medium);
}

TEST(gtestTrain, large)
{
  hyperparameters hparams_large =
      hyperparameters{784, 1013, 10, 1e-4, 0.002, 1087, 4, 67, 0};
  testTrain(hparams_large);
}

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  int run_tests = RUN_ALL_TESTS();
  MPI_Finalize();
  return run_tests;
}