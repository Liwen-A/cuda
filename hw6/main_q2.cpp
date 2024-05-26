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

double calculate_mean(const std::vector<double>& v)
{
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  return sum / v.size();
}

double calculate_std(const std::vector<double>& v, double mean)
{
  double sum_of_squares = std::inner_product(
    v.begin(), v.end(), v.begin(), 0.0,
    [] (double a, double b) { return a + b; },
    [mean] (double a, double b) { return (a - mean) * (b - mean); });
  return std::sqrt(sum_of_squares / (v.size() - 1));
}

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

  // Print hyperparameters
  if (rank == 0)
    std::cout << "batch_size=" << hparams.batch_size << "; " 
              << "num_procs=" << num_procs << "; "
              << "hidden_size=" << hparams.hidden_size << ";" << std::endl;

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

  // Initialize a two-layer neural network on the CPU
  std::vector<int> H = {hparams.input_size, hparams.hidden_size,
                        hparams.num_classes};
  NeuralNetwork nn_seq(H);

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

  std::vector<double> par_dts;
  int repeats = 10;
  for (int i = 0; i < repeats; i++)
  {
    // Initialize a two-layer neural network on the CPU
    NeuralNetwork nn_par(H);
    // Initialize a neural network on the GPU from a neural network on the CPU
    DataParallelNeuralNetwork dpnn(nn_par, hparams.reg, hparams.learning_rate,
                                   hparams.batch_size, num_procs, rank);
    // Train on the GPU
    MPI_Barrier(MPI_COMM_WORLD);
    high_resolution_clock::time_point t0 = high_resolution_clock::now();
    dpnn.train(nn_par, X, y, hparams);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    duration<double> dt = duration_cast<duration<double>>(t1 - t0);
    MPI_Barrier(MPI_COMM_WORLD);
    par_dts.push_back(dt.count());

    // Check that both sets of updated parameters are identical
    if (rank == 0)
    {
      dpnn.to_cpu(nn_par);
      for (int i = 0; i < nn_par.num_layers; i++)
      {
        EXPECT_EQ(
          true,
          arma::approx_equal(nn_seq.W[i], nn_par.W[i], "both", TOL, TOL)
        );
        EXPECT_EQ(
          true,
          arma::approx_equal(nn_seq.b[i], nn_par.b[i], "both", TOL, TOL)
        );
      }
    }
  }

  double mean_par_dt = calculate_mean(par_dts);
  double std_par_dt = calculate_std(par_dts, mean_par_dt);
  if (rank == 0)
  {
    std::cout << "Mean time for Parallel Training: (rank 0; repeats = "
              << repeats << "): " << mean_par_dt << " seconds" << std::endl;
    std::cout << "Std. deviation of time for Parallel Training (repeats = "
              << repeats << "): " << std_par_dt << " seconds" << std::endl;
  }
}

TEST(gtestTrain, custom)
{
  hyperparameters hparams_custom =
      hyperparameters{784, 512, 10, 1e-4, 1e-4, 38400, 1, 3200, 0};
  testTrain(hparams_custom);
}

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  int run_tests = RUN_ALL_TESTS();
  MPI_Finalize();
  return run_tests;
}