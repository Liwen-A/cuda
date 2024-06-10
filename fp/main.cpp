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

double calculate_mean(const std::vector<double> &v)
{
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  return sum / v.size();
}

double calculate_std(const std::vector<double> &v, double mean)
{
  double sum_of_squares = std::inner_product(
      v.begin(), v.end(), v.begin(), 0.0,
      [](double a, double b)
      { return a + b; },
      [mean](double a, double b)
      { return (a - mean) * (b - mean); });
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

  // Loop for more accurate timing estimates
  std::vector<double> par_dts;
  int repeats = 10;
  for (int i = 0; i < repeats; i++)
  {
    // Initialize a two-layer neural network on the GPU
    NeuralNetwork nn_par(H);
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
        float abs_err_W = arma::norm(nn_seq.W[i] - nn_par.W[i], "fro");
        float abs_err_b = arma::norm(nn_seq.b[i] - nn_par.b[i]);
        EXPECT_LT(abs_err_W, TOL);
        EXPECT_LT(abs_err_b, TOL);
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

void profileTrain(hyperparameters &hparams, bool cpu = true)
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

  assert(hparams.input_size == 784 && hparams.num_classes == 10);

  // Print hyperparameters
  if (rank == 0)
    std::cout << "batch_size=" << hparams.batch_size << "; "
              << "num_procs=" << num_procs << "; "
              << "hidden_size=" << hparams.hidden_size << ";" << std::endl;

  // Initialize data
  arma::Mat<nn_real> X(hparams.input_size, hparams.N);
  arma::Row<nn_real> label = arma::zeros<arma::Row<nn_real>>(hparams.N);
  arma::Mat<nn_real> y(hparams.num_classes, hparams.N);
  // y is the matrix of one-hot label vectors where only y[c] = 1,
  // where c is the right class.

  arma::Mat<nn_real> x_test(hparams.input_size, hparams.test_size);
  arma::Row<nn_real> label_test(hparams.test_size);
  arma::Mat<nn_real> y_test(hparams.num_classes, hparams.test_size);

  if (rank == 0)
  {
    assert(X.n_cols == hparams.N &&
           X.n_rows == hparams.input_size);
    assert(label.size() == hparams.N);
    assert(y.n_cols == X.n_cols &&
           y.n_cols == hparams.N);
    assert(y.n_rows == hparams.num_classes);

    // Read MNIST images into Armadillo mat vector
    read_mnist(file_train_images, X, hparams.N);
    read_mnist_label(file_train_labels, label, hparams.N);
    label_to_y(label, hparams.num_classes, y);
    std::cout << "Size of training set =  " << X.n_cols << std::endl;

    if (hparams.test_size > 0)
    {
      assert(x_test.n_cols == hparams.test_size &&
             x_test.n_rows == hparams.input_size);
      assert(label_test.size() == hparams.test_size);
      assert(y_test.n_cols == x_test.n_cols &&
             y_test.n_cols == hparams.test_size);
      assert(y_test.n_rows == hparams.num_classes);
      read_mnist(file_test_images, x_test, hparams.test_size);
      read_mnist_label(file_test_labels, label_test, hparams.test_size);
      label_to_y(label_test, hparams.num_classes, y_test);
    }
    std::cout << "Size of testing set =   " << x_test.n_cols << std::endl;
  }

  std::vector<int> H = {hparams.input_size, hparams.hidden_size,
                        hparams.num_classes};
  // Initialize a two-layer neural network on the CPU
  NeuralNetwork nn_seq(H);

  if (cpu && rank == 0)
  {
    // Train on the CPU
    train(nn_seq, X, y, hparams);

    if (hparams.test_size > 0)
    {
      arma::Row<nn_real> label_pred;
      predict(nn_seq, x_test, label_pred);
      nn_real prec = precision(label_pred, label_test);
      printf("Precision on testing set for sequential training = %20.16f\n", prec);
    }
  }

  // Initialize a two-layer neural network on the GPU
  NeuralNetwork nn_par(H);
  DataParallelNeuralNetwork dpnn(nn_par, hparams.reg, hparams.learning_rate,
                                 hparams.batch_size, num_procs, rank);
  // Train on the GPU
  MPI_Barrier(MPI_COMM_WORLD);
  nvtxRangePushA("GPU parallel train");
  dpnn.train(nn_par, X, y, hparams);
  nvtxRangePop();
  MPI_Barrier(MPI_COMM_WORLD);

  // Calculate the precision of the model
  if (rank == 0 && hparams.test_size > 0)
  {
    dpnn.to_cpu(nn_par);
    arma::Row<nn_real> label_pred;
    predict(nn_par, x_test, label_pred);
    nn_real prec = precision(label_pred, label_test);
    printf("Precision on testing set for parallel training =   %20.16f\n", prec);
  }

  // Check result of parallel training using the GPUs
  if (cpu && rank == 0)
  {
    for (int i = 0; i < nn_par.num_layers; i++)
    {
      float rel_err_W = arma::norm(nn_seq.W[i] - nn_par.W[i], "fro") / arma::norm(nn_seq.W[i], "fro");
      float rel_err_b = arma::norm(nn_seq.b[i] - nn_par.b[i]) / arma::norm(nn_seq.b[i]);

      printf("Rel. err. in W[%1d]: %g\n", i, rel_err_W);
      printf("Rel. err. in b[%1d]: %g\n", i, rel_err_b);
      EXPECT_LT(rel_err_W, TOL);
      EXPECT_LT(rel_err_b, TOL);
    }
  }
}

// hyperparameters:
//   int N;
//   int test_size = 0;
//   int input_size;
//   int hidden_size;
//   int batch_size;
//   int num_epochs;
//   int num_classes;
//   nn_real reg;
//   nn_real learning_rate;
//   int debug;

TEST(gtestTrain, small1)
{
  hyperparameters hparams_small =
      hyperparameters{32, 0, 784, 32, 32, 1, 10, 1e-4, 0.01, 0};
  testTrain(hparams_small);
}

TEST(gtestTrain, small2)
{
  hyperparameters hparams_small =
      hyperparameters{512, 0, 784, 32, 32, 2, 10, 1e-4, 0.002, 0};
  testTrain(hparams_small);
}

TEST(gtestTrain, small3)
{
  hyperparameters hparams_small =
      hyperparameters{512, 0, 784, 32, 32, 16, 10, 1e-4, 0.002, 0};
  testTrain(hparams_small);
}

TEST(gtestTrain, medium)
{
  hyperparameters hparams_medium =
      hyperparameters{1087, 0, 784, 128, 64, 4, 10, 1e-4, 0.004, 0};
  testTrain(hparams_medium);
}

TEST(gtestTrain, large)
{
  hyperparameters hparams_large =
      hyperparameters{1087, 0, 784, 1013, 67, 4, 10, 1e-4, 0.002, 0};
  testTrain(hparams_large);
}

TEST(gtestTrain, custom)
{
  hyperparameters hparams_custom =
      hyperparameters{38400, 0, 784, 512, 3200, 1, 10, 1e-4, 1e-4, 0};
  testTrain(hparams_custom);
}

// hyperparameters:
//   int N;
//   int test_size = 0;
//   int input_size;
//   int hidden_size;
//   int batch_size;
//   int num_epochs;
//   int num_classes;
//   nn_real reg;
//   nn_real learning_rate;
//   int debug;

// Number of images in training file: 60000
// Number of images in test file: 10000

// Train on both CPU and GPU; difference between both models is computed
TEST(gtestProfile, nsys1)
{
  hyperparameters hparams;
  hparams = hyperparameters{60000, 10000, 784, 512, 3000, 1,
                            10, 1e-4, 1e-2, 0};
  profileTrain(hparams);
}

// Train on the GPU only
TEST(gtestProfile, nsys2)
{
  hyperparameters hparams;
  hparams = hyperparameters{60000, 10000, 784, 512, 3000, 1,
                            10, 1e-4, 1e-2, 0};
  profileTrain(hparams, false);
}

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  int run_tests = RUN_ALL_TESTS();
  MPI_Finalize();
  return run_tests;
}