#ifndef TWO_LAYER_MLP_H_
#define TWO_LAYER_MLP_H_

#include <cmath>
#include <iomanip>
#include <armadillo>
#include <cassert>

#include "common.h"
#include "gpu_func.h"
#include "nvToolsExt.h"

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                                 Utilities                                  //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

#define ASSERT_MAT_SAME_SIZE(mat1, mat12) \
  assert(mat1.n_rows == mat2.n_rows && mat1.n_cols == mat2.n_cols)

#define MPI_SAFE_CALL(call)                                                  \
  do                                                                         \
  {                                                                          \
    int err = call;                                                          \
    if (err != MPI_SUCCESS)                                                  \
    {                                                                        \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

/**
 * Converts a row vector of labels into a matrix of horizontally-stacked one-hot
 * column vectors.
 *
 * @param label : row vector of labels.
 * @param C : Number of classes.
 * @param y : matrix of horizontally-stacked one-hot column vectors.
 */
void label_to_y(arma::Row<nn_real> label, int C, arma::Mat<nn_real> &y);

/**
 * Compute the precision of the prediction
 * @param vec1 : label 1
 * @param vec2 : label 2
 */
nn_real precision(arma::Row<nn_real> vec1, arma::Row<nn_real> vec2);

int get_num_batches(int N, int batch_size);

int get_batch_size(int N, int batch_size, int batch);

int get_mini_batch_size(int batch_size, int num_procs, int rank);

int get_offset(int batch_size, int num_procs, int rank);

class hyperparameters
{
public:
  int N;
  int test_size = 0;
  int input_size;
  int hidden_size;
  int batch_size;
  int num_epochs;
  int num_classes;
  nn_real reg;
  nn_real learning_rate;
  int debug;

  hyperparameters() { reset(); }

  hyperparameters(int N, int test_size, int input_size,
                  int hidden_size, int batch_size,
                  int num_epochs, int num_classes,
                  nn_real reg, nn_real learning_rate, int debug)
      : N(N), test_size(test_size), input_size(input_size), hidden_size(hidden_size), batch_size(batch_size), num_epochs(num_epochs),
        num_classes(num_classes), reg(reg), learning_rate(learning_rate),
        debug(debug)
  {
    assert(N > 0);
    assert(test_size >= 0);
    assert(input_size > 0);
    assert(hidden_size > 0);
    assert(batch_size > 0);
    assert(num_epochs > 0);
    assert(num_classes > 0);
    assert(reg > 0);
    assert(learning_rate > 0);
    assert(debug == 0 || debug == 1);

    // printf("N = %d, test_size = %d, input_size = %d\n", N, test_size, input_size);
    // printf("hidden_size = %d, batch_size = %d, num_epochs = %d, num_classes = %d\n", hidden_size, batch_size, num_epochs, num_classes);
    // printf("reg = %g, learning_rate = %g, debug = %1d\n", reg, learning_rate, debug);
  }

  void reset()
  {
    N = -1;
    test_size = 0;
    input_size = -1;
    hidden_size = -1;
    batch_size = -1;
    num_epochs = -1;
    num_classes = -1;
    reg = -1.;
    learning_rate = -1.;
    debug = -1;
  }
};

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                             CPU implementation                             //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

class NeuralNetwork
{
public:
  const int num_layers = 2;
  // H[i] is the number of neurons in layer i (where i=0 implies input layer)
  std::vector<int> H;
  // W[i] are the weights of the i^th layer
  std::vector<arma::Mat<nn_real>> W;
  // b[i] is the row vector biases of the i^th layer
  std::vector<arma::Col<nn_real>> b;

  NeuralNetwork(std::vector<int> &_H);
};

void sigmoid(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2);

void softmax(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2);

struct cache
{
  arma::Mat<nn_real> X;
  std::vector<arma::Mat<nn_real>> z;
  std::vector<arma::Mat<nn_real>> a;
  arma::Mat<nn_real> yc;
};

void forward(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             struct cache &bpcache);

nn_real norms(NeuralNetwork &nn);

nn_real loss(NeuralNetwork &nn, const arma::Mat<nn_real> &yc,
             const arma::Mat<nn_real> &y, nn_real reg);

struct grads
{
  std::vector<arma::Mat<nn_real>> dW;
  std::vector<arma::Col<nn_real>> db;
};

/**
 * Computes the derivative of the loss w.r.t each parameter. Must be called
 * after forward since it uses the bpcache.
 *
 * @param y : C x N one-hot column vectors
 * @param bpcache : Output of forward
 * @param bpgrads: Populates the gradients for each param
 */
void backward(NeuralNetwork &nn, const arma::Mat<nn_real> &y, nn_real reg,
              const struct cache &bpcache, struct grads &bpgrads);

void train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
           const arma::Mat<nn_real> &y, hyperparameters &hparams);

/**
 * Calculate the labels given a trained neural network and some input X
 * @param nn: trained neural network
 * @param X: input data
 * @param label: output labels
 */
void predict(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             arma::Row<nn_real> &label);

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                             GPU implementation                             //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

/**
 * A struct that represents the GPU cache used in the forward and backward
 * passes of a neural network.
 */
struct GPUCache
{
  DeviceMatrix X;              // The input data matrix
  std::vector<DeviceMatrix> z; // The intermediate matrices z
  std::vector<DeviceMatrix> a; // The intermediate matrices a
  DeviceMatrix yc;             // The output matrix yc
};

/**
 * A struct that represents the gradients of a neural network in GPU memory.
 * To reduce the number of MPI_Allreduce calls, allocate all gradients in a big
 * chunk of contiguous memory.
 */
struct GPUGrads
{
  std::vector<DeviceMatrix> dW;
  // The gradients of the weights for each layer
  std::vector<DeviceMatrix> db;
  // The gradients of the biases for each layer
  std::shared_ptr<DeviceAllocator> gpu_mem_pool;
  // GPU memory for gradients
  size_t total_elements;
  // Number of elements allocated for storing the GPU gradients.
  void *h_pinned;
  // Pinned CPU memory for MPI All reduce

  GPUGrads() = default;

  /**
   * Constructs a new gradient object for a NeuralNetwork.
   *
   * @param nn The NeuralNetwork object to construct from.
   */
  GPUGrads(const NeuralNetwork &nn);

  ~GPUGrads();
};

/**
 * Data Parallel neural network model
 */
class DataParallelNeuralNetwork
{
public:
  std::vector<int> H;          // The dimensions of the neural network
  std::vector<DeviceMatrix> W; // The weights of the neural network
  std::vector<DeviceMatrix> b; // The biases of the neural network
  GPUCache cache;              // The intermediate results of the forward pass
  GPUGrads grads;              // The gradients of the neural network
  nn_real reg;                 // The regularization parameter
  nn_real lr;                  // The learning rate
  int num_procs;               // The number of processes

  DataParallelNeuralNetwork() = default;
  /**
   * Constructs a new DataParallelGPUNeuralNetwork object.
   *
   * @param nn The CPU neural network model.
   * @param reg The regularization parameter.
   * @param batch_size The batch size for training.
   * @param num_procs The number of processes in the GPU cluster.
   */
  DataParallelNeuralNetwork(NeuralNetwork &nn, nn_real reg, nn_real lr,
                            int batch_size, int num_procs, int rank);

  /**
   * Performs forward propagation with the neural network model.
   *
   * @param X The input data matrix
   */
  void forward(const DeviceMatrix &X);

  /**
   * Calculates the loss of the neural network model.
   *
   * @param y The ground truth labels.
   * @param weight The weight to apply to the loss.
   * @return The loss of the model.
   */
  nn_real loss(const DeviceMatrix &y, nn_real weight);

  /**
   * Performs backward propagation on the neural network model.
   *
   * @param y The ground truth labels.
   * @param grad_weight The weight to apply to the gradients.
   */
  void backward(const DeviceMatrix &y, nn_real grad_weight);

  void step();

  void train(NeuralNetwork &nn, arma::Mat<nn_real> &X,
             arma::Mat<nn_real> &y, hyperparameters &hparams);

  /**
   * Copy the weights on the GPU to the CPU.
   */
  void to_cpu(NeuralNetwork &nn);
};

#endif // TWO_LAYER_MLP_H_