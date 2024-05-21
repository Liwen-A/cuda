#define ARMA_ALLOW_FAKE_GCC
#include "neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <mpi.h>

#include <armadillo>
#include <iomanip>

#include "cublas_v2.h"
#include "gpu_func.h"

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                                 Utilities                                  //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

void label_to_y(arma::Row<nn_real> label, int num_classes,
                arma::Mat<nn_real> &y) {
  y.set_size(num_classes, label.size());
  y.fill(0);
  for (int i = 0; i < label.size(); ++i) {
    assert(label(i) >= 0);
    assert(label(i) < num_classes);
    y(label(i), i) = 1;
  }
}

int get_num_batches(int N, int batch_size) {
  return (N + batch_size - 1) / batch_size;
}

int get_batch_size(int N, int batch_size, int batch) {
  int num_batches = get_num_batches(N, batch_size);
  return (batch == num_batches - 1) ? N - batch_size * batch : batch_size;
}

int get_mini_batch_size(int batch_size, int num_procs, int rank) {
  int mini_batch_size = batch_size / num_procs;
  return rank < batch_size % num_procs ? mini_batch_size + 1 : mini_batch_size;
}

int get_offset(int batch_size, int num_procs, int rank) {
  const int blockSize = (batch_size + num_procs - 1) / num_procs;
  const int nBlockFull = batch_size - num_procs * (blockSize - 1);
  return rank * (blockSize - 1) + std::min(rank, nBlockFull);
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                             CPU implementation                             //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

NeuralNetwork::NeuralNetwork(std::vector<int> &_H) {
  W.resize(num_layers);
  b.resize(num_layers);
  H = _H;

  assert(H.size() == 3);

  for (int i = 0; i < num_layers; i++) {
    arma::arma_rng::set_seed(arma::arma_rng::seed_type(i));
    assert(i + 1 < 3);
    W[i] = 0.01 * arma::randn<arma::Mat<nn_real>>(H[i + 1], H[i]);
    b[i].zeros(H[i + 1]);
  }
}

void sigmoid(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2) {
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1. / (1. + arma::exp(-mat));
}

void softmax(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2) {
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
}

void forward(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             struct cache &cache) {
  cache.z.resize(2);
  cache.a.resize(2);

  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<nn_real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<nn_real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

nn_real norms(NeuralNetwork &nn) {
  nn_real norm_sum = 0;
  for (int i = 0; i < nn.num_layers; ++i) {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }
  return norm_sum;
}

nn_real loss(NeuralNetwork &nn, const arma::Mat<nn_real> &yc,
             const arma::Mat<nn_real> &y, nn_real reg) {
  int N = yc.n_cols;
  nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  nn_real data_loss = ce_sum / N;
  nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real loss = data_loss + reg_loss;
  return loss;
}

void backward(NeuralNetwork &nn, const arma::Mat<nn_real> &y, nn_real reg,
              const struct cache &bpcache, struct grads &bpgrads) {
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<nn_real> da1 = nn.W[1].t() * diff;

  arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

void train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
           const arma::Mat<nn_real> &y, hyperparameters &hparams) {
  int N = X.n_cols;
  int iter = 0;
  bool print_flag;

  assert(X.n_cols == y.n_cols);

  int num_batches = get_num_batches(N, hparams.batch_size);

  for (int epoch = 0; epoch < hparams.num_epochs; ++epoch) {
    int batch_start = 0;
    for (int batch = 0; batch < num_batches; ++batch) {
      int last_col = batch_start + get_batch_size(N, hparams.batch_size, batch);
      assert(last_col <= X.n_cols);
      assert(last_col <= y.n_cols);
      assert(last_col > batch_start);
      assert(batch < num_batches - 1 || last_col == X.n_cols);

      arma::Mat<nn_real> X_batch = X.cols(batch_start, last_col - 1);
      arma::Mat<nn_real> y_batch = y.cols(batch_start, last_col - 1);

      struct cache bpcache;
      forward(nn, X_batch, bpcache);

      struct grads bpgrads;
      backward(nn, y_batch, hparams.reg, bpcache, bpgrads);

      print_flag = (hparams.debug == 1) && 
                   ((iter % ((hparams.num_epochs * num_batches) / 4)) == 0);

      if (print_flag) {
        printf("Seq loss at iteration %d of epoch %d/%d = %25.20f\n", iter,
               epoch, hparams.num_epochs,
               loss(nn, bpcache.yc, y_batch, hparams.reg));
      }

      // Optimizer step
      for (int i = 0; i < nn.W.size(); ++i)
        nn.W[i] -= hparams.learning_rate * bpgrads.dW[i];

      for (int i = 0; i < nn.b.size(); ++i)
        nn.b[i] -= hparams.learning_rate * bpgrads.db[i];

      batch_start = last_col;
      iter++;
    }
  }
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                             GPU implementation                             //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

GPUGrads::GPUGrads(const NeuralNetwork &nn) {
  total_elements = 0;
  for (int i = 0; i < nn.num_layers; i++) {
    total_elements += nn.W[i].n_rows * nn.W[i].n_cols;
    total_elements += nn.b[i].n_rows * nn.b[i].n_cols;
  }
  gpu_mem_pool =
      std::make_shared<DeviceAllocator>(total_elements);
  nn_real *gpu_mem_pool_start = gpu_mem_pool.get()->memptr();

  size_t offset = 0;
  dW.resize(nn.num_layers);
  db.resize(nn.num_layers);
  for (int i = 0; i < nn.num_layers; i++) {
    assert(offset < total_elements);  // Check if we are within bounds
    dW[i] = DeviceMatrix(gpu_mem_pool_start + offset, nn.W[i].n_rows,
                         nn.W[i].n_cols);
    offset += nn.W[i].n_rows * nn.W[i].n_cols;
    assert(offset < total_elements);  // Check if we are within bounds
    db[i] = DeviceMatrix(gpu_mem_pool_start + offset, nn.b[i].n_rows,
                         nn.b[i].n_cols);
    offset += nn.b[i].n_rows * nn.b[i].n_cols;
  }
}

DataParallelNeuralNetwork::DataParallelNeuralNetwork(NeuralNetwork &nn,
                                                     nn_real reg, nn_real lr,
                                                     int batch_size,
                                                     int num_procs, int rank)
    : H(nn.H),
      W(nn.num_layers),
      b(nn.num_layers),
      grads(nn),
      reg(reg),
      lr(lr),
      num_procs(num_procs) {
  assert(H.size() == 3);
  assert(nn.num_layers == 2);

  reg /= num_procs;
  int mini_batch_size = get_mini_batch_size(batch_size, num_procs, rank);

  cache.X = DeviceMatrix(H[0], mini_batch_size);
  cache.a.resize(nn.num_layers);
  cache.z.resize(nn.num_layers);
  for (int i = 0; i < nn.num_layers; i++) {
    assert(i + 1 < 3);
    W[i] = DeviceMatrix(nn.W[i]);
    b[i] = DeviceMatrix(nn.b[i]);
    cache.z[i] = DeviceMatrix(nn.W[i].n_rows, mini_batch_size);
    cache.a[i] = DeviceMatrix(nn.W[i].n_rows, mini_batch_size);

    assert(W[i].n_rows == H[i + 1] && W[i].n_cols == H[i]);
    assert(b[i].n_rows == H[i + 1] && b[i].n_cols == 1);
  }
  cache.yc = DeviceMatrix(nn.W[1].n_rows, mini_batch_size);
}

void DataParallelNeuralNetwork::forward(const DeviceMatrix &X) {
  assert(X.n_rows == W[0].n_cols);
  X.to_gpu(cache.X);
  assert(cache.X.n_rows == X.n_rows && cache.X.n_cols == X.n_cols);
  int N = X.n_cols;

  // z1 = W[0] * X + b[0]
  assert(cache.z[0].n_rows == W[0].n_rows);
  assert(cache.z[0].n_cols == X.n_cols);
  DRepeatColVec(b[0], cache.z[0], N);
  tiledGEMM(W[0], false, X, false, cache.z[0], 1, 1);

  // a1 = sigmoid(z1)
  assert(cache.a[0].n_rows == cache.z[0].n_rows);
  assert(cache.a[0].n_cols == cache.z[0].n_cols);
  DSigmoid(cache.z[0], cache.a[0]);

  // z2 = W[1] * a1 + b[1]
  assert(cache.a[0].n_rows == W[1].n_cols);
  assert(cache.z[1].n_rows == W[1].n_rows);
  assert(cache.z[1].n_cols == cache.a[0].n_cols);
  DRepeatColVec(b[1], cache.z[1], N);
  tiledGEMM(W[1], false, cache.a[0], false, cache.z[1], 1, 1);

  // a2 = softmax(z2)
  assert(cache.a[1].n_rows = cache.z[1].n_rows);
  assert(cache.a[1].n_cols = cache.z[1].n_cols);
  DSoftmax(cache.z[1], cache.a[1], 0);
  
  cache.a[1].to_gpu(cache.yc);
}

nn_real DataParallelNeuralNetwork::loss(const DeviceMatrix &y, nn_real weight) {
  nn_real ce_sum;
  DeviceMatrix ce_sum_mat_gpu(1, 1);
  DCELoss(cache.yc, y, ce_sum_mat_gpu);
  arma::Mat<nn_real> ce_sum_mat_cpu(1, 1);
  ce_sum_mat_gpu.to_cpu(ce_sum_mat_cpu);
  ce_sum = ce_sum_mat_cpu(0, 0);

  nn_real W0_norm;
  DeviceMatrix W0_squared(W[0].n_rows, W[0].n_cols);
  DSquare(W[0], W0_squared);
  DeviceMatrix W0_squared_vec(W[0].n_rows, 1);
  DeviceMatrix W0_norm_mat_gpu(1, 1);
  DSum(W0_squared, W0_squared_vec, 1., 1);
  DSum(W0_squared_vec, W0_norm_mat_gpu, 1., 0);
  arma::Mat<nn_real> W0_norm_mat_cpu(1, 1);
  W0_norm_mat_gpu.to_cpu(W0_norm_mat_cpu);
  W0_norm = W0_norm_mat_cpu(0, 0);

  nn_real W1_norm;
  DeviceMatrix W1_squared(W[1].n_rows, W[1].n_cols);
  DSquare(W[1], W1_squared);
  DeviceMatrix W1_squared_vec(W[1].n_rows, 1);
  DeviceMatrix W1_norm_mat_gpu(1, 1);
  DSum(W1_squared, W1_squared_vec, 1., 1);
  DSum(W1_squared_vec, W1_norm_mat_gpu, 1., 0);
  arma::Mat<nn_real> W1_norm_mat_cpu(1, 1);
  W1_norm_mat_gpu.to_cpu(W1_norm_mat_cpu);
  W1_norm = W1_norm_mat_cpu(0, 0);

  nn_real data_loss = ce_sum * weight;
  nn_real reg_loss = 0.5 * reg * (W0_norm + W1_norm);

  /**
   * TODO: Return the sum of (data_loss + reg_loss) across all MPI processes.
   */
}

void DataParallelNeuralNetwork::backward(const DeviceMatrix &y,
                                         nn_real grad_weight) {
  DeviceMatrix diff = cache.a[1];  // shallow copy
  DElemArith(diff, y, grad_weight, -1.0);

  W[1].to_gpu(grads.dW[1]);
  tiledGEMM(diff, false, cache.a[0], true, grads.dW[1], 1.0, reg);

  DSum(diff, grads.db[1], 1., 1);

  DeviceMatrix da1 = cache.z[0];  // cache.z[0] is unused. We can overwrite it
  tiledGEMM(W[1], true, diff, false, da1, 1.0, 0.0);

  DeviceMatrix dz1 = da1;  // shallow copy
  DSigmoidBackprop(da1, cache.a[0], dz1);

  W[0].to_gpu(grads.dW[0]);
  tiledGEMM(dz1, false, cache.X, true, grads.dW[0], 1.0, reg);

  DSum(dz1, grads.db[0], 1., 1);

  /**
   * TODO: Average the elements of grads across all MPI processes. 
   * HINT: Use grads.gpu_mem_pool and grads.total_elements. The 'division' step
   * of computing the average has already been completed using grad_weight.
   */
}

void DataParallelNeuralNetwork::step() {
  assert(W.size() == b.size());
  for (int i = 0; i < W.size(); i++) {
    DElemArith(W[i], grads.dW[i], 1.0, -lr);
    DElemArith(b[i], grads.db[i], 1.0, -lr);
  }
}

void DataParallelNeuralNetwork::train(NeuralNetwork &nn, arma::Mat<nn_real> &X,
                                      arma::Mat<nn_real> &y, 
                                      hyperparameters &hparams) {
  /**
   * TODO: Implement this function.
   * HINT: See the CPU implementation "void train()" above. Get the rank of the 
   * current MPI process. If num_procs > 1, broadcast X and y from rank 0 MPI
   * process to other MPI processes. Use get_mini_batch_size() and get_offset()
   * to get the minibatch that should be processed by the current MPI process. 
   * The last minibatch in an epoch may be smaller than previous minibatches in 
   * the epoch; you can use DeviceMatrix::set_n_cols() to deal with this case.
   */
}

void DataParallelNeuralNetwork::to_cpu(NeuralNetwork &nn) {
  for (int i = 0; i < nn.num_layers; i++) {
    assert(i + 1 < 3);
    assert(W[i].n_rows == nn.W[i].n_rows && W[i].n_cols == nn.W[i].n_cols);
    assert(b[i].n_rows == nn.b[i].n_rows && b[i].n_cols == nn.b[i].n_cols);
    W[i].to_cpu(nn.W[i]);
    b[i].to_cpu(nn.b[i]);
  }
}