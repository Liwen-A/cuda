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
                arma::Mat<nn_real> &y)
{
  y.set_size(num_classes, label.size());
  y.fill(0);
  for (int i = 0; i < label.size(); ++i)
  {
    assert(label(i) >= 0);
    assert(label(i) < num_classes);
    y(label(i), i) = 1;
  }
}

// Compute the precision of the prediction
nn_real precision(arma::Row<nn_real> vec1, arma::Row<nn_real> vec2)
{
  return arma::accu(vec1 == vec2) / (nn_real)vec1.size();
}

int get_num_batches(int N, int batch_size)
{
  return (N + batch_size - 1) / batch_size;
}

int get_batch_size(int N, int batch_size, int batch)
{
  int num_batches = get_num_batches(N, batch_size);
  return (batch == num_batches - 1) ? N - batch_size * batch : batch_size;
}

int get_mini_batch_size(int batch_size, int num_procs, int rank)
{
  int mini_batch_size = batch_size / num_procs;
  return rank < batch_size % num_procs ? mini_batch_size + 1 : mini_batch_size;
}

int get_offset(int batch_size, int num_procs, int rank)
{
  const int blockSize = (batch_size + num_procs - 1) / num_procs;
  const int nBlockFull = batch_size - num_procs * (blockSize - 1);
  return rank * (blockSize - 1) + std::min(rank, nBlockFull);
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                             CPU implementation                             //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

NeuralNetwork::NeuralNetwork(std::vector<int> &_H)
{
  W.resize(num_layers);
  b.resize(num_layers);
  H = _H;

  assert(H.size() == 3);

  for (int i = 0; i < num_layers; i++)
  {
    arma::arma_rng::set_seed(arma::arma_rng::seed_type(i));
    assert(i + 1 < 3);
    W[i] = 0.01 * arma::randn<arma::Mat<nn_real>>(H[i + 1], H[i]);
    b[i].zeros(H[i + 1]);
  }
}

void sigmoid(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1. / (1. + arma::exp(-mat));
}

void softmax(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
}

void forward(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             struct cache &cache)
{
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

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             arma::Row<nn_real> &label)
{
  struct cache fcache;
  forward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i)
  {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

nn_real norms(NeuralNetwork &nn)
{
  nn_real norm_sum = 0;
  for (int i = 0; i < nn.num_layers; ++i)
  {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }
  return norm_sum;
}

nn_real loss(NeuralNetwork &nn, const arma::Mat<nn_real> &yc,
             const arma::Mat<nn_real> &y, nn_real reg)
{
  int N = yc.n_cols;
  nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  nn_real data_loss = ce_sum / N;
  nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real loss = data_loss + reg_loss;
  return loss;
}

void backward(NeuralNetwork &nn, const arma::Mat<nn_real> &y, nn_real reg,
              const struct cache &bpcache, struct grads &bpgrads)
{
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
           const arma::Mat<nn_real> &y, hyperparameters &hparams)
{
  int N = X.n_cols;
  int iter = 0;
  bool print_flag;

  assert(X.n_cols == y.n_cols);

  int num_batches = get_num_batches(N, hparams.batch_size);

  for (int epoch = 0; epoch < hparams.num_epochs; ++epoch)
  {
    int batch_start = 0;
    for (int batch = 0; batch < num_batches; ++batch)
    {
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
                   ((iter % std::max(1, ((hparams.num_epochs * num_batches) / 4))) == 0);

      if (print_flag)
      {
        printf("Seq loss at iteration %5d of epoch %5d/%5d = %25.20f\n", iter,
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

GPUGrads::GPUGrads(const NeuralNetwork &nn)
{
  total_elements = 0;
  for (int i = 0; i < nn.num_layers; i++)
  {
    total_elements += nn.W[i].n_rows * nn.W[i].n_cols;
    total_elements += nn.b[i].n_rows * nn.b[i].n_cols;
  }
  gpu_mem_pool =
      std::make_shared<DeviceAllocator>(total_elements);
  nn_real *gpu_mem_pool_start = gpu_mem_pool.get()->memptr();

  // Allocating pinned CPU memory for the MPI All reduce
  size_t bytes = total_elements * sizeof(nn_real);
  checkCudaErrors(cudaMallocHost(&h_pinned, bytes));

  size_t offset = 0;
  dW.resize(nn.num_layers);
  db.resize(nn.num_layers);
  for (int i = 0; i < nn.num_layers; i++)
  {
    assert(offset < total_elements); // Check if we are within bounds
    dW[i] = DeviceMatrix(gpu_mem_pool_start + offset, nn.W[i].n_rows,
                         nn.W[i].n_cols);
    offset += nn.W[i].n_rows * nn.W[i].n_cols;
    assert(offset < total_elements); // Check if we are within bounds
    db[i] = DeviceMatrix(gpu_mem_pool_start + offset, nn.b[i].n_rows,
                         nn.b[i].n_cols);
    offset += nn.b[i].n_rows * nn.b[i].n_cols;
  }
}

GPUGrads::~GPUGrads()
{
  checkCudaErrors(cudaFreeHost(h_pinned));
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
      num_procs(num_procs)
{
  assert(H.size() == 3);
  assert(nn.num_layers == 2);

  this->reg /= num_procs;
  int mini_batch_size = get_mini_batch_size(batch_size, num_procs, rank);

  cache.X = DeviceMatrix(H[0], mini_batch_size);
  cache.a.resize(nn.num_layers);
  cache.z.resize(nn.num_layers);
  for (int i = 0; i < nn.num_layers; i++)
  {
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

void DataParallelNeuralNetwork::forward(const DeviceMatrix &X)
{
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

nn_real DataParallelNeuralNetwork::loss(const DeviceMatrix &y, nn_real weight)
{
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
  nn_real total_loss = data_loss + reg_loss;

  if (num_procs > 1)
  {
    nn_real global_loss;
    MPI_SAFE_CALL(MPI_Reduce(&total_loss, &global_loss, 1, MPI_FP, MPI_SUM, 0,
                             MPI_COMM_WORLD));
    return global_loss;
  }
  else
  {
    return total_loss;
  }
}

void DataParallelNeuralNetwork::backward(const DeviceMatrix &y,
                                         nn_real grad_weight)
{
  DeviceMatrix diff = cache.a[1]; // shallow copy
  DElemArith(diff, y, grad_weight, -1.0);

  W[1].to_gpu(grads.dW[1]);
  tiledGEMM(diff, false, cache.a[0], true, grads.dW[1], 1.0, reg);

  DSum(diff, grads.db[1], 1., 1);

  DeviceMatrix da1 = cache.z[0]; // cache.z[0] is unused. We can overwrite it
  tiledGEMM(W[1], true, diff, false, da1, 1.0, 0.0);

  DeviceMatrix dz1 = da1; // shallow copy
  DSigmoidBackprop(da1, cache.a[0], dz1);

  W[0].to_gpu(grads.dW[0]);
  tiledGEMM(dz1, false, cache.X, true, grads.dW[0], 1.0, reg);

  DSum(dz1, grads.db[0], 1., 1);

  if (num_procs > 1)
  {
    size_t bytes = grads.total_elements * sizeof(nn_real);
    cudaDeviceSynchronize();
    nvtxRangePushA("cudaMemcpyDeviceToHost");
    checkCudaErrors(
        cudaMemcpy(grads.h_pinned, grads.gpu_mem_pool.get()->memptr(), bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePushA("MPI_Allreduce");
    MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE,
                                grads.h_pinned,
                                grads.total_elements, MPI_FP, MPI_SUM,
                                MPI_COMM_WORLD));
    nvtxRangePop();
    nvtxRangePushA("cudaMemcpyHostToDevice");
    checkCudaErrors(
        cudaMemcpy(grads.gpu_mem_pool.get()->memptr(), grads.h_pinned, bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();
  }
}

void DataParallelNeuralNetwork::step()
{
  assert(W.size() == b.size());
  for (int i = 0; i < W.size(); i++)
  {
    DElemArith(W[i], grads.dW[i], 1.0, -lr);
    DElemArith(b[i], grads.db[i], 1.0, -lr);
  }
}

void DataParallelNeuralNetwork::train(NeuralNetwork &nn, arma::Mat<nn_real> &X,
                                      arma::Mat<nn_real> &y,
                                      hyperparameters &hparams)
{
  nvtxRangePushA("train setup");

  assert(reg >= 0);
  assert(lr > 0);
  assert(hparams.num_epochs >= 0);
  assert(hparams.batch_size > 0);

  int rank;
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (rank == 0)
  {
    assert(X.n_rows == H[0] && X.n_cols == hparams.N);
    assert(y.n_rows == H[2] && y.n_cols == X.n_cols);
  }

  int N = (rank == 0) ? X.n_cols : 0;
  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));
  assert(N > 0);

  if (num_procs > 1)
  {
    MPI_SAFE_CALL(MPI_Bcast(X.memptr(), X.n_rows * X.n_cols, MPI_FP, 0,
                            MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Bcast(y.memptr(), y.n_rows * y.n_cols, MPI_FP, 0,
                            MPI_COMM_WORLD));
  }

  int iter = 0;
  const int num_batches = get_num_batches(N, hparams.batch_size);
  bool print_flag;

  arma::Mat<nn_real> X_mb;
  arma::Mat<nn_real> y_mb;
  DeviceMatrix dX_mb;
  DeviceMatrix dy_mb;

  nvtxRangePop();

  for (int epoch = 0; epoch < hparams.num_epochs; ++epoch)
  {
    int batch_start = 0;
    for (int batch = 0; batch < num_batches; ++batch)
    {
      cudaDeviceSynchronize();
      nvtxRangePushA("batch iteration");
      nvtxRangePushA("copying images host -> device");

      int batch_size = get_batch_size(N, hparams.batch_size, batch);
      int batch_end = batch_start + batch_size;
      int mini_batch_size = get_mini_batch_size(batch_size, num_procs, rank);
      int offset = get_offset(batch_size, num_procs, rank);
      int mini_batch_start = batch_start + offset;
      int mini_batch_end = mini_batch_start + mini_batch_size;
      assert(0 <= mini_batch_start && mini_batch_end <= batch_end);

      if (mini_batch_size == 0)
      {
        X_mb = arma::Mat<nn_real>(H[0], cache.z[0].n_cols, arma::fill::zeros);
        y_mb = arma::Mat<nn_real>(H[2], cache.z[0].n_cols, arma::fill::zeros);

        dy_mb = DeviceMatrix(y_mb);

        // Set all activations to zero instead of a forward pass
        cache.X = DeviceMatrix(X_mb);
        for (int i = 0; i < nn.num_layers; i++)
        {
          arma::Mat<nn_real> temp_z = arma::Mat<nn_real>(
              W[i].n_rows, cache.z[i].n_cols, arma::fill::zeros);
          arma::Mat<nn_real> temp_a = arma::Mat<nn_real>(
              W[i].n_rows, cache.a[i].n_cols, arma::fill::zeros);
          cache.z[i] = DeviceMatrix(temp_z);
          cache.a[i] = DeviceMatrix(temp_a);
        }
        arma::Mat<nn_real> temp_yc = arma::Mat<nn_real>(
            cache.yc.n_rows, cache.yc.n_cols, arma::fill::ones);
        cache.yc = DeviceMatrix(temp_yc);
      }
      else
      {
        assert(mini_batch_size > 0);
        X_mb = arma::Mat<nn_real>(X.cols(mini_batch_start, mini_batch_end - 1));
        y_mb = arma::Mat<nn_real>(y.cols(mini_batch_start, mini_batch_end - 1));

        if (X_mb.n_cols != cache.z[0].n_cols)
        {
          cache.X.set_n_cols(X_mb.n_cols);
          for (int i = 0; i < nn.num_layers; i++)
          {
            cache.z[i].set_n_cols(X_mb.n_cols);
            cache.a[i].set_n_cols(X_mb.n_cols);
          }
          cache.yc.set_n_cols(X_mb.n_cols);
        }

        dX_mb = DeviceMatrix(X_mb);
        dy_mb = DeviceMatrix(y_mb);

        cudaDeviceSynchronize();
        nvtxRangePop();

        nvtxRangePushA("forward");
        this->forward(dX_mb);
        cudaDeviceSynchronize();
        nvtxRangePop();
      }

      const int printout_frequency = 4;
      const int printout_period = std::max(1, (hparams.num_epochs * num_batches) / printout_frequency);
      print_flag = (hparams.debug == 1) && ((iter % printout_period) == 0);

      if (print_flag)
      {
        nn_real loss_iter = this->loss(dy_mb, 1.0 / batch_size);
        if (rank == 0)
        {
          printf("Par loss at iteration %5d of epoch %5d/%5d = %25.20f\n", iter,
                 epoch, hparams.num_epochs, loss_iter);
        }
      }

      cudaDeviceSynchronize();
      nvtxRangePushA("backward");
      this->backward(dy_mb, 1.0 / batch_size);
      cudaDeviceSynchronize();
      nvtxRangePop();

      nvtxRangePushA("step");
      this->step();
      cudaDeviceSynchronize();
      nvtxRangePop();

      batch_start = batch_end;
      iter++;
      nvtxRangePop();
    }
  }
}

void DataParallelNeuralNetwork::to_cpu(NeuralNetwork &nn)
{
  for (int i = 0; i < nn.num_layers; i++)
  {
    assert(i + 1 < 3);
    assert(W[i].n_rows == nn.W[i].n_rows && W[i].n_cols == nn.W[i].n_cols);
    assert(b[i].n_rows == nn.b[i].n_rows && b[i].n_cols == nn.b[i].n_cols);
    W[i].to_cpu(nn.W[i]);
    b[i].to_cpu(nn.b[i]);
  }
}