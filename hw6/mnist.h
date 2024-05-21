#ifndef MNIST_H_
#define MNIST_H_

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <cstring>

#include "common.h"

int reverse_int(int i);
void read_mnist(std::string filename, arma::Mat<nn_real> &mat, int no_images);
void read_mnist_label(std::string filename, arma::Row<nn_real> &vec,
                      int no_images);

#endif // MNIST_H_