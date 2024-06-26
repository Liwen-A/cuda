#define ARMA_ALLOW_FAKE_GCC
#include "mnist.h"

#include <armadillo>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <mpi.h>

#include "common.h"

int NUM_CLASSES = 10;

bool file_exists(const std::string &name)
{
  std::ifstream f(name.c_str());
  return f.good();
}

void read_mnist(std::string filename, arma::Mat<nn_real> &mat,
                int number_of_images)
{
  if (!file_exists(filename))
  {
    std::cerr << "File " << filename << " does not exist." << std::endl;
    MPI_Finalize();
    exit(0);
  }

  std::ifstream file(filename, std::ios::binary);

  if (file.is_open())
  {
    int magic_number = 0;
    int number_of_images_file = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    file.read((char *)&number_of_images_file, sizeof(number_of_images_file));
    assert(number_of_images <= reverse_int(number_of_images_file));
    file.read((char *)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);

    assert(mat.n_rows == n_rows * n_cols);
    assert(mat.n_cols == number_of_images);

    for (int i = 0; i < number_of_images; ++i)
    {
      for (int r = 0; r < n_rows; ++r)
      {
        for (int c = 0; c < n_cols; ++c)
        {
          unsigned char temp = 0;
          file.read((char *)&temp, sizeof(temp));
          mat(r * n_cols + c, i) = (nn_real)temp;
        }
      }
    }
  }
  else
  {
    std::cerr << "Unable to open file " << filename << std::endl;
    MPI_Finalize();
    exit(0);
  }
}

void read_mnist_label(std::string filename, arma::Row<nn_real> &vec,
                      int number_of_images)
{
  if (!file_exists(filename))
  {
    std::cerr << "File " << filename << " does not exist." << std::endl;
    MPI_Finalize();
    exit(0);
  }

  std::ifstream file(filename, std::ios::binary);

  if (file.is_open())
  {
    int magic_number;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    int number_of_images_file = 0;
    file.read((char *)&number_of_images_file, sizeof(number_of_images_file));
    assert(number_of_images <= reverse_int(number_of_images_file));

    assert(vec.n_cols == number_of_images);

    for (int i = 0; i < number_of_images; ++i)
    {
      unsigned char temp = 0;
      file.read((char *)&temp, sizeof(temp));
      vec(i) = (nn_real)temp;
      assert(vec(i) >= 0);
      assert(vec(i) < NUM_CLASSES);
    }
  }
  else
  {
    std::cerr << "Unable to open file " << filename << std::endl;
    MPI_Finalize();
    exit(0);
  }
}

int reverse_int(int i)
{
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}