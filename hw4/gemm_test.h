#ifndef TESTS_H_
#define TESTS_H_

#include "common.h"
#include "gpu_func.h"
#include "gtest/gtest.h"

void BenchmarkGEMM(int M, int N, int K,
                   void (*myGEMM)(DeviceMatrix, DeviceMatrix, DeviceMatrix,
                                  nn_real, nn_real));

#endif
