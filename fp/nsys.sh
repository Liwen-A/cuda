#!/bin/bash
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

make
nsys profile -f true --trace cuda,mpi,nvtx mpirun -n 2 ./main --gtest_filter=gtestProfile.* --gtest_color=no