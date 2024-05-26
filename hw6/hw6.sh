#!/bin/bash
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo "Starting at `date`"
echo
make

echo
echo Output from main_q1
echo ----------------
mpirun -n 1 ./main_q1
# mpirun -n 2 ./main_q1
# mpirun -n 3 ./main_q1
# mpirun -n 4 ./main_q1

# echo
# echo Output from main_q2
# echo ----------------
# mpirun -n 1 ./main_q2
# mpirun -n 2 ./main_q2
# mpirun -n 4 ./main_q2