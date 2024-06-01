#!/bin/bash
#SBATCH --cpus-per-task=16 ### Number of CPU cores (for OMP threads)
#SBATCH -p gpu-turing

OMP_NUM_THREADS=4
### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo "Starting at `date`"
echo
make

echo
echo Output from main_q1
echo ----------------
./main_q1

echo
echo Output from main_q2
echo ----------------
./main_q2

echo
echo Output from main_q2_part6
echo ----------------
./main_q2_part6
