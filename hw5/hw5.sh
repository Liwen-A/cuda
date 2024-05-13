#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo "Starting at `date`"
echo
make

echo
echo Output from main_q2
echo ----------------
./main_q2

echo
echo Output from main_q3
echo ----------------
./main_q3

echo
echo Output from main_q4
echo ----------------
./main_q4

echo
echo Output from main_q5
echo ----------------
./main_q5