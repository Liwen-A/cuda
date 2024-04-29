#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo "Starting at `date`"
echo

echo
echo Output from main
echo ----------------
./main 

./run.sh
