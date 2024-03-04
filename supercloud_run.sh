#!/bin/bash

# batch options
#SBATCH --exclusive

#loading the required module
module load anaconda/2021b

# run the script
python Parameter_sweep.py
