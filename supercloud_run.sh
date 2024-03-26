#!/bin/bash

# batch options
#SBATCH --exclusive

#loading the required module
module load anaconda/2023a

# run the script
python Parameter_sweep.py
