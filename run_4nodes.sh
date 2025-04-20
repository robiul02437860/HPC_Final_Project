#!/bin/bash
#SBATCH --job-name=4node_hybrid       # A descriptive name for your job
#SBATCH --time=0-00:01:00             # Wall-clock time limit (DD-HH:MM:SS).
#SBATCH --nodes=4                     # Run on 4 nodes
#SBATCH --ntasks=4                    # Overall there are 4 tasks
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                  # Total GPUs requested
#SBATCH -p lonepeak-gpu               # ADJUST THIS if your partition is different
#SBATCH -A lonepeak-gpu               # ADJUST THIS if your account is different
#SBATCH -o sout/slurmjob-%j.out-%N    # Output file per node
#SBATCH -e sout/slurmjob-%j.err-%N    # Error file per node


# --- Load Modules for Runtime ---
# Load the exact same module stack you used for successful compilation
module purge

# --- Navigate to Project Directory ---
# Use the full path on CHPC storage
# ADJUST THIS path!
cd ~/project
module load gcc
module load openmpi
module load cuda

mpirun -np ${SLURM_NTASKS} ./mpi_cuda_split_merge data/input.pgm results/output_dist_mem_gpu.pgm
