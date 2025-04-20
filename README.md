```bash
split-merge-segmentation/
│
├── src/
│   ├── serial/
│   ├── shared_mem_cpu/     # OpenMP or pthreads
│   ├── cuda_gpu/           # CUDA kernels
│   ├── dist_mem_cpu/       # MPI
│   ├── dist_mem_gpu/       # MPI + CUDA
│   └── common/             # Shared code (data loading, utilities)
│
├── data/                   # Sample input images
├── scripts/                # Validation & Visualization scripts (Python)
├── build/                  # Build instructions for CHPC
├── results/                # Output from experiments
├── README.md
└── Makefile                # Unified build system (Make/CMake)
```

# Cleaning the Environment

```bash
  module purge
  module load gcc openmpi cuda
  make clean
```

# Serial
```bash
  make serial
  ./serial_splitmerge data/input.pgm results/output_serial.pgm
```

# Shared Memory CPU
```bash
  make shared_mem_cpu
  ./omp_split_merge data/input.pgm results/output_shared_mem_cpu.pgm
```

# MPI
```bash
  make dist_mem_cpu
  ./mpi_split_merge data/input.pgm results/output_mpi_split_merge.pgm
```

# CUDA
```bash
  make cuda_gpu
  ./cuda_split_merge data/input.pgm results/output_cuda.pgm
```

# MPI + CUDA
Make the distributed gpu binaries:

```bash
  make dist_mem_gpu
```

To run the distributed memory code on 2 nodes (on lonepeak):
```bash 
  sbatch run_2nodes.sh
```

To run it on 3 or 4 nodes run (on lonepeak):
```bash
  sbatch run_3nodes.sh
  sbatch run_4nodes.sh
```

The outputs will be on ```sout/```.
The output file is located in ```results/output_dist_mem_gpu.pgm```.

# Cleaning Environment
To clean the environment run 
``` bash
  make clean
```