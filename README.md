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

# MPI + CUDA
To run the distributed memory gpu first load the modules:
```bash
  module purge
  module load gcc openmpi cuda
```

Make the distributed gpu code:

```bash
  make clean
  make dist_mem_gpu
```

To run the distributed memory code on 2 nodes (on lonepeak):
```bash 
  sbatch run_2nodes.sh
```

The outputs will be on ```sout/```.
The output file is located in ```results/output_dist_mem_gpu.pgm```.

# Cleaning Environment
To clean the environment run 
``` bash
  make clean
```