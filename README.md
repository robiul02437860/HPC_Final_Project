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