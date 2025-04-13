CC = gcc
CFLAGS = -Wall -O2
NVCC = nvcc
MPICC = mpicc

SRC_DIR = src
COMMON_DIR = $(SRC_DIR)/common
# MPI_INC = -I/usr/lib/x86_64-linux-gnu/openmpi/include

.PHONY: all clean serial shared_mem_cpu cuda_gpu dist_mem_cpu dist_mem_gpu

all: serial shared_mem_cpu cuda_gpu dist_mem_cpu dist_mem_gpu

# Serial Implementation
serial:
	$(CC) $(CFLAGS) $(COMMON_DIR)/image_io.c $(SRC_DIR)/serial/serial_split_merge.c -o serial_split_merge

# OpenMP Implementation
shared_mem_cpu:
	$(CC) $(CFLAGS) -fopenmp $(COMMON_DIR)/image_io.c $(SRC_DIR)/shared_mem_cpu/omp_split_merge.c -o omp_split_merge

# CUDA Implementation
cuda_gpu:
	$(NVCC) -O2 $(SRC_DIR)/cuda_gpu/cuda_split_merge.cu $(COMMON_DIR)/image_io.c -o cuda_split_merge

# MPI Implementation
dist_mem_cpu:
	$(MPICC) -O2 $(COMMON_DIR)/image_io.c $(SRC_DIR)/dist_mem_cpu/mpi_split_merge.c -o mpi_split_merge

# MPI + CUDA Hybrid Implementation
MPI_INC = -I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB = -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi

dist_mem_gpu: mpi_cuda_split_merge.o image_io.o
	$(NVCC) -O2 mpi_cuda_split_merge.o image_io.o -o mpi_cuda_split_merge -lcudart $(MPI_LIB)

mpi_cuda_split_merge.o:
	$(NVCC) -O2 $(MPI_INC) -c $(SRC_DIR)/dist_mem_gpu/mpi_cuda_split_merge.cu -o mpi_cuda_split_merge.o

image_io.o:
	$(CC) $(CFLAGS) -c $(COMMON_DIR)/image_io.c -o image_io.o


clean:
	rm -f serial_split_merge omp_split_merge cuda_split_merge mpi_split_merge mpi_cuda_split_merge
