#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/image_io.h"

#define DIFF_THRESHOLD 10
#define BLOCK_SIZE 16

__global__ void init_labels(uint8_t *img, int *labels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        labels[y * width + x] = y * width + x;
    }
}

__global__ void merge_labels(uint8_t *img, int *labels, int width, int height, int *changed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width - 1 && y < height - 1) {
        int idx = y * width + x;
        int right = idx + 1;
        int down = idx + width;
        if (abs(img[idx] - img[right]) < DIFF_THRESHOLD) {
            int min_label = min(labels[idx], labels[right]);
            if (labels[right] != min_label) {
                labels[right] = min_label;
                *changed = 1;
            }
            if (labels[idx] != min_label) {
                labels[idx] = min_label;
                *changed = 1;
            }
        }
        if (abs(img[idx] - img[down]) < DIFF_THRESHOLD) {
            int min_label = min(labels[idx], labels[down]);
            if (labels[down] != min_label) {
                labels[down] = min_label;
                *changed = 1;
            }
            if (labels[idx] != min_label) {
                labels[idx] = min_label;
                *changed = 1;
            }
        }
    }
}

void exchange_boundaries(int *labels, int width, int height, int rank, int size, MPI_Comm comm) {
    MPI_Status status;
    if (rank != 0) {
        MPI_Sendrecv(labels, width, MPI_INT, rank - 1, 0,
                     labels - width, width, MPI_INT, rank - 1, 0, comm, &status);
    }
    if (rank != size - 1) {
        MPI_Sendrecv(labels + (height - 1) * width, width, MPI_INT, rank + 1, 0,
                     labels + height * width, width, MPI_INT, rank + 1, 0, comm, &status);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            printf("Usage: %s input.pgm output.pgm\n", argv[0]);
        MPI_Finalize();
        return -1;
    }

    Image *img = NULL;
    if (rank == 0)
        img = read_pgm(argv[1]);

    int width, total_height;
    if (rank == 0) {
        width = img->width;
        total_height = img->height;
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int height_per_proc = total_height / size;

    uint8_t *local_data = (uint8_t*)malloc(width * height_per_proc * sizeof(uint8_t));
    MPI_Scatter(img ? img->data : NULL, width * height_per_proc, MPI_UINT8_T,
                local_data, width * height_per_proc, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    uint8_t *d_img;
    int *d_labels, *d_changed;
    int changed;
    cudaMalloc(&d_img, width * height_per_proc * sizeof(uint8_t));
    cudaMalloc(&d_labels, width * height_per_proc * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_img, local_data, width * height_per_proc * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height_per_proc + BLOCK_SIZE - 1) / BLOCK_SIZE);

    init_labels<<<grid, block>>>(d_img, d_labels, width, height_per_proc);
    cudaDeviceSynchronize();

    do {
        changed = 0;
        cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);

        merge_labels<<<grid, block>>>(d_img, d_labels, width, height_per_proc, d_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        exchange_boundaries(d_labels, width, height_per_proc, rank, size, MPI_COMM_WORLD);

    } while (changed);

    int *labels = (int*)malloc(width * height_per_proc * sizeof(int));
    cudaMemcpy(labels, d_labels, width * height_per_proc * sizeof(int), cudaMemcpyDeviceToHost);

    uint8_t *output_data = (uint8_t*)malloc(width * height_per_proc * sizeof(uint8_t));
    for (int i = 0; i < width * height_per_proc; i++) {
        output_data[i] = labels[i] % 256;
    }

    if (rank == 0) {
        uint8_t *full_output = (uint8_t*)malloc(width * total_height * sizeof(uint8_t));
        MPI_Gather(output_data, width * height_per_proc, MPI_UINT8_T,
                   full_output, width * height_per_proc, MPI_UINT8_T, 0, MPI_COMM_WORLD);
        img->data = full_output;
        write_pgm(argv[2], img);
        free_image(img);
    } else {
        MPI_Gather(output_data, width * height_per_proc, MPI_UINT8_T,
                   NULL, 0, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    }

    free(local_data);
    free(labels);
    free(output_data);
    cudaFree(d_img);
    cudaFree(d_labels);
    cudaFree(d_changed);

    MPI_Finalize();
    return 0;
}
