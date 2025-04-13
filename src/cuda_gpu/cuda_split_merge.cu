#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "../common/image_io.h"

#define VAR_THRESHOLD 500
#define DIFF_THRESHOLD 10

#define BLOCK_SIZE 16  // CUDA block size

// Kernel: Initialize labels (each pixel gets its own label)
__global__ void init_labels(uint8_t *img, int *labels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        labels[y * width + x] = y * width + x;  // unique label
    }
}

// Kernel: Merge neighboring pixels based on intensity difference
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

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s input.pgm output.pgm\n", argv[0]);
        return -1;
    }

    Image *img = read_pgm(argv[1]);
    int size = img->width * img->height;

    uint8_t *d_img;
    int *d_labels, *d_changed;
    int changed;

    cudaMalloc(&d_img, size * sizeof(uint8_t));
    cudaMalloc(&d_labels, size * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_img, img->data, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((img->width + BLOCK_SIZE - 1) / BLOCK_SIZE, (img->height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    init_labels<<<grid, block>>>(d_img, d_labels, img->width, img->height);
    cudaDeviceSynchronize();

    do {
        changed = 0;
        cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);

        merge_labels<<<grid, block>>>(d_img, d_labels, img->width, img->height, d_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (changed);

    int *labels = (int*)malloc(size * sizeof(int));
    cudaMemcpy(labels, d_labels, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        img->data[i] = labels[i] % 256;
    }

    write_pgm(argv[2], img);

    free(labels);
    free_image(img);
    cudaFree(d_img);
    cudaFree(d_labels);
    cudaFree(d_changed);
    return 0;
}
