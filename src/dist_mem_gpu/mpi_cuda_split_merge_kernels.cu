#include <stdint.h>
#include <cuda_runtime.h>

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


extern "C" void cuda_init_labels(uint8_t **d_img, int **d_labels, int **d_changed, uint8_t *local_data, int width, int height_per_proc)
{

    cudaMalloc(d_img, width * height_per_proc * sizeof(uint8_t));
    cudaMalloc(d_labels, width * height_per_proc * sizeof(int));
    cudaMalloc(d_changed, sizeof(int));

    cudaMemcpy(*d_img, local_data, width * height_per_proc * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height_per_proc + BLOCK_SIZE - 1) / BLOCK_SIZE);

    init_labels<<<grid, block>>>(*d_img, *d_labels, width, height_per_proc);
    cudaDeviceSynchronize();
}

extern "C" void cuda_merge_labels(int *d_changed, int *changed, uint8_t *d_img, int *d_labels, int *labels, int width, int height_per_proc)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height_per_proc + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaMemcpy(labels, d_labels, width * height_per_proc * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_changed, changed, sizeof(int), cudaMemcpyHostToDevice);

    merge_labels<<<grid, block>>>(d_img, d_labels, width, height_per_proc, d_changed);
    cudaDeviceSynchronize();

    cudaMemcpy(changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(labels, d_labels, width * height_per_proc * sizeof(int), cudaMemcpyDeviceToHost);
}

extern "C" void cuda_update_labels(int *labels, int *d_labels, int width, int height_per_proc) {
    cudaMemcpy(d_labels, labels, width * height_per_proc * sizeof(int), cudaMemcpyHostToDevice);
}

extern "C" void cuda_free(uint8_t *d_img, int *d_labels, int *d_changed)
{
    cudaFree(d_img);
    cudaFree(d_labels);
    cudaFree(d_changed);
}