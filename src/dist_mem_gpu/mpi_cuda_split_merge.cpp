#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <cstring>

#include "../common/image_io.h"


extern "C" {
    void cuda_init_labels(uint8_t **d_img, int **d_labels, int **d_changed, uint8_t *local_data, int width, int height_per_proc);
    void cuda_merge_labels(int *d_changed, int *changed, uint8_t *d_img, int *d_labels, int *labels, int width, int height_per_proc);
    void cuda_free(uint8_t *d_img, int *d_labels, int *d_changed);
    void cuda_update_labels(int *labels, int *d_labels, int width, int height_per_proc);
}

void exchange_boundaries(int *labels, int width, int height, int rank, int size, MPI_Comm comm) {
    MPI_Status status;
    int *row = (int*)malloc(width * sizeof(int));
    if (rank != 0) {
        memcpy(row, labels, width * sizeof(int));
        MPI_Sendrecv(row, width, MPI_INT, rank - 1, 0,
                     row, width, MPI_INT, rank - 1, 0, comm, &status);
        memcpy(labels, row, width * sizeof(int));
    }
    if (rank != size - 1) {
        memcpy(row, labels + (height-1)*width, width * sizeof(int));
        MPI_Sendrecv(row, width, MPI_INT, rank + 1, 0,
                     row, width, MPI_INT, rank + 1, 0, comm, &status);
        memcpy(labels + (height-1)*width, row, width * sizeof(int));
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


    uint8_t *d_img     = nullptr;
    int     *d_labels  = nullptr;
    int     *d_changed = nullptr;
    int changed;
    cuda_init_labels(&d_img, &d_labels, &d_changed, local_data, width, height_per_proc);
    int *labels = (int*)malloc(width * height_per_proc * sizeof(int));

    do {
        changed = 0;

        cuda_merge_labels(d_changed, &changed, d_img, d_labels, labels, width, height_per_proc);

        MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        exchange_boundaries(labels, width, height_per_proc, rank, size, MPI_COMM_WORLD);

        cuda_update_labels(labels, d_labels, width, height_per_proc);

    } while (changed);

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

    cuda_free(d_img, d_labels, d_changed);

    MPI_Finalize();
    return 0;
}