
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include "../common/image_io.h"

#define DIFF_THRESHOLD 10

// Exchange top and bottom halo rows with neighbors
void exchange_boundaries(int *labels, int width, int height_per_proc, int rank, int size, MPI_Comm comm) {
    MPI_Status status;

    // Send top row and receive into halo above
    if (rank != 0) {
        MPI_Sendrecv(labels + width, width, MPI_INT, rank - 1, 0,
                     labels,        width, MPI_INT, rank - 1, 0, comm, &status);
    }

    // Send bottom row and receive into halo below
    if (rank != size - 1) {
        MPI_Sendrecv(labels + height_per_proc * width, width, MPI_INT, rank + 1, 0,
                     labels + (height_per_proc + 1) * width, width, MPI_INT, rank + 1, 0, comm, &status);
    }
}

// Simple merge operation within local chunk
void merge(const uint8_t *img, int *labels, int width, int height_per_proc) {
    int merged = 1;
    while (merged) {
        merged = 0;
        for (int y = 1; y <= height_per_proc; y++) {
            for (int x = 0; x < width - 1; x++) {
                int idx = y * width + x;
                int right = idx + 1;
                int down  = idx + width;

                if (x + 1 < width && labels[idx] != labels[right]) {
                    if (abs(img[idx] - img[right]) < DIFF_THRESHOLD) {
                        int old = labels[right], new = labels[idx];
                        for (int i = 0; i < (height_per_proc + 2) * width; i++)
                            if (labels[i] == old) labels[i] = new;
                        merged = 1;
                    }
                }

                if (y < height_per_proc + 1 && labels[idx] != labels[down]) {
                    if (abs(img[idx] - img[down]) < DIFF_THRESHOLD) {
                        int old = labels[down], new = labels[idx];
                        for (int i = 0; i < (height_per_proc + 2) * width; i++)
                            if (labels[i] == old) labels[i] = new;
                        merged = 1;
                    }
                }
            }
        }
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
    int width = 0, total_height = 0;

    if (rank == 0) {
        img = read_pgm(argv[1]);
        width = img->width;
        total_height = img->height;
    }

    // Broadcast width and height to all ranks
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int height_per_proc = total_height / size;

    // Allocate local image chunk and scatter
    uint8_t *local_data = (uint8_t *)malloc(width * height_per_proc * sizeof(uint8_t));
    MPI_Scatter(img ? img->data : NULL, width * height_per_proc, MPI_UINT8_T,
                local_data, width * height_per_proc, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    // Allocate haloed label array (+2 for top/bottom halo rows)
    int *labels = (int *)malloc((height_per_proc + 2) * width * sizeof(int));
    for (int y = 0; y < height_per_proc; y++) {
        for (int x = 0; x < width; x++) {
            labels[(y + 1) * width + x] = (rank * height_per_proc + y) * width + x;
        }
    }

    // Merge neighboring regions using local info and boundary exchange
    for (int iter = 0; iter < 5; iter++) {
        merge(local_data, labels, width, height_per_proc);
        exchange_boundaries(labels, width, height_per_proc, rank, size, MPI_COMM_WORLD);
    }

    // Copy final labels back to uint8_t output (strip halos)
    uint8_t *output_data = (uint8_t *)malloc(width * height_per_proc);
    for (int y = 0; y < height_per_proc; y++) {
        for (int x = 0; x < width; x++) {
            output_data[y * width + x] = labels[(y + 1) * width + x] % 256;
        }
    }

    // Gather all results back to root
    if (rank == 0) {
        uint8_t *full_output = (uint8_t *)malloc(width * total_height);
        MPI_Gather(output_data, width * height_per_proc, MPI_UINT8_T,
                   full_output, width * height_per_proc, MPI_UINT8_T, 0, MPI_COMM_WORLD);
        img->data = full_output;
        write_pgm(argv[2], img);
        free_image(img);
    } else {
        MPI_Gather(output_data, width * height_per_proc, MPI_UINT8_T,
                   NULL, 0, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    }

    // Cleanup
    free(local_data);
    free(labels);
    free(output_data);

    MPI_Finalize();
    return 0;
}

// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <math.h>
// #include <mpi.h>

// #include "../common/image_io.h"

// #define VAR_THRESHOLD 150
// #define DIFF_THRESHOLD 5

// // Exchange boundary rows with neighboring processes
// void exchange_boundaries(int *labels, int width, int height, int rank, int size, MPI_Comm comm) {
//     MPI_Status status;
    


//     if (rank != 0) {
//         MPI_Sendrecv(labels + width, width, MPI_INT, rank - 1, 0,
//             labels,        width, MPI_INT, rank - 1, 0, comm, &status);
//         // MPI_Sendrecv(labels, width, MPI_INT, rank - 1, 0,
//         //              labels - width, width, MPI_INT, rank - 1, 0, comm, &status);
//     }

//     if (rank != size - 1) {
//         MPI_Sendrecv(labels + height_per_proc * width, width, MPI_INT, rank + 1, 0,
//             labels + (height_per_proc + 1) * width, width, MPI_INT, rank + 1, 0, comm, &status);
//         // MPI_Sendrecv(labels + (height - 1) * width, width, MPI_INT, rank + 1, 0,
//         //              labels + height * width, width, MPI_INT, rank + 1, 0, comm, &status);
//     }
// }

// // Simple merge within process region
// void merge(const Image *img, int *labels, int width, int height) {
//     int merged = 1;
//     while (merged) {
//         merged = 0;
//         for (int y = 0; y < height - 1; y++) {
//             for (int x = 0; x < width - 1; x++) {
//                 int curr = labels[y * width + x];
//                 int right = labels[y * width + (x + 1)];
//                 int down  = labels[(y + 1) * width + x];

//                 if (curr != right) {
//                     uint8_t val1 = img->data[y * width + x];
//                     uint8_t val2 = img->data[y * width + (x + 1)];
//                     if (abs(val1 - val2) < DIFF_THRESHOLD) {
//                         for (int i = 0; i < width * height; i++) {
//                             if (labels[i] == right) labels[i] = curr;
//                         }
//                         merged = 1;
//                     }
//                 }

//                 if (curr != down) {
//                     uint8_t val1 = img->data[y * width + x];
//                     uint8_t val2 = img->data[(y + 1) * width + x];
//                     if (abs(val1 - val2) < DIFF_THRESHOLD) {
//                         for (int i = 0; i < width * height; i++) {
//                             if (labels[i] == down) labels[i] = curr;
//                         }
//                         merged = 1;
//                     }
//                 }
//             }
//         }
//     }
// }

// int main(int argc, char *argv[]) {
//     int rank, size;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (argc != 3) {
//         if (rank == 0)
//             printf("Usage: %s input.pgm output.pgm\n", argv[0]);
//         MPI_Finalize();
//         return -1;
//     }

//     Image *img = NULL;
//     if (rank == 0)
//         img = read_pgm(argv[1]);

//     int width, total_height;
//     if (rank == 0) {
//         width = img->width;
//         total_height = img->height;
//     }

//     MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&total_height, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     int height_per_proc = total_height / size;

//     uint8_t *local_data = (uint8_t*)malloc(width * height_per_proc * sizeof(uint8_t));
//     MPI_Scatter(img ? img->data : NULL, width * height_per_proc, MPI_UINT8_T,
//                 local_data, width * height_per_proc, MPI_UINT8_T, 0, MPI_COMM_WORLD);

//     // int *labels = (int*)malloc(width * height_per_proc * sizeof(int));
//     int *labels = malloc(width * (height_per_proc + 2) * sizeof(int));

//     for (int y = 0; y < height_per_proc; y++) {
//         for (int x = 0; x < width; x++) {
//             labels[y * width + x] = y * width + x;
//         }
//     }

//     for (int iter = 0; iter < 5; iter++) {
//         merge((Image*)(&(Image){width, height_per_proc, local_data}), labels, width, height_per_proc);
//         exchange_boundaries(labels, width, height_per_proc, rank, size, MPI_COMM_WORLD);
//     }

//     uint8_t *output_data = (uint8_t*)malloc(width * height_per_proc * sizeof(uint8_t));
//     for (int i = 0; i < width * height_per_proc; i++) {
//         output_data[i] = labels[i] % 256;
//     }

//     if (rank == 0) {
//         uint8_t *full_output = (uint8_t*)malloc(width * total_height * sizeof(uint8_t));
//         MPI_Gather(output_data, width * height_per_proc, MPI_UINT8_T,
//                    full_output, width * height_per_proc, MPI_UINT8_T, 0, MPI_COMM_WORLD);
//         img->data = full_output;
//         write_pgm(argv[2], img);
//         free_image(img);
//     } else {
//         MPI_Gather(output_data, width * height_per_proc, MPI_UINT8_T,
//                    NULL, 0, MPI_UINT8_T, 0, MPI_COMM_WORLD);
//     }

//     free(local_data);
//     free(labels);
//     free(output_data);

//     MPI_Finalize();
//     return 0;
// }
