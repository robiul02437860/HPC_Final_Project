
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include "../common/image_io.h"

#define DIFF_THRESHOLD 4

void init_labels(uint8_t *img, int *labels, int width, int height) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            labels[y * width + x] = y * width + x;
}

int merge_labels(uint8_t *img, int *labels, int width, int height) {
    int changed = 0;
    #pragma omp parallel for collapse(2) reduction(|:changed)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int right = idx + 1;
            int down = idx + width;

            if (x + 1 < width &&
                abs(img[idx] - img[right]) < DIFF_THRESHOLD &&
                labels[idx] != labels[right]) {
                int min_label = labels[idx] < labels[right] ? labels[idx] : labels[right];
                labels[idx] = min_label;
                labels[right] = min_label;
                changed = 1;
            }

            if (y + 1 < height &&
                abs(img[idx] - img[down]) < DIFF_THRESHOLD &&
                labels[idx] != labels[down]) {
                int min_label = labels[idx] < labels[down] ? labels[idx] : labels[down];
                labels[idx] = min_label;
                labels[down] = min_label;
                changed = 1;
            }
        }
    }
    return changed;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s input.pgm output.pgm\n", argv[0]);
        return -1;
    }

    Image *img = read_pgm(argv[1]);
    int width = img->width;
    int height = img->height;
    size_t img_size = width * height;

    int *labels = (int *)malloc(img_size * sizeof(int));
    init_labels(img->data, labels, width, height);

    while (merge_labels(img->data, labels, width, height));

    #pragma omp parallel for
    for (int i = 0; i < img_size; i++)
        img->data[i] = labels[i] % 1024;

    write_pgm(argv[2], img);
    free_image(img);
    free(labels);
    return 0;
}

// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <math.h>
// #include <omp.h>

// #include "../common/image_io.h"

// #define VAR_THRESHOLD 150
// #define DIFF_THRESHOLD 10

// typedef struct {
//     int x, y;
//     int size;
// } Region;

// float compute_variance(const Image *img, Region r) {
//     int sum = 0, sum_sq = 0;
//     for (int i = r.y; i < r.y + r.size; i++) {
//         for (int j = r.x; j < r.x + r.size; j++) {
//             uint8_t val = img->data[i * img->width + j];
//             sum += val;
//             sum_sq += val * val;
//         }
//     }
//     int N = r.size * r.size;
//     float mean = sum / (float)N;
//     float var = (sum_sq / (float)N) - (mean * mean);
//     return var;
// }

// // Parallel Split using OpenMP Tasks
// void split(const Image *img, int *labels, Region r, int *label_id) {
//     float var = compute_variance(img, r);

//     if (var <= VAR_THRESHOLD || r.size == 1) {
//         for (int i = r.y; i < r.y + r.size; i++) {
//             for (int j = r.x; j < r.x + r.size; j++) {
//                 labels[i * img->width + j] = *label_id;
//             }
//         }
//         #pragma omp atomic
//         (*label_id)++;
//         return;
//     }

//     int new_size = r.size / 2;

//     #pragma omp task firstprivate(r, new_size) shared(labels, label_id)
//     split(img, labels, (Region){r.x, r.y, new_size}, label_id);

//     #pragma omp task firstprivate(r, new_size) shared(labels, label_id)
//     split(img, labels, (Region){r.x + new_size, r.y, new_size}, label_id);

//     #pragma omp task firstprivate(r, new_size) shared(labels, label_id)
//     split(img, labels, (Region){r.x, r.y + new_size, new_size}, label_id);

//     #pragma omp task firstprivate(r, new_size) shared(labels, label_id)
//     split(img, labels, (Region){r.x + new_size, r.y + new_size, new_size}, label_id);
// }

// void merge(const Image *img, int *labels, int width, int height) {
//     int merged = 1;
//     while (merged) {
//         merged = 0;
//         #pragma omp parallel for collapse(2)
//         for (int y = 0; y < height - 1; y++) {
//             for (int x = 0; x < width - 1; x++) {
//                 int curr = labels[y * width + x];
//                 int right = labels[y * width + (x + 1)];
//                 int down  = labels[(y + 1) * width + x];

//                 if (curr != right) {
//                     uint8_t val1 = img->data[y * width + x];
//                     uint8_t val2 = img->data[y * width + (x + 1)];
//                     if (abs(val1 - val2) < DIFF_THRESHOLD) {
//                         #pragma omp critical
//                         {
//                             for (int i = 0; i < width * height; i++) {
//                                 if (labels[i] == right) labels[i] = curr;
//                             }
//                         }
//                         merged = 1;
//                     }
//                 }

//                 if (curr != down) {
//                     uint8_t val1 = img->data[y * width + x];
//                     uint8_t val2 = img->data[(y + 1) * width + x];
//                     if (abs(val1 - val2) < DIFF_THRESHOLD) {
//                         #pragma omp critical
//                         {
//                             for (int i = 0; i < width * height; i++) {
//                                 if (labels[i] == down) labels[i] = curr;
//                             }
//                         }
//                         merged = 1;
//                     }
//                 }
//             }
//         }
//     }
// }

// int main(int argc, char *argv[]) {
//     if (argc != 3) {
//         printf("Usage: %s input.pgm output.pgm\n", argv[0]);
//         return -1;
//     }

//     Image *img = read_pgm(argv[1]);
//     int *labels = (int*)malloc(sizeof(int) * img->width * img->height);

//     int label_id = 1;

//     #pragma omp parallel
//     {
//         #pragma omp single
//         {
//             split(img, labels, (Region){0, 0, img->width}, &label_id);
//         }
//     }

//     merge(img, labels, img->width, img->height);

//     for (int i = 0; i < img->width * img->height; i++) {
//         img->data[i] = labels[i] % 256;
//     }

//     write_pgm(argv[2], img);
//     free(labels);
//     free_image(img);
//     return 0;
// }
