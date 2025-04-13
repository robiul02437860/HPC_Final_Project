#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>

Image* read_pgm(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char magic[3];
    fscanf(fp, "%2s\n", magic);
    if (magic[0] != 'P' || magic[1] != '5') {
        fprintf(stderr, "Unsupported file format!\n");
        exit(EXIT_FAILURE);
    }

    Image *img = (Image*)malloc(sizeof(Image));

    int maxval;
    fscanf(fp, "%d %d\n%d\n", &img->width, &img->height, &maxval);

    img->data = (uint8_t*)malloc(img->width * img->height);

    fread(img->data, 1, img->width * img->height, fp);

    fclose(fp);
    return img;
}

void write_pgm(const char *filename, const Image *img) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Error writing file");
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "P5\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, img->width * img->height, fp);

    fclose(fp);
}

void free_image(Image *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}
