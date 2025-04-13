#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int width;
    int height;
    uint8_t *data;
} Image;

Image* read_pgm(const char *filename);
void write_pgm(const char *filename, const Image *img);
void free_image(Image *img);

#ifdef __cplusplus
}
#endif

#endif
