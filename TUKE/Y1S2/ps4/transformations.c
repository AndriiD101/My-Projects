#include <stdio.h>
#include <stdlib.h>
#include "transformations.h"
#include <string.h>

struct bmp_image* flip_horizontally(const struct bmp_image* image) {
    uint32_t rows, cols;
    if (!image || !image->header || !image->data) {
        return NULL;
    }

    struct bmp_image* flipped = malloc(sizeof(struct bmp_image));

    flipped->header = malloc(sizeof(struct bmp_header));

    memcpy(flipped->header, image->header, sizeof(struct bmp_header));

    size_t pixel_data = image->header->width * image->header->height * sizeof(struct pixel);
    flipped->data = malloc(pixel_data);

    for (rows = 0; rows < image->header->height; rows++) {
        for (cols = 0; cols < image->header->width; cols++) {
            flipped->data[rows * image->header->width + cols] = image->data[rows * image->header->width + (image->header->width - 1 - cols)];
        }
    }

    return flipped;
}

struct bmp_image* flip_vertically(const struct bmp_image* image) {
    uint32_t rows, cols;
    if (!image || !image->header || !image->data) return NULL;

    struct bmp_image* flipped = malloc(sizeof(struct bmp_image));

    flipped->header = malloc(sizeof(struct bmp_header));

    memcpy(flipped->header, image->header, sizeof(struct bmp_header));

    flipped->data = malloc(image->header->width * image->header->height * sizeof(struct pixel));

    for (rows = 0; rows < image->header->height; rows++) {
        for (cols = 0; cols < image->header->width; cols++) {
            flipped->data[(image->header->height - 1 - rows) * image->header->width + cols] = image->data[rows * image->header->width + cols];
        }
    }

    return flipped;
}

struct bmp_image* rotate_right(const struct bmp_image* image) {
    uint32_t rows, cols;
    if (!image || !image->header || !image->data) return NULL;

    struct bmp_image* rotated = malloc(sizeof(struct bmp_image));
    if (!rotated) return NULL;

    rotated->header = malloc(sizeof(struct bmp_header));
    if (!rotated->header) {
        free(rotated);
        return NULL;
    }
    memcpy(rotated->header, image->header, sizeof(struct bmp_header));
    rotated->header->width = image->header->height;
    rotated->header->height = image->header->width;

    size_t pixel_data = rotated->header->width * rotated->header->height * sizeof(struct pixel);
    rotated->data = malloc(pixel_data);
    if (!rotated->data) {
        free(rotated->header);
        free(rotated);
        return NULL;
    }

    for (rows = 0; rows < image->header->height; rows++) {
        for (cols = 0; cols < image->header->width; cols++) {
            rotated->data[cols * rotated->header->width + (rotated->header->width - rows - 1)] = image->data[rows * image->header->width + cols];
        }
    }

    return rotated;
}

struct bmp_image* rotate_left(const struct bmp_image* image) {
    uint32_t rows, cols;
    if (!image || !image->header || !image->data) return NULL;

    struct bmp_image* rotated = malloc(sizeof(struct bmp_image));
    if (!rotated) return NULL;

    rotated->header = malloc(sizeof(struct bmp_header));
    if (!rotated->header) {
        free(rotated);
        return NULL;
    }
    memcpy(rotated->header, image->header, sizeof(struct bmp_header));
    rotated->header->width = image->header->height;
    rotated->header->height = image->header->width;

    size_t pixel_data = rotated->header->width * rotated->header->height * sizeof(struct pixel);
    rotated->data = malloc(pixel_data);
    if (!rotated->data) {
        free(rotated->header);
        free(rotated);
        return NULL;
    }

    for (rows = 0; rows < image->header->height; rows++) {
        for (cols = 0; cols < image->header->width; cols++) {
            rotated->data[(rotated->header->height - cols - 1) * rotated->header->width + rows] = image->data[rows * image->header->width + cols];
        }
    }

    return rotated;
}
