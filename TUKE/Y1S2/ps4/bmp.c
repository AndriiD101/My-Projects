#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "bmp.h"

struct bmp_image* read_bmp(FILE* stream) {
    if (stream == NULL) {
        fprintf(stderr, "Error: The file stream is NULL.\n");
        return NULL;
    }

    struct bmp_image* image = malloc(sizeof(struct bmp_image));
    if (image == NULL) {
        fprintf(stderr, "Error: Memory allocation for bmp_image failed.\n");
        return NULL;
    }

    image->header = read_bmp_header(stream);
    if (image->header == NULL) {
        fprintf(stderr, "Error: Failed to read BMP header.\n");
        free(image);
        return NULL;
    }

    image->data = read_data(stream, image->header);
    if (image->data == NULL) {
        fprintf(stderr, "Error: Failed to read BMP data.\n");
        free(image->header);
        free(image);
        return NULL;
    }

    return image;
}

struct bmp_header* allocate_header() {
    return calloc(1, sizeof(struct bmp_header));
}

bool read_header(FILE* stream, struct bmp_header* header) {
    return fread(header, sizeof(struct bmp_header), 1, stream) == 1;
}

bool validate_header(struct bmp_header* header) {
    return header->type == 0x4d42 && header->width > 0 && header->height > 0;
}

struct bmp_header* read_bmp_header(FILE* stream) {
    if (stream == NULL) 
        return NULL;
    fseek(stream, 0, SEEK_SET);
    struct bmp_header* header = allocate_header();
    if(header == NULL || !read_header(stream, header) || !validate_header(header)) {
        free(header);
        return NULL;
    }
    return header;
}

struct pixel* read_data(FILE* stream, const struct bmp_header* header) {
    if (!stream || !header) {
        return NULL;
    }
    size_t pixel_data = header->width * header->height * sizeof(struct pixel);
    struct pixel* pixels = malloc(pixel_data);
    if (!pixels) {
        return NULL;
    }
    fseek(stream, header->offset, SEEK_SET);
    if (fread(pixels, 1, pixel_data, stream) != pixel_data) {
        free(pixels);
        return NULL;
    }

    return pixels;
}

bool write_bmp(FILE* stream, const struct bmp_image* image) {
    if (!stream || !image || !image->header || !image->data) {
        return false;
    }

    if (fwrite(image->header, sizeof(struct bmp_header), 1, stream) != 1) {
        return false;
    }

    size_t pixel_data_size = image->header->width * image->header->height * sizeof(struct pixel);
    if (fwrite(image->data, pixel_data_size, 1, stream) != 1) {
        return false;
    }

    return true;
}

void free_bmp_image(struct bmp_image* image) {
    if(image!=NULL){
        free(image->header);
        free(image->data);
        free(image);
    }
}
