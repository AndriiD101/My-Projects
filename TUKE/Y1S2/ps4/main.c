#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "bmp.h"
#include "transformations.h"


int main() {
    FILE* stream = fopen("image.bmp", "rb");
    if (stream == NULL) {
        return 1;
    }

    struct bmp_image* image = read_bmp(stream);
    if (image == NULL) {
        fclose(stream);
        return 1;
    }

    struct bmp_header* header = read_bmp_header(stream);
    if (header == NULL) {
        fclose(stream);
        return 1;
    }

    struct pixel* data = read_data(stream, header);
    if (data == NULL) {
        free(header);
        fclose(stream);
        return 1;
    }

    struct bmp_image* flipped_horizontally = flip_horizontally(image);
    if (flipped_horizontally == NULL) {
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    FILE* output_stream = fopen("output_horizontal.bmp", "wb");
    if (output_stream == NULL) {
        free_bmp_image(flipped_horizontally);
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    if (!write_bmp(output_stream, flipped_horizontally)) {
        fclose(output_stream);
        free_bmp_image(flipped_horizontally);
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    fclose(output_stream);
    free_bmp_image(flipped_horizontally);

    struct bmp_image* flipped_vertically = flip_vertically(image);
    if (flipped_vertically == NULL) {
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    output_stream = fopen("output_vertical.bmp", "wb");
    if (output_stream == NULL) {
        free_bmp_image(flipped_vertically);
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    if (!write_bmp(output_stream, flipped_vertically)) {
        fclose(output_stream);
        free_bmp_image(flipped_vertically);
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    fclose(output_stream);
    free_bmp_image(flipped_vertically);

    struct bmp_image* rotated_right = rotate_right(image);
    if (rotated_right == NULL) {
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    output_stream = fopen("output_rotated_right.bmp", "wb");
    if (output_stream == NULL) {
        free_bmp_image(rotated_right);
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    if (!write_bmp(output_stream, rotated_right)) {
        fclose(output_stream);
        free_bmp_image(rotated_right);
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    fclose(output_stream);
    free_bmp_image(rotated_right);

    struct bmp_image* rotated_left = rotate_left(image);
    if (rotated_left == NULL) {
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }


    output_stream = fopen("output_rotated_left.bmp", "wb");
    if (output_stream == NULL) {
        free_bmp_image(rotated_left);
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    if (!write_bmp(output_stream, rotated_left)) {
        fclose(output_stream);
        free_bmp_image(rotated_left);
        free_bmp_image(image);
        fclose(stream);
        return 1;
    }

    fclose(output_stream);
    free_bmp_image(rotated_left);
    free_bmp_image(image);
    fclose(stream);

    return 0;
}
