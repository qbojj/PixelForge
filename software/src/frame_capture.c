#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#include "frame_capture.h"

/* STB_IMAGE_WRITE implementation - must be defined before include */
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

/**
 * Capture an RGBA framebuffer to PNG file
 *
 * Handles conversion from hardware format (32-bit BGRA with stride)
 * to standard RGBA format expected by stb_image_write.
 */
int frame_capture_rgba(const char *filename, const uint8_t *buffer,
                       uint32_t width, uint32_t height, uint32_t stride)
{
    if (!filename || !buffer) {
        fprintf(stderr, "frame_capture_rgba: invalid arguments\n");
        return -1;
    }

    /* Allocate temporary RGBA buffer for conversion */
    uint8_t *rgb_buffer = malloc(width * height * 4);
    if (!rgb_buffer) {
        fprintf(stderr, "frame_capture_rgba: failed to allocate buffer\n");
        return -1;
    }

    /* Convert from hardware format (BGRA with stride) to linear RGBA */
    for (uint32_t y = 0; y < height; y++) {
        const uint8_t *src_row = buffer + (y * stride);
        uint8_t *dst_row = rgb_buffer + (y * width * 4);

        for (uint32_t x = 0; x < width; x++) {
            /* Hardware format: BGRA */
            uint8_t b = src_row[x * 4 + 0];
            uint8_t g = src_row[x * 4 + 1];
            uint8_t r = src_row[x * 4 + 2];
            uint8_t a = src_row[x * 4 + 3];

            /* Output format: RGBA */
            dst_row[x * 4 + 0] = r;
            dst_row[x * 4 + 1] = g;
            dst_row[x * 4 + 2] = b;
            dst_row[x * 4 + 3] = a;
        }
    }

    const char *ext = strrchr(filename, '.');
    if (!ext) {
        fprintf(stderr, "frame_capture_rgba: filename '%s' has no extension\n", filename);
        free(rgb_buffer);
        return -1;
    }
    if (strcmp(ext, ".png") != 0) {
        fprintf(stderr, "frame_capture_rgba: unsupported file extension '%s', only .png supported\n", ext);
        free(rgb_buffer);
        return -1;
    }

    int result = stbi_write_png(filename, width, height, 4, rgb_buffer, width * 4);
    free(rgb_buffer);

    if (result) {
        printf("Frame captured: %s (%ux%u)\n", filename, width, height);
        return 0;
    } else {
        fprintf(stderr, "Failed to write frame: %s\n", filename);
        return -1;
    }
}

/**
 * Generate a timestamped filename for frame capture
 */
int frame_capture_gen_filename(char *output, size_t maxlen,
                               const char *prefix, uint32_t frame_num,
                               const char *suffix)
{
    if (!output || maxlen == 0) {
        return -1;
    }

    /* Build filename with frame number */
    const char *pre = prefix ? prefix : "frame";
    const char *suf = suffix ? suffix : ".png";

    int written = snprintf(output, maxlen, "%s_%05u%s", pre, frame_num, suf);

    if (written < 0 || (size_t)written >= maxlen) {
        fprintf(stderr, "Filename buffer too small\n");
        return -1;
    }

    return 0;
}
