#ifndef PIXELFORGE_FRAME_CAPTURE_H
#define PIXELFORGE_FRAME_CAPTURE_H

#include <stdint.h>
#include <stddef.h>

/**
 * Capture a framebuffer to a PNG/PPM file
 *
 * Converts RGBA8 framebuffer data to PNG or PPM format and writes it to disk.
 * The filename extension determines the output format (.png or .ppm).
 *
 * @param filename    Output filename (e.g., "frame_001.png" or "frame_001.ppm")
 * @param buffer      Pointer to framebuffer data (RGBA8 format)
 * @param width       Width in pixels
 * @param height      Height in pixels
 * @param stride      Bytes per scanline (may include padding)
 * @return            0 on success, -1 on failure
 */
int frame_capture_rgba(const char *filename, const uint8_t *buffer,
                       uint32_t width, uint32_t height, uint32_t stride);

/**
 * Generate a timestamped filename
 *
 * Creates a filename with optional prefix and suffix, suitable for frame capture.
 *
 * @param output      Output buffer for filename
 * @param maxlen      Maximum length of output buffer
 * @param prefix      Prefix string (e.g., "frame") or NULL
 * @param frame_num   Frame number to embed in filename
 * @param suffix      File extension (e.g., ".png") or NULL
 * @return            0 on success, -1 on buffer overflow
 */
int frame_capture_gen_filename(char *output, size_t maxlen,
                               const char *prefix, uint32_t frame_num,
                               const char *suffix);

#endif /* PIXELFORGE_FRAME_CAPTURE_H */
