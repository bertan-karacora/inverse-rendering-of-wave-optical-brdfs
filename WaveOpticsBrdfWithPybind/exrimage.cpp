#include "exrimage.h"
#include <iostream>

EXRImage::EXRImage(const char *filename) {
    readImage(filename);
}

void EXRImage::readImage(const char *filename) {
    RgbaInputFile file(filename);
    Imath::Box2i dw = file.dataWindow();
    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    // Ugly, help
    Array2D<Rgba> image;
    image.resizeErase(height, width);
    file.setFrameBuffer(&image[0][0] - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels(dw.min.y, dw.max.y);
    values = new double[width * height];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            values[i * width + j] = image[i][j].r;
        }
    }
}

void EXRImage::writeImage(const double *image, const char *filename, int outputHeight, int outputWidth) {
    Rgba *pixels = new Rgba[outputHeight * outputWidth];

    // Write to image
    for (int i = 0; i < outputHeight; i++)
        for (int j = 0; j < outputWidth; j++) {
            pixels[i * outputWidth + j].r = image[(i * outputWidth + j) * 3 + 0];
            pixels[i * outputWidth + j].g = image[(i * outputWidth + j) * 3 + 1];
            pixels[i * outputWidth + j].b = image[(i * outputWidth + j) * 3 + 2];
            pixels[i * outputWidth + j].a = 1.0f;
        }

    RgbaOutputFile file(filename, outputHeight, outputWidth, WRITE_RGBA);
    file.setFrameBuffer(pixels, 1, outputWidth);
    file.writePixels(outputHeight);

    delete[] pixels;
}
