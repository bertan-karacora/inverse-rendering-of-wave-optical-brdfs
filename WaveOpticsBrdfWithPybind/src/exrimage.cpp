#include "exrimage.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

using namespace Eigen;
namespace py = pybind11;

PYBIND11_MODULE(exrimage, m) {
    py::class_<EXRImage>(m, "EXRImage")
        .def(py::init<const char *>())
        .def_readwrite("width", &EXRImage::width)
        .def_readwrite("height", &EXRImage::height)
        .def_readwrite("values", &EXRImage::values)
        .def("readImage", &EXRImage::readImage)
        .def("writeImageRGB", &EXRImage::writeImageRGB)
        .def("writeImage", &EXRImage::writeImage);
}


EXRImage::EXRImage(const char *filename) {
    readImage(filename);
}

void EXRImage::readImage(const char *filename) {
    RgbaInputFile file(filename);
    Imath::Box2i dw = file.dataWindow();
    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    Array2D<Rgba> image;
    image.resizeErase(height, width);
    file.setFrameBuffer(&image[0][0] - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels(dw.min.y, dw.max.y);

    values = MatrixXf(width, height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            values(i, j) = image[i][j].r;
        }
    }
}

void EXRImage::writeImageRGB(const MatrixXf r_image, const MatrixXf g_image, const MatrixXf b_image, int outputWidth, int outputHeight, const char *filename) {
    Rgba *pixels = new Rgba[outputHeight * outputWidth];

    // Write to image
    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            pixels[i * outputWidth + j].r = r_image(i, j);
            pixels[i * outputWidth + j].g = g_image(i, j);
            pixels[i * outputWidth + j].b = b_image(i, j);
            pixels[i * outputWidth + j].a = 1.0f;
        }
    }

    RgbaOutputFile file(filename, outputHeight, outputWidth, WRITE_RGBA);
    file.setFrameBuffer(pixels, 1, outputWidth);
    file.writePixels(outputHeight);

    delete[] pixels;
}

void EXRImage::writeImage(const MatrixXf image, int outputWidth, int outputHeight, const char *filename) {
    writeImageRGB(image, image, image, outputWidth, outputHeight, filename);
}
