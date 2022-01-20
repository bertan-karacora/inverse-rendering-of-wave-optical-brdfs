#ifndef EXR_IMAGE_H
#define EXR_IMAGE_H

#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>
#include <Eigen/Dense>
#include "helpers.h"

using namespace Imf;
using namespace Eigen;

class EXRImage {
    public:
        EXRImage(const char *filename);
        void readImage(const char *filename);
        static string writeImageRGB(const MatrixXf r_image, const MatrixXf g_image, const MatrixXf b_image, int outputWidth, int outputHeight, string filename);
        static void writeImage(const MatrixXf image, int outputWidth, int outputHeight, string filename);
    
    public:
        MatrixXf values;
        int width, height;
};

#endif
