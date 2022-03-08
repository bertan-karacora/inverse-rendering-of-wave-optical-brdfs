#ifndef EXR_IMAGE_H
#define EXR_IMAGE_H

#include "helpers.h"

#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>

class EXRImage {
    public:
        EXRImage(const char *filename);
        void readImage(const char *filename);
        static string writeImageRGB(const Eigen::MatrixXf r_image, const Eigen::MatrixXf g_image, const Eigen::MatrixXf b_image, int outputWidth, int outputHeight, string filename);
        static void writeImage(const Eigen::MatrixXf image, int outputWidth, int outputHeight, string filename);
    
    public:
        Eigen::MatrixXf values;
        int width, height;
};

#endif
