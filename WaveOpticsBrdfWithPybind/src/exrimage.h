#ifndef EXR_IMAGE_H
#define EXR_IMAGE_H

#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>
#include "helpers.h"

using namespace Imf;
using namespace std;

class EXRImage {
public:
    EXRImage(const char *filename);
    void readImage(const char *filename);
    static void writeImage(const double *image, const char *filename, int outputHeight, int outputWidth);
    
public:
    double *values;
    int width, height;
};

#endif
