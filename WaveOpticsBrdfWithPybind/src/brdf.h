#ifndef WDF_H
#define WDF_H

#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include "heightfield.h"
#include "gaborkernel.h"
#include "spectrum.h"

using namespace std;
using namespace Eigen;

struct Query {
    Vector2 mu_p;
    Float sigma_p;
    Vector2 omega_i;
    Vector2 omega_o;
    Float lambda;   // in microns.
};

struct BrdfImage {
    MatrixXf r;
    MatrixXf g;
    MatrixXf b;
};

class GeometricBrdf {
    public:
        GeometricBrdf() {}
        GeometricBrdf(Heightfield *heightfield, int sampleNum = 10000000) : heightfield(heightfield), sampleNum(sampleNum) {}

        MatrixXf genNdfImage(const Query &query, int resolution);
        Float queryBrdf(const Query &query);
        BrdfImage genBrdfImage(const Query &query, int resolution);

    protected:
        Heightfield *heightfield;

    private:
        int sampleNum;
};

class WaveBrdfAccel {
    public:
        WaveBrdfAccel() {}
        WaveBrdfAccel(string diff_model, int width, int height, Float texelWidth, int resolution)
                        : diff_model(diff_model), width(width), height(height), texelWidth(texelWidth), resolution(resolution) {}

        comp queryIntegral(const Query &query, const GaborBasis &gaborBasis, int layer, int xIndex, int yIndex);
        Float queryBrdf(const Query &query, const GaborBasis &gaborBasis);
        BrdfImage genBrdfImage(const Query &query, const GaborBasis &gaborBasis);

    public:
        string diff_model;
        int width, height;
        Float texelWidth;
        int resolution;
};

#endif
