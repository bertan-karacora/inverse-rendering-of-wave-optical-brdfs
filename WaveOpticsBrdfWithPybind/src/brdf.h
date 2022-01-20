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

class BrdfBase {
    public:
        BrdfBase() {}

    public:
        virtual Float queryBrdf(const Query &query) = 0;
        virtual MatrixXf genBrdfImage(const Query &query, int resolution) = 0;

    protected:
};

class GeometricBrdf: public BrdfBase {
    public:
        GeometricBrdf() {}
        GeometricBrdf(Heightfield *heightfield, int sampleNum = 10000000) : heightfield(heightfield), sampleNum(sampleNum) {}

        MatrixXf genNdfImage(const Query &query, int resolution);
        virtual Float queryBrdf(const Query &query);
        virtual MatrixXf genBrdfImage(const Query &query, int resolution);

    protected:
        Heightfield *heightfield;

    private:
        int sampleNum;
};

class WaveBrdfAccel: public BrdfBase {
    public:
        WaveBrdfAccel() {}
        WaveBrdfAccel(string method, GaborBasis gaborBasis, int width, int height, Float texelWidth)
                        : method(method), gaborBasis(gaborBasis), width(width), height(height), texelWidth(texelWidth) {}

        comp queryIntegral(const Query &query, int layer, int xIndex, int yIndex);
        virtual Float queryBrdf(const Query &query);
        virtual MatrixXf genBrdfImage(const Query &query, int resolution);

    public:
        string method;

    private:
        int width, height;
        Float texelWidth;
        GaborBasis gaborBasis;
};

#endif
