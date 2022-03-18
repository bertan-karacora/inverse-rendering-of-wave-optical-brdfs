#ifndef BRDF_H
#define BRDF_H

#include "helpers.h"
#include "heightfield.h"
#include "gaborkernel.h"
#include "spectrum.h"

#include <enoki/python.h>
#include <enoki/autodiff.h>
#include <enoki/cuda.h>

using FloatC = enoki::CUDAArray<Float>;
using FloatD = enoki::DiffArray<FloatC>;

struct Query {
    Vector2 mu_p;
    Float sigma_p;
    Vector2 omega_i;
    Vector2 omega_o;
    Float lambda;   // in microns.
};

struct BrdfImage {
    Eigen::MatrixXf r;
    Eigen::MatrixXf g;
    Eigen::MatrixXf b;
};

class GeometricBrdf {
    public:
        GeometricBrdf() {}
        GeometricBrdf(Heightfield *heightfield, int sampleNum = 10000000) : heightfield(heightfield), sampleNum(sampleNum) {}

        Eigen::MatrixXf genNdfImage(const Query &query, int resolution);
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
        BrdfImage genBrdfImageDiff(const Query &query, Heightfield &heightfield);
        FloatD backpropagate(Float loss);

    public:
        string diff_model;
        int width, height;
        Float texelWidth;
        int resolution;
};

#endif
