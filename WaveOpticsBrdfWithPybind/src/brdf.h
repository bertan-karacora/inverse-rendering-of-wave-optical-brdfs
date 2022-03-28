#ifndef BRDF_H
#define BRDF_H

#include "helpers.h"
#include "heightfield.h"
#include "heightfielddiff.h"
#include "gaborkernel.h"
#include "gaborkerneldiff.h"
#include "spectrum.h"

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

class WaveBrdfAccel {
    public:
        WaveBrdfAccel() {}
        WaveBrdfAccel(string diff_model, int width, int height, Float texelWidth, int resolution)
                        : diff_model(diff_model), width(width), height(height), texelWidth(texelWidth), resolution(resolution) {}

        comp queryIntegral(const Query &query, const GaborBasis &gaborBasis, int layer, int xIndex, int yIndex);
        Float queryBrdf(const Query &query, const GaborBasis &gaborBasis);
        BrdfImage genBrdfImage(const Query &query, const GaborBasis &gaborBasis);
        void genBrdfImageDiff(const Query &query, const HeightfieldDiff &hf, BrdfImage ref);

    public:
        string diff_model;
        int width, height;
        Float texelWidth;
        int resolution;
};

#endif
