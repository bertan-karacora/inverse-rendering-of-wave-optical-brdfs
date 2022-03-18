#ifndef HEIGHTFIELD_H
#define HEIGHTFIELD_H

#include "helpers.h"
#include "gaborkernel.h"

#include <enoki/python.h>
#include <enoki/autodiff.h>
#include <enoki/cuda.h>

using FloatC = enoki::CUDAArray<Float>;
using FloatD = enoki::DiffArray<FloatC>;

class Heightfield;

class GaborBasis {
    public:
        GaborBasis() {};
        GaborBasis(const Heightfield &hf);

    public:
        vector<vector<GaborKernelPrime>> gaborKernelPrime;
        vector<vector<vector<AABB>>> angularBB;
        int topLayer;
};

// Suppose it starts from (0, 0) and extends to positive (w * mTexelWidth, h * mTexelWidth) and is tiling along both axes.
class Heightfield {
    public:
        Heightfield() {};
        Heightfield(Eigen::MatrixXf values, int width, int height, Float texelWidth, Float vertScale);
        Heightfield(GaborBasis gaborBasis, int width, int height, Float texelWidth, Float vertScale);

        // Bicubic interpolation.
        Float getValue(Float x, Float y);
        // Bicubic interpolation. u and v are from 0 to 1.
        Float getValueUV(Float u, Float v);

        GaborKernel g(int i, int j, Float F, Float lambda);
        Vector2 n(Float i, Float j);

    public:
        FloatD values;
        int width, height;
        Float texelWidth;   // in microns.
        Float vertScale;

    private:
        void computeCoeff(Float *alpha, const Float *x);

        inline int mod(int x, int y) {
            return ((x % y) + y) % y;
        }

        inline Float hp(int x, int y) {
            return values[mod(x, height) * width + mod(y, width)];
        }
        
        inline Float hpx(int x, int y) {
            return (values[mod(x + 1, height) * width + mod(y, width)] - values[mod(x - 1, height) * width + mod(y, width)]) / 2.0;
        }
        
        inline Float hpy(int x, int y) {
            return (values[mod(x, height) * width + mod(y + 1, width)] - values[mod(x, height) * width + mod(y - 1, width)]) / 2.0;
        }
        
        inline Float hpxy(int x, int y) {
            return (hp(x + 1, y + 1) - hp(x + 1, y) - hp(x, y + 1) + 2.0 * hp(x, y) - hp(x - 1, y) - hp(x, y - 1) + hp(x - 1, y - 1)) / 2.0;
        }
};

#endif
