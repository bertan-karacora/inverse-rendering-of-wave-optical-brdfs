#ifndef HEIGHTFIELD_H
#define HEIGHTFIELD_H

#include "helpers.h"
#include "gaborkernel.h"

class Heightfield;

class GaborBasis {
    public:
        GaborBasis() {};
        GaborBasis(const Heightfield &hf);
        GaborBasis(Heightfield Heightfield, bool diff);

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

        FloatD getValueDiff(Float x, Float y);
        FloatD getValueUVDiff(Float u, Float v);

        GaborKernel g(int i, int j, Float F, Float lambda);
        Vector2 n(Float i, Float j);

    public:
        FloatD valuesDiff;
        Eigen::MatrixXf values;
        int width, height;
        Float texelWidth;   // in microns.
        Float vertScale;

    private:
        void computeCoeff(Float *alpha, const Float *x);

        inline int mod(int x, int y) {
            return ((x % y) + y) % y;
        }

        inline Float hp(int x, int y) {
            return values(mod(x, height), mod(y, width));
        }
        
        inline Float hpx(int x, int y) {
            return (values(mod(x + 1, height), mod(y, width)) - values(mod(x - 1, height), mod(y, width))) / 2.0;
        }
        
        inline Float hpy(int x, int y) {
            return (values(mod(x, height), mod(y + 1, width)) - values(mod(x, height), mod(y - 1, width))) / 2.0;
        }
        
        inline Float hpxy(int x, int y) {
            return (hp(x + 1, y + 1) - hp(x + 1, y) - hp(x, y + 1) + 2.0 * hp(x, y) - hp(x - 1, y) - hp(x, y - 1) + hp(x - 1, y - 1)) / 2.0;
        }

        inline FloatD hpDiff(int x, int y) {
            return valuesDiff[mod(x, height) * width + mod(y, width)];
        }
        
        inline FloatD hpxDiff(int x, int y) {
            return (valuesDiff[mod(x + 1, height) * width + mod(y, width)] - valuesDiff[mod(x - 1, height) * width + mod(y, width)]) / 2.0;
        }
        
        inline FloatD hpyDiff(int x, int y) {
            return (valuesDiff[mod(x, height) * width + mod(y + 1, width)] - valuesDiff[mod(x, height) * width + mod(y - 1, width)]) / 2.0;
        }
        
        inline FloatD hpxyDiff(int x, int y) {
            return (hpDiff(x + 1, y + 1) - hpDiff(x + 1, y) - hpDiff(x, y + 1) + 2.0 * hpDiff(x, y) - hpDiff(x - 1, y) - hpDiff(x, y - 1) + hpDiff(x - 1, y - 1)) / 2.0;
        }
};

#endif
