#ifndef HEIGHTFIELDDiff_H
#define HEIGHTFIELDDiff_H

#include "helpers.h"
#include "gaborkerneldiff.h"

class HeightfieldDiff;

class GaborBasisDiff {
    public:
        GaborBasisDiff() {};
        GaborBasisDiff(const HeightfieldDiff &hf);

    public:
        vector<vector<GaborKernelPrimeDiff>> gaborKernelPrime;
        vector<vector<vector<AABB>>> angularBB;
        int topLayer;
};

// Suppose it starts from (0, 0) and extends to positive (w * mTexelWidth, h * mTexelWidth) and is tiling along both axes.
class HeightfieldDiff {
    public:
        HeightfieldDiff() {};
        HeightfieldDiff(Eigen::MatrixXf values, int width, int height, Float texelWidth, Float vertScale);
        HeightfieldDiff(GaborBasisDiff gaborBasisDiff, int width, int height, Float texelWidth, Float vertScale);

        FloatD getValue(Float x, Float y);
        FloatD getValueUV(Float u, Float v);

        GaborKernelDiff g(int i, int j, Float F, Float lambda);
        Vector2fD n(Float i, Float j);

    public:
        FloatXXD values;
        int width, height;
        Float texelWidth;   // in microns.
        Float vertScale;

    private:
        inline int mod(int x, int y) {
            return ((x % y) + y) % y;
        }

        inline FloatD hp(int x, int y) {
            return values[mod(x, height)][mod(y, width)];
        }
        
        inline FloatD hpx(int x, int y) {
            return (values[mod(x + 1, height)][mod(y, width)] - values[mod(x - 1, height)][mod(y, width)]) / 2.0f;
        }
        
        inline FloatD hpy(int x, int y) {
            return (values[mod(x, height)][mod(y + 1, width)] - values[mod(x, height)][mod(y - 1, width)]) / 2.0f;
        }
        
        inline FloatD hpxy(int x, int y) {
            return (hp(x + 1, y + 1) - hp(x + 1, y) - hp(x, y + 1) + 2.0f * hp(x, y) - hp(x - 1, y) - hp(x, y - 1) + hp(x - 1, y - 1)) / 2.0f;
        }
};

#endif
