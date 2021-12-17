#ifndef HEIGHTFIELD_H
#define HEIGHTFIELD_H

#include <iostream>
#include <cmath>
#include <cstdio>
#include "gaborkernel.h"
#include "helpers.h"

using namespace std;

// Suppose it starts from (0, 0) and extends to positive (w * mTexelWidth, h * mTexelWidth) and is tiling along both axes.
class Heightfield {
    public:
        Heightfield() {}
        Heightfield(double *heightfieldArray, int width, int height, Float texelWidth = 1.0, Float vertScale = 1.0) : mHeightfieldArray(heightfieldArray), width(width), height(height), mTexelWidth(texelWidth), mVertScale(vertScale) {}

        // Bicubic interpolation.
        Float getValue(Float x, Float y);
        // Bicubic interpolation. u and v are from 0 to 1.
        Float getValueUV(Float u, Float v);

        GaborKernel g(int i, int j, Float F, Float lambda);
        Vector2 n(Float i, Float j);
    public:
        double *mHeightfieldArray;
        int width, height;
        Float mTexelWidth;  // in microns.
        Float mVertScale;

    private:
        void computeCoeff(Float *alpha, const Float *x);

        inline int mod(int x, int y) {
            return ((x % y) + y) % y;
        }

        inline Float hp(int x, int y) {
            return mHeightfieldArray[mod(x, height) * height + mod(y, width)];
        }
        
        inline Float hpx(int x, int y) {
            return (mHeightfieldArray[mod(x + 1, height) * height + mod(y, width)] - mHeightfieldArray[mod(x - 1, height) * height + mod(y, width)]) / 2.0;
        }
        
        inline Float hpy(int x, int y) {
            return (mHeightfieldArray[mod(x, height) * height + mod(y + 1, width)] - mHeightfieldArray[mod(x, height) * height + mod(y - 1, width)]) / 2.0;
        }
        
        inline Float hpxy(int x, int y) {
            return (hp(x + 1, y + 1) - hp(x + 1, y) - hp(x, y + 1) + 2.0 * hp(x, y) - hp(x - 1, y) - hp(x, y - 1) + hp(x - 1, y - 1)) / 2.0;
        }
};

#endif
