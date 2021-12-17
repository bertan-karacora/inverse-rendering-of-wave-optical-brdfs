#ifndef WDF_H
#define WDF_H

#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <vector>
#include <Eigen/Dense>
#include "heightfield.h"
#include "gaborkernel.h"

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
        BrdfBase(Heightfield *heightfield) : mHeightfield(heightfield) {}
    public:
        virtual Float queryBrdf(const Query &query) = 0;
        virtual Float* genBrdfImage(const Query &query, int resolution) = 0;

    protected:
        Heightfield *mHeightfield;
};

class GeometricBrdf: public BrdfBase {
    public:
        GeometricBrdf() {}
        GeometricBrdf(Heightfield *heightfield, int sampleNum = 10000000) : mHeightfield(heightfield), mSampleNum(sampleNum) {}
        Float* genNdfImage(const Query &query, int resolution);
        virtual Float queryBrdf(const Query &query);
        virtual Float* genBrdfImage(const Query &query, int resolution);
    protected:
        Heightfield *mHeightfield;
    private:
        int mSampleNum;
};


class WaveBrdf: public BrdfBase {
    public:
        WaveBrdf() {}
        WaveBrdf(Heightfield *heightfield) : mHeightfield(heightfield) {}
        virtual Float queryBrdf(const Query &query);
        virtual Float* genBrdfImage(const Query &query, int resolution);
    protected:
        Heightfield *mHeightfield;
};

class WaveBrdfAccel: public WaveBrdf {
    public:
        WaveBrdfAccel() {}

        WaveBrdfAccel(Heightfield *heightfield, string method);

        comp queryIntegral(const Query &query, int layer, int xIndex, int yIndex);
        virtual Float queryBrdf(const Query &query);
        virtual Float* genBrdfImage(const Query &query, int resolution);

    protected:
        Heightfield *mHeightfield;

    private:
        vector<vector<GaborKernelPrime>> gaborKernelPrime;
        vector<vector<vector<AABB>>> angularBB;
        int mTopLayer;

    public:
        string mMethod;
};

#endif
