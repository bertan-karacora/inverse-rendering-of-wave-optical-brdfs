#ifndef GABORKERNELDiff_H
#define GABORKERNELDiff_H

#include "helpers.h"

class GaborKernelDiff {
    public:
        GaborKernelDiff() {}
        GaborKernelDiff(Vector2fD mu_, Float sigma_, Vector2fD a_, ComplexfD C_ = ComplexfD(FloatD(1.0), FloatD(0.0))) :
                    mu(mu_), sigma(sigma_), a(a_), C(C_) {}

        ComplexfD eval(Vector2 s);
        ComplexfD xform(Vector2 u);

    public:
        Vector2fD mu;
        Float sigma;
        Vector2fD a;
        ComplexfD C;
};

class GaborKernelPrimeDiff {
    public:
        GaborKernelPrimeDiff() {}
        GaborKernelPrimeDiff(Vector2 mu_, Float sigma_, Vector2fD aInfo_, FloatD cInfo_) :
                            mu(mu_), sigma(sigma_), aInfo(aInfo_), cInfo(cInfo_) {}

        GaborKernelDiff toGaborKernel(Float lambda);
        Vector2fD getFFTCenter(Float lambda);     // -a = -aInfo / lambda.
        Float getFFTWidth();                    // 1 / (2 pi sigma).

    public:
        Vector2 mu;
        Float sigma;
        Vector2fD aInfo;
        FloatD cInfo;
};

#endif
