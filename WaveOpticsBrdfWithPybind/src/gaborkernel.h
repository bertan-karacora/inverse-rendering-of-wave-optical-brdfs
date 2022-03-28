#ifndef GABOR_KERNEL_H
#define GABOR_KERNEL_H

#include "helpers.h"

class GaborKernel {
    public:
        GaborKernel() {}
        GaborKernel(Vector2 mu_, Float sigma_, Vector2 a_, comp C_ = comp(Float(1.0), Float(0.0))) :
                    mu(mu_), sigma(sigma_), a(a_), C(C_) {}

        comp eval(Vector2 s);
        comp xform(Vector2 u);

    public:
        Vector2 mu;
        Float sigma;
        Vector2 a;
        comp C;
};

class GaborKernelPrime {
    public:
        GaborKernelPrime() {}
        GaborKernelPrime(Vector2 mu_, Float sigma_, Vector2 aInfo_, Float cInfo_) :
                            mu(mu_), sigma(sigma_), aInfo(aInfo_), cInfo(cInfo_) {}

        GaborKernel toGaborKernel(Float lambda);
        Vector2 getFFTCenter(Float lambda);     // -a = -aInfo / lambda.
        Float getFFTWidth();                    // 1 / (2 pi sigma).

    public:
        Vector2 mu;
        Float sigma;
        Vector2 aInfo;
        Float cInfo;
};

#endif
