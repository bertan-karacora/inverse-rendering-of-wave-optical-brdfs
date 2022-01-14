#include "gaborkernel.h"

comp GaborKernel::eval(Vector2 s) {
    return C * G(s, mu, sigma) * cnis(Float(2.0 * M_PI * a.dot(s)));
}

comp GaborKernel::xform(Vector2 u) {
    Float sigmaPrime = Float(1.0 / (2.0 * M_PI * sigma));
    comp CPrime = C * Float(1.0 / (2.0 * M_PI * sigma * sigma)) * cnis(Float(2.0 * M_PI * a.dot(mu)));
    GaborKernel g(-a, sigmaPrime, mu, CPrime);
    return g.eval(u);
}

GaborKernel GaborKernelPrime::toGaborKernel(Float lambda) {
    Float l = sigma * SCALE_FACTOR;
    comp C = l * l * cnis(4.0 * M_PI / lambda * cInfo);
    Vector2 a = aInfo / lambda;

    return GaborKernel(mu, sigma, a, C);
}

Vector2 GaborKernelPrime::getFFTCenter(Float lambda) {
    return -aInfo / lambda;
}

Float GaborKernelPrime::getFFTWidth() {
    return 1.0 / (2.0 * M_PI * sigma);
}
