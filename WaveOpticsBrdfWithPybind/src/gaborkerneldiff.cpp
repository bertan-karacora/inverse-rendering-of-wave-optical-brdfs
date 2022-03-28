#include "gaborkerneldiff.h"


PYBIND11_MODULE(gaborkerneldiff, m) {
    py::class_<GaborKernelDiff>(m, "GaborKernelDiff")
        .def(py::init<>())
        .def(py::init<Vector2fD, Float, Vector2fD, ComplexfD>())
        .def_readwrite("mu", &GaborKernelDiff::mu)
        .def_readwrite("sigma", &GaborKernelDiff::sigma)
        .def_readwrite("a", &GaborKernelDiff::a)
        .def_readwrite("C", &GaborKernelDiff::C)
        .def("eval", &GaborKernelDiff::eval)
        .def("xform", &GaborKernelDiff::xform);

    py::class_<GaborKernelPrimeDiff>(m, "GaborKernelPrimeDiff")
        .def(py::init<>())
        .def(py::init<Vector2, Float, Vector2fD, FloatD>())
        .def_readwrite("mu", &GaborKernelPrimeDiff::mu)
        .def_readwrite("sigma", &GaborKernelPrimeDiff::sigma)
        .def_readwrite("aInfo", &GaborKernelPrimeDiff::aInfo)
        .def_readwrite("cInfo", &GaborKernelPrimeDiff::cInfo)
        .def("toGaborKernel", &GaborKernelPrimeDiff::toGaborKernel)
        .def("getFFTCenter", &GaborKernelPrimeDiff::getFFTCenter)
        .def("getFFTWidth", &GaborKernelPrimeDiff::getFFTWidth);
}


ComplexfD GaborKernelDiff::eval(Vector2 s) {
    return C * G(s, mu, sigma) * cnis(2.0 * M_PI * enoki::dot(a, enoki::Array<Float, 2>(s[0], s[1])));
}

ComplexfD GaborKernelDiff::xform(Vector2 u) {
    Float sigmaPrime = Float(1.0 / (2.0 * M_PI * sigma));
    ComplexfD CPrime = C * Float(1.0 / (2.0 * M_PI * sigma * sigma)) * cnis(2.0 * M_PI * enoki::dot(a, mu));
    GaborKernelDiff g(-a, sigmaPrime, mu, CPrime);
    return g.eval(u);
}

GaborKernelDiff GaborKernelPrimeDiff::toGaborKernel(Float lambda) {
    Float l = sigma * SCALE_FACTOR;
    ComplexfD C = l * l * cnis(4.0 * M_PI / lambda * cInfo);
    Vector2fD a = aInfo / lambda;

    return GaborKernelDiff(enoki::Array<Float, 2>(mu[0], mu[1]), sigma, a, C);
}

Vector2fD GaborKernelPrimeDiff::getFFTCenter(Float lambda) {
    return -aInfo / lambda;
}

Float GaborKernelPrimeDiff::getFFTWidth() {
    return 1.0 / (2.0 * M_PI * sigma);
}
