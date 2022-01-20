#include "gaborkernel.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

using namespace Eigen;
namespace py = pybind11;

PYBIND11_MODULE(gaborkernel, m) {
    py::class_<GaborKernel>(m, "GaborKernel")
        .def(py::init<>())
        .def(py::init<Vector2, Float, Vector2, comp>())
        .def_readwrite("mu", &GaborKernel::mu)
        .def_readwrite("sigma", &GaborKernel::sigma)
        .def_readwrite("a", &GaborKernel::a)
        .def_readwrite("C", &GaborKernel::C)
        .def("eval", &GaborKernel::eval)
        .def("xform", &GaborKernel::xform);

    py::class_<GaborKernelPrime>(m, "GaborKernelPrime")
        .def(py::init<>())
        .def(py::init<Vector2, Float, Vector2, Float>())
        .def_readwrite("mu", &GaborKernelPrime::mu)
        .def_readwrite("sigma", &GaborKernelPrime::sigma)
        .def_readwrite("aInfo", &GaborKernelPrime::aInfo)
        .def_readwrite("cInfo", &GaborKernelPrime::cInfo)
        .def("toGaborKernel", &GaborKernelPrime::toGaborKernel)
        .def("getFFTCenter", &GaborKernelPrime::getFFTCenter)
        .def("getFFTWidth", &GaborKernelPrime::getFFTWidth);
}

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
