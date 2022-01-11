#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "heightfield.h"
#include "exrimage.h"

namespace py = pybind11;

PYBIND11_MODULE(visualize, m) {
    m.def("readImage", [](
        string heightfieldFilename,
        double texelWidth,
        double vertScale
    ) {
        EXRImage heightfieldImage(heightfieldFilename.c_str());
        Heightfield heightfield(heightfieldImage.values, heightfieldImage.width, heightfieldImage.height, texelWidth, vertScale);
        return pybind11::array_t<double>({heightfieldImage.width, heightfieldImage.height}, heightfield.mHeightfieldArray);
    },
    "Read image into heightfield array.");

    m.def("genImage", [](
        py::array_t<double> brdfValues, // BRDF values.
        string outputFilename,          // Output filename.
        double resolution               // Resolution.
    ) {
        py::buffer_info buf = brdfValues.request();
        EXRImage::writeImage(static_cast<double *>(buf.ptr), outputFilename.c_str(), resolution, resolution);
        return outputFilename;
    },
    "Generate image from brdf values.");
}
