#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "heightfield.h"
#include "waveBrdf.h"
#include "spectrum.h"

using namespace std;
using namespace Eigen;
namespace py = pybind11;

PYBIND11_MODULE(genBrdf, m) {
    m.def("genBrdf", [](
        py::array_t<double> heightfieldArray,   // Heightfield
        double texelWidth,                      // The width of a texel in microns on the heightfield.
        double vertScale,                       // The vertical scaling factor of the heightfield.
        double x,                               // Center x of the Gaussian footprint.
        double y,                               // Center y of the Gaussian footprint.
        double sigma,                           // Size (1 sigma) of the Gaussian footprint.
        string method,                          // Method. Choose between "Geom" and "Wave".
        int sampleNum,                          // Number of binning samples. Only valid for geometric optics.
        string diffModel,                       // Diffraction model. Choose between "OHS", "GHS", "ROHS", "RGHS" and "Kirchhoff". And expect only subtle difference.
        double lambda,                          // Wavelength in microns. Once set, single wavelength mode is ON.
        double omega_i_x,                       // Incoming light's x coordinate (assuming z = 1).
        double omega_i_y,                       // Incoming light's y coordinate (assuming z = 1).
        int resolution                          // Output resolution.
    ) {
        srand(time(NULL));

        SpectrumInit();
        py::buffer_info buf = heightfieldArray.request();
        Heightfield heightfield(static_cast<double *>(buf.ptr), buf.shape[0], buf.shape[1], texelWidth, vertScale);

        Query query;
        query.mu_p = Vector2(x, y);
        query.sigma_p = sigma;
        query.omega_i = Vector3(omega_i_x, omega_i_y, 1.0).normalized().head(2);
        query.lambda = lambda;

        Float *image;
        /*
        if (method == "Geom") {
            GeometricBrdf geometricBrdf(&heightfield, sampleNum);
            image = geometricBrdf.genBrdfImage(query, resolution);
            //EXRImage::writeImage(brdfImage, outputFilename, outputResolution, outputResolution);
        } else if (method == "Wave") {
            WaveBrdfAccel waveBrdfAccel(&heightfield, diffModel);
            image = waveBrdfAccel.genBrdfImage(query, resolution);
        } else if (method == "GeomNdf") {
            GeometricBrdf geometricBrdf(&heightfield, sampleNum);
            image = geometricBrdf.genNdfImage(query, resolution);
        }
        delete[] image;
        */
        return heightfield.mHeightfieldArray;
    },
    "Generate BRDF");
}
