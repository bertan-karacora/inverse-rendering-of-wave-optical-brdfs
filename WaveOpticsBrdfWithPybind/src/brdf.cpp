#include "brdf.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(brdf, m) {
    m.def("init", [] () {
        srand(time(NULL));
        SpectrumInit();
    });

    m.def("makeQuery", [] (double x, double y, double sigma, double omega_i_x, double omega_i_y, double lbd) {
        Query query;
        query.mu_p = Vector2(x, y);
        query.sigma_p = sigma;
        query.omega_i = Vector3(omega_i_x, omega_i_y, 1.0).normalized().head(2);
        query.lambda = lbd;
        return query;
    });

    py::class_<Query>(m, "Query")
        .def_readwrite("mu_p", &Query::mu_p)
        .def_readwrite("sigma_p", &Query::sigma_p)
        .def_readwrite("omega_i", &Query::omega_i)
        .def_readwrite("lambda", &Query::lambda);

    py::class_<BrdfImage>(m, "BrdfImage")
        .def_readwrite("r", &BrdfImage::r)
        .def_readwrite("g", &BrdfImage::g)
        .def_readwrite("b", &BrdfImage::b);

    py::class_<BrdfBase>(m, "BrdfBase")
        .def("queryBrdf", &BrdfBase::queryBrdf)
        .def("genBrdfImage", &BrdfBase::genBrdfImage);

    py::class_<GeometricBrdf>(m, "GeometricBrdf")
        .def(py::init<>())
        .def(py::init<Heightfield *, int>())
        .def("genNdfImage", &GeometricBrdf::genNdfImage)
        .def("queryBrdf", &GeometricBrdf::queryBrdf)
        .def("genBrdfImage", &GeometricBrdf::genBrdfImage);

    py::class_<WaveBrdfAccel>(m, "WaveBrdfAccel")
        .def(py::init<>())
        .def(py::init<string, GaborBasis, int, int, Float>())
        .def("queryIntegral", &WaveBrdfAccel::queryIntegral)
        .def("queryBrdf", &WaveBrdfAccel::queryBrdf)
        .def("genBrdfImage", &WaveBrdfAccel::genBrdfImage);
}

comp WaveBrdfAccel::queryIntegral(const Query &query, int layer, int xIndex, int yIndex) {
    Float C3 = 2.0f;
    if (diff_model == "GHS" || diff_model == "RGHS" || diff_model == "Kirchhoff") {
        Float oDotN = sqrt(abs(1.0f - query.omega_o.dot(query.omega_o)));
        Float iDotN = sqrt(abs(1.0f - query.omega_i.dot(query.omega_i)));
        C3 = oDotN + iDotN;
    }

    if (layer == 0) {
        Vector2 omega_a = (query.omega_i + query.omega_o) / 2.0;
        Vector2 m_k(Float((xIndex + 0.5) * texelWidth), Float((yIndex + 0.5) * texelWidth));

        Float period = texelWidth * height;
        Float pDistSqr = distSqrPeriod(m_k, query.mu_p, period);
        if (pDistSqr > (3.0 * query.sigma_p) * (3.0 * query.sigma_p))
            return comp(Float(0.0), Float(0.0));

        Vector2 uQuery = 2.0 * omega_a / query.lambda;
        AABB &aabb = gaborBasis.angularBB[layer][xIndex][yIndex];
        Vector2 abLambda(aabb.xMin / query.lambda, aabb.yMin / query.lambda);
        
        abLambda *= C3 / 2.0f;

        Float sigma_k = texelWidth / SCALE_FACTOR;
        Float aSigma = 1.0 / (2.0 * M_PI * sigma_k);
        Float aDistSqr = (abLambda - uQuery).squaredNorm();
        if (aDistSqr > (3.0 * aSigma) * (3.0 * aSigma))
            return comp(Float(0.0), Float(0.0));

        Float C2 = 1.0f;
        if (diff_model == "Kirchhoff") {
            Vector2 HPrime = gaborBasis.gaborKernelPrime[xIndex][yIndex].aInfo / 2.0f;
            Vector3 m(-HPrime(0), -HPrime(1), 1.0);
            m.normalize();

            Float oDotN = sqrt(abs(1.0f - query.omega_o.dot(query.omega_o)));
            Float iDotN = sqrt(abs(1.0f - query.omega_i.dot(query.omega_i)));
            Vector3 omega_o(query.omega_o(0), query.omega_o(1), oDotN);
            Vector3 omega_i(query.omega_i(0), query.omega_i(1), iDotN);

            C2 = (omega_o + omega_i).dot(m) / m(2);
        }

        GaborKernelPrime gp = gaborBasis.gaborKernelPrime[xIndex][yIndex];
        gp.aInfo *= C3 / 2.0f;
        gp.cInfo *= C3 / 2.0f;

        Float w_mk = Float(1.0 / (sqrt(M_PI) * query.sigma_p)) * exp(-pDistSqr / (2.0 * query.sigma_p * query.sigma_p));
        GaborKernel g = gp.toGaborKernel(query.lambda);
        return w_mk * C2 * g.xform(2.0 * omega_a / query.lambda);
    }

    // Reject the node positionally.
    int layerScale = floor(pow(2.0, layer) + 1e-4);
    AABB positionalBB((xIndex + 0.0) * layerScale * texelWidth,
                      (xIndex + 1.0) * layerScale * texelWidth,
                      (yIndex + 0.0) * layerScale * texelWidth,
                      (yIndex + 1.0) * layerScale * texelWidth);
    AABB queryBB(query.mu_p(0) - 3.0 * query.sigma_p, query.mu_p(0) + 3.0 * query.sigma_p,
                 query.mu_p(1) - 3.0 * query.sigma_p, query.mu_p(1) + 3.0 * query.sigma_p);
    Float period = texelWidth * height;
    if (!intersectAABBRot(positionalBB, queryBB, period))
        return comp(Float(0.0), Float(0.0));

    // Reject the node angularly.
    Vector2 omega_a = (query.omega_i + query.omega_o) / 2.0;
    Vector2 uQuery = 2.0 * omega_a / query.lambda;
    AABB &aabb = gaborBasis.angularBB[layer][xIndex][yIndex];
    AABB aabbLambda(aabb.xMin * C3 / 2.0f / query.lambda, aabb.xMax * C3 / 2.0f / query.lambda,
                    aabb.yMin * C3 / 2.0f / query.lambda, aabb.yMax * C3 / 2.0f / query.lambda);

    Float sigma_k = texelWidth / SCALE_FACTOR;
    Float sigma = 1.0 / (2.0 * M_PI * sigma_k);

    AABB aabbLambdaExpanded(aabbLambda.xMin - sigma * 3.0,
                            aabbLambda.xMax + sigma * 3.0,
                            aabbLambda.yMin - sigma * 3.0,
                            aabbLambda.yMax + sigma * 3.0);

    if (!insideAABB(aabbLambdaExpanded, uQuery))
        return comp(Float(0.0), Float(0.0));

    return queryIntegral(query, layer - 1, xIndex * 2 + 0, yIndex * 2 + 0) +
           queryIntegral(query, layer - 1, xIndex * 2 + 1, yIndex * 2 + 0) +
           queryIntegral(query, layer - 1, xIndex * 2 + 1, yIndex * 2 + 1) +
           queryIntegral(query, layer - 1, xIndex * 2 + 0, yIndex * 2 + 1);
}

Float WaveBrdfAccel::queryBrdf(const Query &query) {
    // Move the query center to the first HF period.
    Query q = query;
    Float hfHeightWorld = texelWidth * height;
    Float hfWidthWorld = texelWidth * width;
    q.mu_p(0) -= ((int) floor(q.mu_p(0) / hfHeightWorld)) * hfHeightWorld;
    q.mu_p(1) -= ((int) floor(q.mu_p(1) / hfWidthWorld)) * hfWidthWorld;

    // Traverse the hierarchy to calculate the inner integral.
    comp I = queryIntegral(q, gaborBasis.topLayer, 0, 0);

    // Update corresponding C1, C2 and C3 based on different diffraction models.
    Float oDotN = sqrt(abs(1.0f - query.omega_o.dot(query.omega_o)));
    Float iDotN = sqrt(abs(1.0f - query.omega_i.dot(query.omega_i)));
    Float C1 = 0.0f;

    if (diff_model == "OHS" || diff_model == "GHS") {
        C1 = oDotN / (query.lambda * query.lambda * iDotN);
    } else if (diff_model == "ROHS" || diff_model == "RGHS") {
        C1 = (iDotN + oDotN) * (iDotN + oDotN) / (query.lambda * query.lambda * 4.0f * iDotN * oDotN);
    } else if (diff_model == "Kirchhoff") {
        C1 = 1.0f / (query.lambda * query.lambda * 4.0f * iDotN * oDotN);
    }
    return C1 * pow(abs(I), 2.0f);
}

BrdfImage WaveBrdfAccel::genBrdfImage(const Query &query, int resolution) {
    MatrixXf brdfImage_r(resolution, resolution);
    MatrixXf brdfImage_g(resolution, resolution);
    MatrixXf brdfImage_b(resolution, resolution);

    if (query.lambda != 0.0) {
        // Single wavelength.
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < resolution; i++) {
            printf("Generating BRDF image: row %d...\n", i);
            for (int j = 0; j < resolution; j++) {
                Vector2 omega_o((i + 0.5) / resolution * 2.0 - 1.0,
                                (j + 0.5) / resolution * 2.0 - 1.0);
                Float brdfValue;

                if (omega_o.norm() > 1.0) {
                   brdfValue = 0.0;
                } else {
                    Query q = query;
                    q.omega_o = omega_o;
                    brdfValue = queryBrdf(q);
                }

                if (std::isnan(brdfValue)) {
                    brdfValue = 0.0;
                }

                brdfImage_r(i, j) = brdfValue;
                brdfImage_g(i, j) = brdfValue;
                brdfImage_b(i, j) = brdfValue;
            }
        }
    } else {
        // Multiple wavelengths.
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < resolution; i++) {
            printf("Generating BRDF image: row %d...\n", i);
            for (int j = 0; j < resolution; j++) {
                Vector2 omega_o((i + 0.5) / resolution * 2.0 - 1.0,
                                (j + 0.5) / resolution * 2.0 - 1.0);

                vector<float> spectrumSamples;

                for (int k = 0; k < SPECTRUM_SAMPLES; k++) {
                    Float brdfValue;

                    if (omega_o.norm() > 1.0) {
                        brdfValue = 0.0;
                    } else {
                        Query q = query;
                        q.omega_o = omega_o;
                        q.lambda = (k + 0.5) / SPECTRUM_SAMPLES * (0.83 - 0.36) + 0.36;
                        brdfValue = queryBrdf(q);
                    }

                    if (std::isnan(brdfValue))
                        brdfValue = 0.0;

                    spectrumSamples.push_back(brdfValue);
                }

                float r, g, b;
                SpectrumToRGB(spectrumSamples, r, g, b);

                brdfImage_r(i, j) = r;
                brdfImage_g(i, j) = g;
                brdfImage_b(i, j) = b;
            }
        }
    }
    
    BrdfImage brdfImage;
    brdfImage.r = brdfImage_r;
    brdfImage.g = brdfImage_g;
    brdfImage.b = brdfImage_b;

    return brdfImage;
}

Float GeometricBrdf::queryBrdf(const Query &query) {
    cout << "Not implemented" << endl;
    return 0.0;
}

inline Vector2 sampleGauss2d(Float r1, Float r2) {
    // https://en.wikipedia.org/wiki/Box-Muller_transform
    Float tmp = std::sqrt(-2 * std::log(r1));
    Float x = tmp * std::cos(2 * Float(M_PI) * r2);
    Float y = tmp * std::sin(2 * Float(M_PI) * r2);
    return Vector2(x, y);
}

MatrixXf GeometricBrdf::genNdfImage(const Query &query, int resolution) {
    int N = (int) std::sqrt(sampleNum);
    const Float intrinsicRoughness = Float(1) / N;
    int *inds = new int[N * N];

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // sample query Gaussian, stratified
            Float rx = (i + randUniform<Float>()) / N;
            Float ry = (j + randUniform<Float>()) / N;
            Vector2 g = sampleGauss2d(rx, ry);

            // look up normal
            Vector2 x = g * query.sigma_p / heightfield->texelWidth;
            x += query.mu_p;
            Vector2 normal = heightfield->n(x[0], x[1]);

            // intrinsic roughness
            normal += intrinsicRoughness * sampleGauss2d(randUniform<Float>(), randUniform<Float>());

            int xi = (int)((1 + normal[0]) / 2 * resolution);
            int yi = (int)((1 - normal[1]) / 2 * resolution);
            if (xi < 0 || xi >= resolution || yi < 0 || yi >= resolution) continue;
            inds[i*N + j] = yi * resolution + xi;
        }
    }

    // bin the samples using indices computed above
    int npix = resolution * resolution;
    int *bins = new int[npix];
    memset(bins, 0, npix * sizeof(int));
    for (int i = 0; i < N*N; i++) bins[inds[i]]++;

    Vector3 *ndfImage = new Vector3[npix];
    Float scale = Float(npix) / (4 * N * N);
    for (int i = 0; i < npix; i++) ndfImage[i] = Vector3::Constant(scale * bins[i]);

    delete[] inds;
    delete[] bins;
    double* ndfIm = (double*) ndfImage;

    MatrixXf values(heightfield->width, heightfield->height);
    for (int i = 0; i < heightfield->height; i++) {
        for (int j = 0; j < heightfield->width; j++) {
            values(i, j) = ndfIm[i * heightfield->width + j];
        }
    }

    return values;
}

BrdfImage GeometricBrdf::genBrdfImage(const Query &query, int resolution) {
    const int ndfResolution = resolution * 2;
    MatrixXf ndfImage = genNdfImage(query, ndfResolution);

    BrdfImage brdfImage;
    brdfImage.r = MatrixXf(resolution, resolution);
    brdfImage.g = MatrixXf(resolution, resolution);
    brdfImage.b = MatrixXf(resolution, resolution);

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            const int numSamples = 16;
            for (int k = 0; k < numSamples; k++) {
                Vector2 omega_o((i + randUniform<Float>()) / resolution * 2.0 - 1.0,
                                (j + randUniform<Float>()) / resolution * 2.0 - 1.0);

                Vector3 omega_i_3(query.omega_i(0), query.omega_i(1), sqrt(1.0 - query.omega_i.norm()));
                Vector3 omega_o_3(omega_o(0), omega_o(1), sqrt(1.0 - omega_o.norm()));
                Vector2 omega_h = (omega_i_3 + omega_o_3).normalized().head(2);

                Vector2 xy = (omega_h + Vector2(1.0, 1.0)) / 2.0 * ndfResolution;

                int xInt = (int)(xy(0));
                int yInt = (int)(xy(1));
                if (xInt < 0 || xInt >= ndfResolution ||
                    yInt < 0 || yInt >= ndfResolution)
                    continue;

                Float D = ndfImage(xInt, yInt);
                Float F = 1.0;
                Float G = 1.0;
                Float cosThetaI = sqrt(abs(1.0 - query.omega_i.dot(query.omega_i)));
                Float cosThetaO = sqrt(abs(1.0 - omega_o.dot(omega_o)));
                Float brdfValue = D * F * G / (4.0 * cosThetaI * cosThetaO);

                if (std::isnan(brdfValue / numSamples))
                    continue;

                brdfImage.r(i, j) += brdfValue / numSamples;
                brdfImage.g(i, j) += brdfValue / numSamples;
                brdfImage.b(i, j) += brdfValue / numSamples;
            }
        }
    }

    return brdfImage;
}
