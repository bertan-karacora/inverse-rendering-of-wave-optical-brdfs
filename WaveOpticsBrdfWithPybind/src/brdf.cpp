#include "brdf.h"


PYBIND11_MODULE(brdf, m) {
    m.def("initialize", [] () {
        cout << "Initalizing..." << endl;
        srand(time(NULL));
        SpectrumInit();
        cout << "Initalizing finished!" << endl;
    });

    m.def("makeQuery", [] (double x, double y, double sigma, double omega_i_x, double omega_i_y, double lbd) {
        cout << "Generating query..." << endl;
        Query query;
        query.mu_p = Vector2(x, y);
        query.sigma_p = sigma;
        query.omega_i = Vector3(omega_i_x, omega_i_y, 1.0).normalized().head(2);
        query.lambda = lbd;
        cout << "Generating query finished!" << endl;
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
        .def_readwrite("b", &BrdfImage::b)
        .def_readwrite("grad", &BrdfImage::grad);

    py::class_<WaveBrdfAccel>(m, "WaveBrdfAccel")
        .def(py::init<>())
        .def(py::init<string, int, int, Float, int>())
        .def("queryIntegral", &WaveBrdfAccel::queryIntegral)
        .def("queryBrdf", &WaveBrdfAccel::queryBrdf)
        .def("genBrdfImage", &WaveBrdfAccel::genBrdfImage)
        .def("genBrdfImageDiff", &WaveBrdfAccel::genBrdfImageDiff);
}


comp WaveBrdfAccel::queryIntegral(const Query &query, const GaborBasis &gaborBasis, int layer, int xIndex, int yIndex) {
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
        AABB aabb = gaborBasis.angularBB[layer][xIndex][yIndex];
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
    AABB aabb = gaborBasis.angularBB[layer][xIndex][yIndex];
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

    return queryIntegral(query, gaborBasis, layer - 1, xIndex * 2 + 0, yIndex * 2 + 0) +
           queryIntegral(query, gaborBasis, layer - 1, xIndex * 2 + 1, yIndex * 2 + 0) +
           queryIntegral(query, gaborBasis, layer - 1, xIndex * 2 + 1, yIndex * 2 + 1) +
           queryIntegral(query, gaborBasis, layer - 1, xIndex * 2 + 0, yIndex * 2 + 1);
}

ComplexfD WaveBrdfAccel::queryIntegralDiff(const Query &query, const GaborBasisDiff &gaborBasis, int layer, int xIndex, int yIndex) {
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
            return ComplexfD(Float(0.0), Float(0.0));

        Vector2 uQuery = 2.0 * omega_a / query.lambda;
        AABB aabb = gaborBasis.angularBB[layer][xIndex][yIndex];
        Vector2 abLambda(aabb.xMin / query.lambda, aabb.yMin / query.lambda);
        
        abLambda *= C3 / 2.0f;

        Float sigma_k = texelWidth / SCALE_FACTOR;
        Float aSigma = 1.0f / (2.0f * M_PI * sigma_k);
        Float aDistSqr = (abLambda - uQuery).squaredNorm();
        if (aDistSqr > (3.0f * aSigma) * (3.0f * aSigma))
            return ComplexfD(Float(0.0), Float(0.0));

        FloatD C2 = 1.0f;
        if (diff_model == "Kirchhoff") {
            Vector2fD HPrime = gaborBasis.gaborKernelPrime[xIndex][yIndex].aInfo / 2.0f;
            Vector3fD m(-HPrime[0], -HPrime[1], 1.0f);
            enoki::normalize(m);

            Float oDotN = sqrt(abs(1.0f - query.omega_o.dot(query.omega_o)));
            Float iDotN = sqrt(abs(1.0f - query.omega_i.dot(query.omega_i)));
            Vector3fD omega_o(query.omega_o(0), query.omega_o(1), oDotN);
            Vector3fD omega_i(query.omega_i(0), query.omega_i(1), iDotN);

            C2 = enoki::dot(omega_o + omega_i, m) / m[2];
        }

        GaborKernelPrimeDiff gp = gaborBasis.gaborKernelPrime[xIndex][yIndex];
        gp.aInfo *= C3 / 2.0f;
        gp.cInfo *= C3 / 2.0f;

        Float w_mk = Float(1.0f / (sqrt(M_PI) * query.sigma_p)) * exp(-pDistSqr / (2.0f * query.sigma_p * query.sigma_p));
        GaborKernelDiff g = gp.toGaborKernel(query.lambda);
        return w_mk * C2 * g.xform(2.0f * omega_a / query.lambda);
    }

    // Reject the node positionally.
    int layerScale = floor(pow(2.0f, layer) + 1e-4);
    AABB positionalBB((xIndex + 0.0) * layerScale * texelWidth,
                      (xIndex + 1.0) * layerScale * texelWidth,
                      (yIndex + 0.0) * layerScale * texelWidth,
                      (yIndex + 1.0) * layerScale * texelWidth);
    AABB queryBB(query.mu_p(0) - 3.0f * query.sigma_p, query.mu_p(0) + 3.0f * query.sigma_p,
                 query.mu_p(1) - 3.0f * query.sigma_p, query.mu_p(1) + 3.0f * query.sigma_p);
    Float period = texelWidth * height;
    if (!intersectAABBRot(positionalBB, queryBB, period))
        return ComplexfD(Float(0.0), Float(0.0));

    // Reject the node angularly.
    Vector2 omega_a = (query.omega_i + query.omega_o) / 2.0;
    Vector2 uQuery = 2.0 * omega_a / query.lambda;
    AABB aabb = gaborBasis.angularBB[layer][xIndex][yIndex];
    AABB aabbLambda(aabb.xMin * C3 / 2.0f / query.lambda, aabb.xMax * C3 / 2.0f / query.lambda,
                    aabb.yMin * C3 / 2.0f / query.lambda, aabb.yMax * C3 / 2.0f / query.lambda);

    Float sigma_k = texelWidth / SCALE_FACTOR;
    Float sigma = 1.0 / (2.0 * M_PI * sigma_k);

    AABB aabbLambdaExpanded(aabbLambda.xMin - sigma * 3.0,
                            aabbLambda.xMax + sigma * 3.0,
                            aabbLambda.yMin - sigma * 3.0,
                            aabbLambda.yMax + sigma * 3.0);

    if (!insideAABB(aabbLambdaExpanded, uQuery))
        return ComplexfD(Float(0.0), Float(0.0));

    return queryIntegralDiff(query, gaborBasis, layer - 1, xIndex * 2 + 0, yIndex * 2 + 0) +
           queryIntegralDiff(query, gaborBasis, layer - 1, xIndex * 2 + 1, yIndex * 2 + 0) +
           queryIntegralDiff(query, gaborBasis, layer - 1, xIndex * 2 + 1, yIndex * 2 + 1) +
           queryIntegralDiff(query, gaborBasis, layer - 1, xIndex * 2 + 0, yIndex * 2 + 1);
}

Float WaveBrdfAccel::queryBrdf(const Query &query, const GaborBasis &gaborBasis) {
    // Move the query center to the first HF period.
    Query q = query;
    Float hfHeightWorld = texelWidth * height;
    Float hfWidthWorld = texelWidth * width;
    q.mu_p(0) -= ((int) floor(q.mu_p(0) / hfHeightWorld)) * hfHeightWorld;
    q.mu_p(1) -= ((int) floor(q.mu_p(1) / hfWidthWorld)) * hfWidthWorld;

    // Traverse the hierarchy to calculate the inner integral.
    comp I = queryIntegral(q, gaborBasis, gaborBasis.topLayer, 0, 0);

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

FloatD WaveBrdfAccel::queryBrdfDiff(const Query &query, const GaborBasisDiff &gaborBasis) {
    // Move the query center to the first HF period.
    Query q = query;
    Float hfHeightWorld = texelWidth * height;
    Float hfWidthWorld = texelWidth * width;
    q.mu_p(0) -= ((int) floor(q.mu_p(0) / hfHeightWorld)) * hfHeightWorld;
    q.mu_p(1) -= ((int) floor(q.mu_p(1) / hfWidthWorld)) * hfWidthWorld;

    // Traverse the hierarchy to calculate the inner integral.
    ComplexfD I = queryIntegralDiff(q, gaborBasis, gaborBasis.topLayer, 0, 0);

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
    return C1 * enoki::pow(enoki::abs(I), 2.0f);
}

BrdfImage WaveBrdfAccel::genBrdfImage(const Query &query, const GaborBasis &gaborBasis) {
    Eigen::MatrixXf brdfImage_r(resolution, resolution);
    Eigen::MatrixXf brdfImage_g(resolution, resolution);
    Eigen::MatrixXf brdfImage_b(resolution, resolution);

    if (query.lambda != 0.0) {
        // Single wavelength.
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < resolution; i++) {
            cout << "\r" << "Generating BRDF image: row " << i << "\t\r" << flush;
            for (int j = 0; j < resolution; j++) {
                Vector2 omega_o((i + 0.5) / resolution * 2.0 - 1.0,
                                (j + 0.5) / resolution * 2.0 - 1.0);
                Float brdfValue;

                if (omega_o.norm() > 1.0) {
                   brdfValue = 0.0;
                } else {
                    Query q = query;
                    q.omega_o = omega_o;
                    brdfValue = queryBrdf(q, gaborBasis);
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
            cout << "\r" << "Generating BRDF image: row " << i << "\t\r" << flush;
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
                        brdfValue = queryBrdf(q, gaborBasis);
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

    cout << "\nGenerating BRDF image finished!" << endl;
    return brdfImage;
}

BrdfImage WaveBrdfAccel::genBrdfImageDiff(const Query &query, const HeightfieldDiff &hf, BrdfImage ref) {
    HeightfieldDiff heightfield = hf;

    for (int i = 0; i < heightfield.height; i++) {
        for (int j = 0; j < heightfield.width; j++) {
            enoki::set_requires_gradient(heightfield.values[i][j]);
        }
    }

    GaborBasisDiff gaborBasis(heightfield);

    Eigen::MatrixXf brdfImage_r(resolution, resolution);
    Eigen::MatrixXf brdfImage_g(resolution, resolution);
    Eigen::MatrixXf brdfImage_b(resolution, resolution);
    Eigen::MatrixXf brdfImage_grad(heightfield.width, heightfield.height);

    // enoki::Array<FloatD, 32 * 32 * 3> mse;
    enoki::Array<FloatD, 32 * 32 * 3> mse_hdr;
    // enoki::Array<FloatD, 16 * 16 * 3> mse_rgb;

    Float eps = 0.01f;

    // #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < resolution; i++) {
        // cout << "Generating BRDF image: row " << i << endl;
        for (int j = 0; j < resolution; j++) {
            Vector2 omega_o((i + 0.5) / resolution * 2.0 - 1.0,
                                (j + 0.5) / resolution * 2.0 - 1.0);

            vector<FloatD> spectrumSamples;

            for (int k = 0; k < SPECTRUM_SAMPLES; k++) {
                FloatD brdfValue;

                if (omega_o.norm() > 1.0) {
                    brdfValue = 0.0f;
                } else {
                    Query q = query;
                    q.omega_o = omega_o;
                    q.lambda = (k + 0.5) / SPECTRUM_SAMPLES * (0.83 - 0.36) + 0.36;
                    brdfValue = queryBrdfDiff(q, gaborBasis);
                }

                if (std::isnan(enoki::detach(brdfValue)))
                    brdfValue = 0.0f;

                spectrumSamples.push_back(brdfValue);
            }

            FloatD r, g, b;
            SpectrumToRGB(spectrumSamples, r, g, b);

            if (enoki::any(r < 0.0f)) r -= r;
            if (enoki::any(g < 0.0f)) g -= g;
            if (enoki::any(b < 0.0f)) b -= b;

            brdfImage_r(i, j) = enoki::detach(r);
            brdfImage_g(i, j) = enoki::detach(g);
            brdfImage_b(i, j) = enoki::detach(b);

            // mse[3 * (i * width + j)] = r - ref.r(i, j);
            // mse[3 * (i * width + j) + 1] = g - ref.g(i, j);
            // mse[3 * (i * width + j) + 2] = b - ref.b(i, j);
            mse_hdr[3 * (i * width + j)] = enoki::log(r + eps) - log(ref.r(i, j) + eps);
            mse_hdr[3 * (i * width + j) + 1] = enoki::log(g + eps) - log(ref.g(i, j) + eps);
            mse_hdr[3 * (i * width + j) + 2] = enoki::log(b + eps) - log(ref.b(i, j) + eps);
            // mse_rgb[3 * (i * width + j)] = enoki::pow(r / (1.0f + r), 1.0f / 2.2f) - std::pow(ref.r(i, j) / (1.0f + ref.r(i, j)), 1.0f / 2.2f);
            // mse_rgb[3 * (i * width + j) + 1] = enoki::pow(g / (1.0f + g), 1.0f / 2.2f) - std::pow(ref.g(i, j) / (1.0f + ref.g(i, j)), 1.0f / 2.2f);
            // mse_rgb[3 * (i * width + j) + 2] = enoki::pow(b / (1.0f + b), 1.0f / 2.2f) - std::pow(ref.b(i, j) / (1.0f + ref.b(i, j)), 1.0f / 2.2f);
        }
    }
    
    BrdfImage brdfImage;
    brdfImage.r = brdfImage_r;
    brdfImage.g = brdfImage_g;
    brdfImage.b = brdfImage_b;

    FloatD loss = enoki::hsum(enoki::sqr(mse_hdr));

    enoki::backward(loss);

    for (int i = 0; i < heightfield.height; i++) {
        for (int j = 0; j < heightfield.width; j++) {
            brdfImage_grad(i, j) = enoki::gradient(heightfield.values[i][j]);
        }
    }

    brdfImage.grad = brdfImage_grad;

    // cout << enoki::graphviz(loss) << endl;

    // cout << enoki::cuda_whos() << endl;

    cout << "\nGenerating BRDF image finished!" << endl;

    return brdfImage;
}

inline Vector2 sampleGauss2d(Float r1, Float r2) {
    // https://en.wikipedia.org/wiki/Box-Muller_transform
    Float tmp = std::sqrt(-2 * std::log(r1));
    Float x = tmp * std::cos(2 * Float(M_PI) * r2);
    Float y = tmp * std::sin(2 * Float(M_PI) * r2);
    return Vector2(x, y);
}
