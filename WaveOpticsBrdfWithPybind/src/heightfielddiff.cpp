#include "heightfielddiff.h"


PYBIND11_MODULE(heightfielddiff, m) {
    py::class_<HeightfieldDiff>(m, "HeightfieldDiff")
        .def(py::init<>())
        .def(py::init<Eigen::MatrixXf, int, int, Float, Float>())
        .def(py::init<GaborBasisDiff, int, int, Float, Float>())
        .def_readwrite("width", &HeightfieldDiff::width)
        .def_readwrite("height", &HeightfieldDiff::height)
        .def_readwrite("values", &HeightfieldDiff::values)
        .def_readwrite("texelWidth", &HeightfieldDiff::texelWidth)
        .def_readwrite("vertScale", &HeightfieldDiff::vertScale)
        .def("g", &HeightfieldDiff::g);

    py::class_<GaborBasisDiff>(m, "GaborBasisDiff")
        .def(py::init<>())
        .def(py::init<HeightfieldDiff>())
        .def_readwrite("gaborKernelPrime", &GaborBasisDiff::gaborKernelPrime)
        .def_readwrite("angularBB", &GaborBasisDiff::angularBB)
        .def_readwrite("topLayer", &GaborBasisDiff::topLayer);
}


Float A_inv[16][16] = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                       {-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                       {2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0},
                       {0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0},
                       {-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0},
                       {9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1},
                       {-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1},
                       {2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0},
                       {-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1},
                       {4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1}};


HeightfieldDiff::HeightfieldDiff(Eigen::MatrixXf values, int width, int height, Float texelWidth, Float vertScale) {
    cout << "Generating Heightfield from Numpy..." << endl;

    this->texelWidth = texelWidth;
    this->vertScale = vertScale;
    this->width = width;
    this->height = height;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            this->values[i][j] = FloatD(values(i, j));
        }
    }

    cout << "Generating Heightfield from Numpy finished!" << endl;
}

HeightfieldDiff::HeightfieldDiff(GaborBasisDiff gaborBasis, int width, int height, Float texelWidth, Float vertScale) {
    cout << "Generating Heightfield from GaborBasis..." << endl;

    this->texelWidth = texelWidth;
    this->vertScale = vertScale;
    this->width = width;
    this->height = height;

    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            this->values[i][j] = enoki::dot(gaborBasis.gaborKernelPrime[i][j].aInfo / 2.0, enoki::Array<Float, 2>(i, j)) + gaborBasis.gaborKernelPrime[i][j].cInfo;
        }
    }

    cout << "Generating Heightfield from GaborBasis finished!" << endl;
}

// Bicubic interpolation
FloatD HeightfieldDiff::getValue(Float x, Float y) {
    return getValueUV(x / height, y / width);
}

// Bicubic interpolation
FloatD HeightfieldDiff::getValueUV(Float u, Float v) {
    Float x = u * height;
    Float y = v * width;
    int x1 = (int) floor(x);
    int y1 = (int) floor(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    enoki::Array<FloatD, 16> a(FloatD(0.0));
    enoki::Array<FloatD, 16> xp(hp(x1, y1), hp(x2, y1), hp(x1, y2), hp(x2, y2),
                    hpx(x1, y1), hpx(x2, y1), hpx(x1, y2), hpx(x2, y2),
                    hpy(x1, y1), hpy(x2, y1), hpy(x1, y2), hpy(x2, y2),
                    hpxy(x1, y1), hpxy(x2, y1), hpxy(x1, y2), hpxy(x2, y2));

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            a[i] += A_inv[i][j] * xp[j];
        }
    }
    
    FloatD h = 0.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            h += a[i * 4 + j] * pow(x - x1, (Float)(i)) * pow(y - y1, (Float)(j));
        }
    }

    return h;
}

GaborKernelDiff HeightfieldDiff::g(int i, int j, Float F, Float lambda) {
    Vector2fD m_k(Float((i + 0.5) * texelWidth), Float((j + 0.5) * texelWidth));

    Float sigma_k = texelWidth / SCALE_FACTOR;
    FloatD H_mk = getValue(i + 0.5, j + 0.5) * texelWidth * vertScale;   // Assuming texelWidth doesn't affect the heightfield's shape.
    Vector2fD HPrime_mk((getValue(i + 1, j) - getValue(i, j)) * vertScale,
                      (getValue(i, j + 1) - getValue(i, j)) * vertScale);

    ComplexfD C_k = sqrt(F) * texelWidth * texelWidth * cnis(4.0 * M_PI / lambda * (H_mk - enoki::dot(HPrime_mk, m_k)));
    Vector2fD a_k = 2.0 * HPrime_mk / lambda;

    return GaborKernelDiff(m_k, sigma_k, a_k, C_k);
}

GaborBasisDiff::GaborBasisDiff(HeightfieldDiff heightfield) {
    cout << "Generating GaborBasis from Heightfield..." << endl;

    assert(heightfield.height == heightfield.width);
        
    topLayer = (int) floor(log(heightfield.height * 1.0) / log(2.0) + 1e-4);    // 0, 1, 2, ..., topLayer.

    cout << "Preprocessing heightfield: layer 0" << "\t\r" << flush;
    angularBB.push_back(vector<vector<AABB>>());
    vector<vector<AABB>> &angularBBLayer = angularBB.back();
    angularBBLayer.reserve(heightfield.height);
    for (int i = 0; i < heightfield.height; i++) {
        angularBBLayer.push_back(vector<AABB>());
        angularBBLayer.back().reserve(heightfield.width);
        cout << i << endl;
        gaborKernelPrime.push_back(vector<GaborKernelPrimeDiff>());
        gaborKernelPrime.back().reserve(heightfield.width);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < heightfield.height; i++) {
        vector<GaborKernelPrimeDiff> &gaborKernelPrimeRow = gaborKernelPrime[i];
        vector<AABB> &angularBBRow = angularBBLayer[i];
        for (int j = 0; j < heightfield.width; j++) {
            cout << "\rRow " << i << " | Col " << j << "\t\r" << flush;
            Vector2 m_k(Float((i + 0.5) * heightfield.texelWidth), Float((j + 0.5) * heightfield.texelWidth));
            Float l_k = heightfield.texelWidth;

            Vector2 mu_k = m_k;
            Float sigma_k = l_k / SCALE_FACTOR;
            FloatD H_mk = heightfield.getValue(i, j) * heightfield.texelWidth * heightfield.vertScale;   // Assuming texelWidth doesn't affect the heightfield's shape.

            Vector2fD HPrime_mk(((heightfield.getValue(i + 1, j) - heightfield.getValue(i - 1, j)) / 2.0 * heightfield.vertScale),
                                    ((heightfield.getValue(i, j + 1) - heightfield.getValue(i, j - 1)) / 2.0 * heightfield.vertScale));

            FloatD cInfo_k = H_mk - enoki::dot(HPrime_mk, enoki::Array<Float, 2>(Float((i + 0.5) * heightfield.texelWidth), Float((j + 0.5) * heightfield.texelWidth)));
            Vector2fD aInfo_k = 2.0 * HPrime_mk;

            gaborKernelPrimeRow.push_back(GaborKernelPrimeDiff(mu_k, sigma_k, aInfo_k, cInfo_k));
            // FloatC aInfo_k_cpu = enoki::detach(aInfo_k);
            angularBBRow.push_back(AABB(-aInfo_k[0][0], -aInfo_k[0][0], -aInfo_k[1][0], -aInfo_k[1][0]));
        }
    }

    int currentHeight = heightfield.height;
    int currentWidth = heightfield.width;
    for (int currentLayer = 1; currentLayer <= topLayer; currentLayer++) {
        cout << "\r" << "Preprocessing heightfield: layer " << currentLayer << "\t\r" << flush;
        vector<vector<AABB>> angularBBLayer;
        currentHeight >>= 1;
        currentWidth >>= 1;
        angularBBLayer.reserve(currentHeight);
        for (int i = 0; i < currentHeight; i++) {
            vector<AABB> angularBBRow;
            angularBBRow.reserve(currentWidth);
            for (int j = 0; j < currentWidth; j++) {
                AABB aabb1 = angularBB[currentLayer - 1][i * 2 + 0][j * 2 + 0];
                AABB aabb2 = angularBB[currentLayer - 1][i * 2 + 1][j * 2 + 0];
                AABB aabb3 = angularBB[currentLayer - 1][i * 2 + 1][j * 2 + 1];
                AABB aabb4 = angularBB[currentLayer - 1][i * 2 + 0][j * 2 + 1];
                angularBBRow.push_back(combineAABB(aabb1, aabb2, aabb3, aabb4));
            }
            angularBBLayer.push_back(angularBBRow);
        }
        angularBB.push_back(angularBBLayer);
    }

    cout << "\n" << "Preprocessing heightfield finished!" << endl;
}
