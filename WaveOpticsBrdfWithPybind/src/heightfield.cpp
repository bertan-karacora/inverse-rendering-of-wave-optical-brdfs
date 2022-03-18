#include "heightfield.h"


PYBIND11_MODULE(heightfield, m) {
    py::class_<Heightfield>(m, "Heightfield")
        .def(py::init<>())
        .def(py::init<Eigen::MatrixXf, int, int, Float, Float>())
        .def(py::init<GaborBasis, int, int, Float, Float>())
        .def_readwrite("width", &Heightfield::width)
        .def_readwrite("height", &Heightfield::height)
        .def_readwrite("values", &Heightfield::values)
        .def_readwrite("texelWidth", &Heightfield::texelWidth)
        .def_readwrite("vertScale", &Heightfield::vertScale)
        .def("getValue", &Heightfield::getValue)
        .def("getValueUV", &Heightfield::getValueUV)
        .def("g", &Heightfield::g)
        .def("n", &Heightfield::n);

    py::class_<GaborBasis>(m, "GaborBasis")
        .def(py::init<>())
        .def(py::init<Heightfield>())
        .def_readwrite("gaborKernelPrime", &GaborBasis::gaborKernelPrime)
        .def_readwrite("angularBB", &GaborBasis::angularBB)
        .def_readwrite("topLayer", &GaborBasis::topLayer);
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

Heightfield::Heightfield(Eigen::MatrixXf values, int width, int height, Float texelWidth, Float vertScale) {
    cout << "Generating Heightfield from Numpy..." << endl;

    this->texelWidth = texelWidth;
    this->vertScale = vertScale;
    this->width = width;
    this->height = height;
    this->values = FloatD::copy(values.data(), values.size());

    cout << "Generating Heightfield from Numpy finished!" << endl;
}

Heightfield::Heightfield(GaborBasis gaborBasis, int width, int height, Float texelWidth, Float vertScale) {
    cout << "Generating Heightfield from GaborBasis..." << endl;

    this->texelWidth = texelWidth;
    this->vertScale = vertScale;
    this->width = width;
    this->height = height;
    Eigen::MatrixXf val(this->width, this->height);

    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            val(i, j) = (gaborBasis.gaborKernelPrime[i][j].aInfo / 2.0).dot(Eigen::Vector2f(i, j)) + gaborBasis.gaborKernelPrime[i][j].cInfo;
        }
    }

    this->values = FloatD::copy(val.data(), val.size());

    cout << "Generating Heightfield from GaborBasis finished!" << endl;
}

void Heightfield::computeCoeff(Float *alpha, const Float *x) {
    memset(alpha, 0, sizeof(Float) * 16);
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            alpha[i] += A_inv[i][j] * x[j];
        }
    }
}

// Bicubic interpolation
Float Heightfield::getValue(Float x, Float y) {
    return getValueUV(x / height, y / width);
}

// Bicubic interpolation
Float Heightfield::getValueUV(Float u, Float v) {
    Float x = u * height;
    Float y = v * width;
    int x1 = (int) floor(x);
    int y1 = (int) floor(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    Float a[16];
    Float xp[16] = {hp(x1, y1), hp(x2, y1), hp(x1, y2), hp(x2, y2),
                    hpx(x1, y1), hpx(x2, y1), hpx(x1, y2), hpx(x2, y2),
                    hpy(x1, y1), hpy(x2, y1), hpy(x1, y2), hpy(x2, y2),
                    hpxy(x1, y1), hpxy(x2, y1), hpxy(x1, y2), hpxy(x2, y2)};

    computeCoeff(a, xp);

    Float coeffA[4][4] = {{a[0], a[4], a[8], a[12]},
                          {a[1], a[5], a[9], a[13]},
                          {a[2], a[6], a[10], a[14]},
                          {a[3], a[7], a[11], a[15]}};
    
    Float h = 0.0f;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            h += coeffA[i][j] * pow(x - x1, (Float)(i)) * pow(y - y1, (Float)(j));
        }
    }

    return h;
}

GaborKernel Heightfield::g(int i, int j, Float F, Float lambda) {
    Vector2 m_k(Float((i + 0.5) * texelWidth), Float((j + 0.5) * texelWidth));
    Float l_k = texelWidth;

    Vector2 mu_k = m_k;
    Float sigma_k = l_k / SCALE_FACTOR;

    Float H_mk = getValue(i + 0.5, j + 0.5) * texelWidth * vertScale;   // Assuming texelWidth doesn't affect the heightfield's shape.
    Vector2 HPrime_mk(Float((getValue(i + 1, j) - getValue(i, j)) * vertScale),
                      Float((getValue(i, j + 1) - getValue(i, j)) * vertScale));

    comp C_k = sqrt(F) * l_k * l_k * cnis(4.0 * M_PI / lambda * (H_mk - HPrime_mk.dot(m_k)));
    Vector2 a_k = 2.0 * HPrime_mk / lambda;

    return GaborKernel(mu_k, sigma_k, a_k, C_k);
}

Vector2 Heightfield::n(Float i, Float j) {
    Vector2 HPrime(Float((getValue(i + 0.5f, j) - getValue(i - 0.5f, j)) * vertScale),
                   Float((getValue(i, j + 0.5f) - getValue(i, j - 0.5f)) * vertScale));

    Vector3 n(-HPrime(0), -HPrime(1), Float(1.0));
    n.normalize();

    return n.head(2);
}

GaborBasis::GaborBasis(const Heightfield &hf) {
    Heightfield heightfield = hf;
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
        gaborKernelPrime.push_back(vector<GaborKernelPrime>());
        gaborKernelPrime.back().reserve(heightfield.width);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < heightfield.height; i++) {
        vector<GaborKernelPrime> &gaborKernelPrimeRow = gaborKernelPrime[i];
        vector<AABB> &angularBBRow = angularBBLayer[i];
        for (int j = 0; j < heightfield.width; j++) {
            Vector2 m_k(Float((i + 0.5) * heightfield.texelWidth), Float((j + 0.5) * heightfield.texelWidth));
            Float l_k = heightfield.texelWidth;

            Vector2 mu_k = m_k;
            Float sigma_k = l_k / SCALE_FACTOR;

            Float H_mk = heightfield.getValue(i, j) * heightfield.texelWidth * heightfield.vertScale;   // Assuming texelWidth doesn't affect the heightfield's shape.

            Vector2 HPrime_mk(Float((heightfield.getValue(i + 1, j) - heightfield.getValue(i - 1, j)) / 2.0 * heightfield.vertScale),
                                    Float((heightfield.getValue(i, j + 1) - heightfield.getValue(i, j - 1)) / 2.0 * heightfield.vertScale));

            Float cInfo_k = H_mk - HPrime_mk.dot(m_k);
            Vector2 aInfo_k = 2.0 * HPrime_mk;

            gaborKernelPrimeRow.push_back(GaborKernelPrime(mu_k, sigma_k, aInfo_k, cInfo_k));
            angularBBRow.push_back(AABB(-aInfo_k(0), -aInfo_k(0), -aInfo_k(1), -aInfo_k(1)));
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
