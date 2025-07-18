#ifndef HELPERS_H
#define HELPERS_H

#define M_PIf Float(M_PI)
#define M_2PIf 6.28318530718f
#define M_2PIf_inv 0.15915494309f

#define SCALE_FACTOR 2.0
// #define SCALE_FACTOR 2.35482004503

#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <vector>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <enoki/python.h>
#include <enoki/autodiff.h>
#include <enoki/cuda.h>
#include <enoki/complex.h>

typedef float Float;

// using FloatC = enoki::CUDAArray<Float>;
using FloatD = enoki::DiffArray<Float>;
using Vector2fD = enoki::Array<FloatD, 2>;
using Vector3fD = enoki::Array<FloatD, 3>;
using ComplexfD = enoki::Complex<FloatD>;
using FloatXXD = enoki::Array<enoki::Array<FloatD, 16>, 16>;
// using FloatXXD = enoki::Array<enoki::Array<FloatD, 32>, 32>;

using namespace std;
namespace py = pybind11;

typedef Eigen::Vector2f Vector2;
typedef Eigen::Vector3f Vector3;
typedef std::complex<Float> comp;

const Float eps = 1e-6f;

inline Float dist(Float x1, Float y1, Float x2, Float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

inline Float G(Float x, Float mu, Float sigma) {
    return 1.0 / (sqrt(2.0 * M_PI) * sigma) * exp(-0.5 * pow((x - mu) / sigma, 2.0));
}

inline FloatD G(Float x, FloatD mu, Float sigma) {
    return 1.0f / (sqrt(2.0f * M_PI) * sigma) * enoki::exp(-0.5f * enoki::pow((x - mu) / sigma, 2.0f));
}

inline Float G(Vector2 x, Vector2 mu, Float sigma) {
    return G(x(0), mu(0), sigma) * G(x(1), mu(1), sigma);
}

inline FloatD G(Vector2 x, Vector2fD mu, Float sigma) {
    return G(x(0), mu[0], sigma) * G(x(1), mu[1], sigma);
}

inline Float GRot(Float x, Float mu, Float sigma, Float period) {
    return G(x - period, mu, sigma) + G(x, mu, sigma) + G(x + period, mu, sigma);
}

inline Float GRot(Vector2 x, Vector2 mu, Float sigma, Float period) {
    return GRot(x(0), mu(0), sigma, period) * GRot(x(1), mu(1), sigma, period);
}

// e^(ix) = cos x + i sin x
inline comp cis(Float x) {
    Float sinx, cosx;
    // sincosf(x, &sinx, &cosx);
    // return comp(cosx, sinx);
   return comp(std::cos(x), std::sin(x));
}

// e^(-ix) = cos x - i sin x
inline comp cnis(Float x) {
    Float sinx, cosx;
    // sincosf(x, &sinx, &cosx);
    // return comp(cosx, -sinx);
   return comp(std::cos(x), -std::sin(x));
}

inline ComplexfD cnis(FloatD x) {
   return ComplexfD(enoki::cos(x), -enoki::sin(x));
}

template <class T>
inline T randUniform() {
    return rand() / (T) RAND_MAX;
}

template <class T>
inline T normrnd(T mu, T sigma) {
    T U = randUniform<T>();
    T V = randUniform<T>();
    T X = sqrt(-2.0 * log(U)) * cos(2.0 * M_PI * V);
    return X * sigma + mu;
}

inline Vector2 normrnd2D(Vector2 mu, Float sigma) {
    return Vector2(normrnd<Float>(mu(0), sigma), normrnd<Float>(mu(1), sigma));
}


struct AABB {
    Float xMin, xMax, yMin, yMax;
    AABB(Float xMin_ = Float(0.0f), Float xMax_ = Float(0.0f), Float yMin_ = Float(0.0f), Float yMax_ = Float(0.0f)) :
        xMin(xMin_), xMax(xMax_), yMin(yMin_), yMax(yMax_) {}
};

inline bool intersectAABB(const AABB &aabb1, const AABB &aabb2) {
    if (aabb1.xMax < aabb2.xMin)
        return false;
    if (aabb1.xMin > aabb2.xMax)
        return false;
    if (aabb1.yMax < aabb2.yMin)
        return false;
    if (aabb1.yMin > aabb2.yMax)
        return false;
    return true;
}

inline bool intersectAABBRot(const AABB &aabb1, const AABB &aabb2, Float period) {
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            AABB aabb0(aabb1.xMin + i * period, aabb1.xMax + i * period,
                       aabb1.yMin + j * period, aabb1.yMax + j * period);
            if (intersectAABB(aabb0, aabb2))
                return true;
        }
    }
    return false;
}

inline bool insideAABB(const AABB &aabb, Float x, Float y) {
    if (x < aabb.xMin || x > aabb.xMax)
        return false;
    if (y < aabb.yMin || y > aabb.yMax)
        return false;
    return true;
}

inline bool insideAABB(const AABB &aabb, Vector2 xy) {
    return insideAABB(aabb, xy(0), xy(1));
}

inline AABB combineAABB(const AABB &aabb1, const AABB &aabb2) {
    AABB aabb;
    aabb.xMin = std::min(aabb1.xMin, aabb2.xMin);
    aabb.xMax = std::max(aabb1.xMax, aabb2.xMax);
    aabb.yMin = std::min(aabb1.yMin, aabb2.yMin);
    aabb.yMax = std::max(aabb1.yMax, aabb2.yMax);
    return aabb;
}

inline AABB combineAABB(const AABB &aabb1, const AABB &aabb2, const AABB &aabb3, const AABB &aabb4) {
    AABB aabbA = combineAABB(aabb1, aabb2);
    AABB aabbB = combineAABB(aabb3, aabb4);
    return combineAABB(aabbA, aabbB);
}

inline Float distSqrPeriod(Vector2 p1, Vector2 p2, Float period) {
    Float minDistX = std::abs(p1(0) - p2(0));
    minDistX = std::min(minDistX, std::abs(p1(0) + period - p2(0)));
    minDistX = std::min(minDistX, std::abs(p1(0) - period - p2(0)));
    Float minDistY = std::abs(p1(1) - p2(1));
    minDistY = std::min(minDistY, std::abs(p1(1) + period - p2(1)));
    minDistY = std::min(minDistY, std::abs(p1(1) - period - p2(1)));
    return minDistX * minDistX + minDistY * minDistY;
}

#endif
