#ifndef SPECTRUM_H
#define SPECTRUM_H

#include <vector>
#include <cmath>
using namespace std;

const int SPECTRUM_SAMPLES = 8;
const int CIE_samples = 471;
extern const float CIE_wavelengths[CIE_samples];
extern const float CIE_X_entries[CIE_samples];
extern const float CIE_Y_entries[CIE_samples];
extern const float CIE_Z_entries[CIE_samples];
extern const float CIE_D65_entries[CIE_samples];

void SpectrumInit();

void SpectrumToXYZ(const vector<float> &s, float &x, float &y, float &z);

void XYZToRGB(float x, float y, float z, float &r, float &g, float &b);

void SpectrumToRGB(const vector<float> &s, float &r, float &g, float &b);

#endif
