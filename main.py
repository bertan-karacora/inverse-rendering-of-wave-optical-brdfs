import numpy as np
from options import args
from WaveOpticsBrdfWithPybind.exrimage import EXRImage
from WaveOpticsBrdfWithPybind.heightfield import Heightfield, GaborBasis
from WaveOpticsBrdfWithPybind.gaborkernel import GaborKernel, GaborKernelPrime
from WaveOpticsBrdfWithPybind.brdf import init, makeQuery, Query, BrdfImage, GeometricBrdf, WaveBrdfAccel

# Setup
init()
query = makeQuery(args.x, args.y, args.sigma, args.lambda_, args.light_x, args.light_y)
refImage = EXRImage(args.reference)
brdfFunction = WaveBrdfAccel(args.diff_model, refImage.width, args.resolution, refImage.height, args.texel_width)

# Generate reference/target BRDF
refHeightfield = Heightfield(refImage.values, refImage.width, refImage.height, args.texel_width, args.vert_scale)
refGaborBasis = refHeightfield.toGaborBasis()
refResult = brdfFunction.genBrdfImage(query, refGaborBasis)
EXRImage.writeImageRGB(refResult.r, refResult.g, refResult.b, args.resolution, args.resolution, "Results/RefBrdf.exr")

# Test transformation

# Generate hypothesis
gaborBasis = refGaborBasis

# Optimization
iterations = 1     # args
for i in range(iterations):
    heightfield = Heightfield(gaborBasis, refImage.width, refImage.height, args.texel_width, args.vert_scale)
    EXRImage.writeImage(np.absolute(heightfield.values - refHeightfield.values), args.resolution, args.resolution, f"Results/HeightfieldResiduum_{i}.exr")

    result = brdfFunction.genBrdfImage(query, gaborBasis)
    EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, f"Results/Brdf_{i}.exr")

    # manipulate gaborBasis
