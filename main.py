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
brdfFunction = WaveBrdfAccel(args.diff_model, refImage.width, refImage.height, args.texel_width, args.resolution)

# Generate reference/target BRDF
refHeightfield = Heightfield(refImage.values, refImage.width, refImage.height, args.texel_width, args.vert_scale)
refGaborBasis = refHeightfield.toGaborBasis()
refResult = brdfFunction.genBrdfImage(query, refGaborBasis)
EXRImage.writeImageRGB(refResult.r, refResult.g, refResult.b, args.resolution, args.resolution, "Results/RefBrdf.exr")

hf = Heightfield(refGaborBasis, refImage.width, refImage.height, args.texel_width, args.vert_scale)
EXRImage.writeImage(hf.values, hf.width, hf.height, "Results/test.exr")

# Generate hypothesis
heightfield = Heightfield(np.zeros((refImage.width, refImage.height)), refImage.width, refImage.height, args.texel_width, args.vert_scale)
gaborBasis = heightfield.toGaborBasis()

# Optimization
iterations = 1     # args
for i in range(iterations):
    heightfield = Heightfield(gaborBasis, refImage.width, refImage.height, args.texel_width, args.vert_scale)
    EXRImage.writeImage(np.square(np.subtract(heightfield.values, refHeightfield.values)), heightfield.width, heightfield.height, f"Results/HeightfieldResiduum_{i}.exr")
    opVal = np.sum(np.square(np.subtract(heightfield.values, refHeightfield.values)))
    print(opVal)

    result = brdfFunction.genBrdfImage(query, gaborBasis)
    EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, f"Results/Brdf_{i}.exr")

    # manipulate gaborBasis
    # sum of result = sum of result after manipulation
    # assert integral über genBrdfPrime(H(k)) = 0