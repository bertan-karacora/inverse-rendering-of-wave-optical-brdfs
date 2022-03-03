import numpy as np
from options import args
from WaveOpticsBrdfWithPybind.exrimage import EXRImage
from WaveOpticsBrdfWithPybind.heightfield import Heightfield, GaborBasis
from WaveOpticsBrdfWithPybind.gaborkernel import GaborKernel, GaborKernelPrime
from WaveOpticsBrdfWithPybind.brdf import initialize, makeQuery, Query, BrdfImage, GeometricBrdf, WaveBrdfAccel

# Setup
initialize()
query = makeQuery(args.x, args.y, args.sigma, args.lambda_, args.light_x, args.light_y)
refImage = EXRImage(args.reference_path)
brdfFunction = WaveBrdfAccel(args.diff_model, refImage.width, refImage.height, args.texel_width, args.resolution)

# Generate reference BRDF image
refHeightfield = Heightfield(refImage.values, refImage.width, refImage.height, args.texel_width, args.vert_scale)
refGaborBasis = refHeightfield.toGaborBasis()
refResult = brdfFunction.genBrdfImageFromGaborBasis(query, refGaborBasis)
# EXRImage.writeImageRGB(refResult.r, refResult.g, refResult.b, args.resolution, args.resolution, "Results/RefBrdf.exr")


# Test if heightfield read correctly
hf = Heightfield(refGaborBasis, refImage.width, refImage.height, args.texel_width, args.vert_scale)
print("Heightfield difference", np.sum(np.square(np.subtract(hf.values, refHeightfield.values))))
# EXRImage.writeImage(hf.values, hf.width, hf.height, "Results/test.exr")


# Generate hypothesis
heightfield = Heightfield(np.zeros((refImage.width, refImage.height)), refImage.width, refImage.height, args.texel_width, args.vert_scale)
gaborBasis = heightfield.toGaborBasis()
result = brdfFunction.genBrdfImageFromGaborBasis(query, gaborBasis)
# EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, "Results/starting_hypothesis.exr")

# Test single input change
# array = np.zeros((refImage.width, refImage.height))
# array[10][10] = 10
# heightfield2 = Heightfield(array, refImage.width, refImage.height, args.texel_width, args.vert_scale)
# gaborBasis2 = heightfield2.toGaborBasis()
# result2 = brdfFunction.genBrdfImageFromGaborBasis(query, gaborBasis2)
# EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, "Results/h2.exr")
# print("Differenz: ", np.sum(result.r - result2.r))
# EXRImage.writeImage(np.square(np.subtract(result.r, result2.r)), args.resolution, args.resolution, "Results/testdifference.exr")

# Optimization
# for i in range(args.iterations):
#     heightfield = Heightfield(gaborBasis, refImage.width, refImage.height, args.texel_width, args.vert_scale)
#     EXRImage.writeImage(np.square(np.subtract(heightfield.values, refHeightfield.values)), heightfield.width, heightfield.height, f"Results/HeightfieldResiduum_{i}.exr")
#     opVal = np.sum(np.square(np.subtract(heightfield.values, refHeightfield.values)))
#     print(opVal)

#     result = brdfFunction.genBrdfImage(query, gaborBasis)
#     EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, f"Results/Brdf_{i}.exr")

# Test differentiation
# manipulate gaborBasis
# sum of result = sum of result after manipulation
# assert integral über genBrdfPrime(H(k)) = 0
