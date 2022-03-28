import numpy as np
from options import args

from WaveOpticsBrdfWithPybind.exrimage import EXRImage
from WaveOpticsBrdfWithPybind.heightfield import Heightfield, GaborBasis
from WaveOpticsBrdfWithPybind.gaborkernel import GaborKernel, GaborKernelPrime
from WaveOpticsBrdfWithPybind.heightfielddiff import HeightfieldDiff, GaborBasisDiff
from WaveOpticsBrdfWithPybind.gaborkerneldiff import GaborKernelDiff, GaborKernelPrimeDiff
from WaveOpticsBrdfWithPybind.brdf import initialize, makeQuery, Query, BrdfImage, WaveBrdfAccel


# Setup
initialize()
query = makeQuery(args.x, args.y, args.sigma, args.lambda_, args.light_x, args.light_y)
refImage = EXRImage(args.reference_path)

# Cut smaller
refImage.width = refImage.height = w = h = args.resolution
refImage.values = refImage.values[0:w, 0:h]


brdfFunction = WaveBrdfAccel(args.diff_model, refImage.width, refImage.height, args.texel_width, args.resolution)

# Generate reference BRDF output
refHeightfield = Heightfield(refImage.values, refImage.width, refImage.height, args.texel_width, args.vert_scale)
refGaborBasis = GaborBasis(refHeightfield)
refResult = brdfFunction.genBrdfImage(query, refGaborBasis)
EXRImage.writeImageRGB(refResult.r, refResult.g, refResult.b, args.resolution, args.resolution, "Results/ReferenceBrdf.exr")



# Generate hypothesis
heightfield = HeightfieldDiff(np.zeros((refImage.width, refImage.height)), refImage.width, refImage.height, args.texel_width, args.vert_scale)
gaborbasis = GaborBasisDiff(heightfield)
result = brdfFunction.genBrdfImageDiff(query, heightfield, refResult)
# print(result.r)
# EXRImage.writeImage(result.r, args.resolution, args.resolution, "Results/testtest.exr")




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
