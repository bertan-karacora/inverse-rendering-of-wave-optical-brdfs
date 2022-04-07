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
x_off = 0
y_off = 0
refImage.values = refImage.values[x_off:x_off+w, y_off:y_off+h]
EXRImage.writeImage(refImage.values, args.resolution, args.resolution, "Results/Reference.exr")


brdfFunction = WaveBrdfAccel(args.diff_model, refImage.width, refImage.height, args.texel_width, args.resolution)

# Generate reference BRDF output
refHeightfield = Heightfield(refImage.values, refImage.width, refImage.height, args.texel_width, args.vert_scale)
refGaborBasis = GaborBasis(refHeightfield)
refResult = brdfFunction.genBrdfImage(query, refGaborBasis)
EXRImage.writeImageRGB(refResult.r, refResult.g, refResult.b, args.resolution, args.resolution, "Results/ReferenceBrdf.exr")



# Generate hypothesis
heightfield = HeightfieldDiff(np.zeros((refImage.width, refImage.height)), refImage.width, refImage.height, args.texel_width, args.vert_scale)
# heightfield = HeightfieldDiff(refImage.values, refImage.width, refImage.height, args.texel_width, args.vert_scale)
# gaborbasis = GaborBasisDiff(heightfield)
EXRImage.writeImage(np.transpose(heightfield.values[0]), args.resolution, args.resolution, "Results/hypo_0.exr")
result = brdfFunction.genBrdfImageDiff(query, heightfield, refResult)
EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, "Results/Brdf_0.exr")
EXRImage.writeImageRGB(np.subtract(result.r, refResult.r), np.subtract(result.g, refResult.g), np.subtract(result.b, refResult.b), args.resolution, args.resolution, "Results/diff_0.exr")
EXRImage.writeImage(result.grad, args.resolution, args.resolution, "Results/grad_0.exr")

# Learning rate for stochastic gradient descent
lr = 0.005

# Optimization
for i in range(args.iterations):
    heightfield = HeightfieldDiff(np.transpose(heightfield.values[0]) - result.grad * lr, refImage.width, refImage.height, args.texel_width, args.vert_scale)
    EXRImage.writeImage(np.transpose(heightfield.values[0]), args.resolution, args.resolution, f"Results/hypo_{i}.exr")
    # EXRImage.writeImage(np.square(np.subtract(heightfield.values, refHeightfield.values)), heightfield.width, heightfield.height, f"Results/HeightfieldResiduum_{i}.exr")
    opVal = np.sum(np.square(np.subtract(np.transpose(heightfield.values[0]), refHeightfield.values)))
    print("Real diff: ", opVal)
    # np.subtract(result.r, refResult.r), np.subtract(result.g, refResult.g), np.subtract(result.b, refResult.b)
    opVal2 = np.sum(np.square(np.subtract(result.r, refResult.r)))
    print("Objective: ", opVal2)

    result = brdfFunction.genBrdfImageDiff(query, heightfield, refResult)
    EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, f"Results/Brdf_{i}.exr")
    EXRImage.writeImage(result.grad, args.resolution, args.resolution, f"Results/grad_{i}.exr")




# Test single input change
# array = np.zeros((refImage.width, refImage.height))
# array[10][10] = 10
# heightfield2 = Heightfield(array, refImage.width, refImage.height, args.texel_width, args.vert_scale)
# gaborBasis2 = heightfield2.toGaborBasis()
# result2 = brdfFunction.genBrdfImageFromGaborBasis(query, gaborBasis2)
# EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, "Results/h2.exr")
# print("Differenz: ", np.sum(result.r - result2.r))
# EXRImage.writeImage(np.square(np.subtract(result.r, result2.r)), args.resolution, args.resolution, "Results/testdifference.exr")


# Test differentiation
# manipulate gaborBasis
# sum of result = sum of result after manipulation
# assert integral über genBrdfPrime(H(k)) = 0
