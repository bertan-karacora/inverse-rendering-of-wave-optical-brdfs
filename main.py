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
ground_truth = EXRImage(args.reference_path)

# Cropping
ground_truth.width = ground_truth.height = args.size
ground_truth.values = ground_truth.values[args.crop_x:args.crop_x+args.size, args.crop_y:args.crop_x+args.size]
EXRImage.writeImage(ground_truth.values, args.size, args.size, "Results/Reference.exr")

brdfFunction = WaveBrdfAccel(args.diff_model, ground_truth.width, ground_truth.height, args.texel_width, args.resolution)

# Generate reference BRDF output
refHeightfield = Heightfield(ground_truth.values, ground_truth.width, ground_truth.height, args.texel_width, args.vert_scale)
refGaborBasis = GaborBasis(refHeightfield)
refResult = brdfFunction.genBrdfImage(query, refGaborBasis)
EXRImage.writeImageRGB(refResult.r, refResult.g, refResult.b, args.resolution, args.resolution, "Results/ReferenceBrdf.exr")

# Generate hypothesis
heightfield = HeightfieldDiff(np.zeros((refHeightfield.width, refHeightfield.height)), refHeightfield.width, refHeightfield.height, args.texel_width, args.vert_scale)
# gaborbasis = GaborBasisDiff(heightfield)
EXRImage.writeImage(np.transpose(heightfield.values[0]), heightfield.width, heightfield.height, "Results/Heightfield/Hypo_0.exr")
result = brdfFunction.genBrdfImageDiff(query, heightfield, refResult)
EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, "Results/Brdf/Brdf_0.exr")
EXRImage.writeImage(result.grad, heightfield.width, heightfield.height, "Results/Gradients/Grad_0.exr")

# Optimization
m = np.zeros((heightfield.width, heightfield.height))
v = np.zeros((heightfield.width, heightfield.height))
for i in range(args.iterations):
    values = np.transpose(heightfield.values[0])
    m = args.beta1 * m + (1.0 - args.beta1) * result.grad
    v = args.beta2 * v + (1.0 - args.beta2) * np.square(result.grad)
    mhat = m / (1.0 - args.beta1**(i+1))
    vhat = v / (1.0 - args.beta2**(i+1))
    values = np.subtract(values, np.divide(args.lr * mhat, np.sqrt(vhat) + args.eps))

    heightfield = HeightfieldDiff(values, heightfield.width, heightfield.height, args.texel_width, args.vert_scale)
    EXRImage.writeImage(values, heightfield.width, heightfield.height, f"Results/Heightfield/Hypo_{i+1}.exr")

    print("Real diff: ", np.sum(np.square(np.subtract(values, refHeightfield.values))))
    print("Objective: ", np.sum(np.square(np.subtract(result.r, refResult.r)) + np.square(np.subtract(result.g, refResult.g)) + np.square(np.subtract(result.b, refResult.b))))

    result = brdfFunction.genBrdfImageDiff(query, heightfield, refResult)
    EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, f"Results/Brdf/Brdf_{i+1}.exr")
    EXRImage.writeImage(result.grad, heightfield.width, heightfield.height, f"Results/Gradients/Grad_{i+1}.exr")




# Test single input change
# array = np.zeros((refImage.width, refImage.height))
# array[10][10] = 0.9
# heightfield2 = Heightfield(array, refImage.width, refImage.height, args.texel_width, args.vert_scale)
# gaborBasis2 = heightfield2.toGaborBasis()
# result2 = brdfFunction.genBrdfImageFromGaborBasis(query, gaborBasis2)
# EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, "Results/h2.exr")
# print("Differenz: ", np.sum(result.r - result2.r))
# EXRImage.writeImage(np.square(np.subtract(result.r, result2.r)), args.resolution, args.resolution, "Results/testdifference.exr")
