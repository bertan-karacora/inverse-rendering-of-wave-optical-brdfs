import numpy as np
from options import args
from WaveOpticsBrdfWithPybind.exrimage import EXRImage
from WaveOpticsBrdfWithPybind.heightfield import Heightfield, GaborBasis
from WaveOpticsBrdfWithPybind.gaborkernel import GaborKernel, GaborKernelPrime
from WaveOpticsBrdfWithPybind.brdf import init, makeQuery, Query, BrdfImage, BrdfBase, GeometricBrdf, WaveBrdfAccel

init()

refImage = EXRImage(args.reference)
# EXRImage.writeImage(refImage.values, args.resolution, args.resolution, args.save_path)

refHeightfield = Heightfield(refImage.values, refImage.width, refImage.height, args.texel_width, args.vert_scale)
# EXRImage.writeImage(refHeightfield.values, args.resolution, args.resolution, args.save_path)

print(refHeightfield.getValue(40, 50))

gaborBasis = refHeightfield.toGaborBasis()
hf = Heightfield(gaborBasis, refImage.width, refImage.height, args.texel_width, args.vert_scale)
EXRImage.writeImage(hf.values, args.resolution, args.resolution, args.save_path)

print(hf.getValue(40, 50))

# query = makeQuery(args.x, args.y, args.sigma, args.lambda_, args.light_x, args.light_y)

# brdf = WaveBrdfAccel(args.diff_model, gaborBasis, refHeightfield.width, refHeightfield.height, refHeightfield.texelWidth)
# result = brdf.genBrdfImage(query, args.resolution)

# EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, args.save_path)
