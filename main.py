import numpy as np
from options import args
from WaveOpticsBrdfWithPybind.exrimage import EXRImage
from WaveOpticsBrdfWithPybind.heightfield import Heightfield, GaborBasis
from WaveOpticsBrdfWithPybind.gaborkernel import GaborKernel, GaborKernelPrime
from WaveOpticsBrdfWithPybind.brdf import init, makeQuery, Query, BrdfBase, GeometricBrdf, WaveBrdfAccel

init()

refImage = EXRImage(args.reference)
# print(refImage.values)
# print(EXRImage.writeImage(refImage.values, args.resolution, args.resolution, args.save_path))

refHeightfield = Heightfield(refImage.values, refImage.width, refImage.height, args.texel_width, args.vert_scale)
# print(refHeightfield.values)
# print(refHeightfield.getValue(55.5, 4.0))
# print(EXRImage.writeImage(refHeightfield.values, args.resolution, args.resolution, args.save_path))

gaborBasis = refHeightfield.toGaborBasis()
print(gaborBasis.gaborKernelPrime[500][450].cInfo)

query = makeQuery(args.x, args.y, args.sigma, args.lambda_, args.light_x, args.light_y)

brdf = WaveBrdfAccel(args.method, gaborBasis, refHeightfield.width, refHeightfield.height, refHeightfield.texelWidth)
result = brdf.genBrdfImage(query, args.resolution)
print(result[200][200])
EXRImage.writeImage(result, args.resolution, args.resolution, args.save_path)
