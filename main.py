import numpy as np
from options import args
from WaveOpticsBrdfWithPybind.exrimage import EXRImage
from WaveOpticsBrdfWithPybind.heightfield import Heightfield, GaborBasis
from WaveOpticsBrdfWithPybind.gaborkernel import GaborKernel, GaborKernelPrime
from WaveOpticsBrdfWithPybind.brdf import init, makeQuery, Query, BrdfImage, BrdfBase, GeometricBrdf, WaveBrdfAccel

init()

refImage = EXRImage(args.reference)
refHeightfield = Heightfield(refImage.values, refImage.width, refImage.height, args.texel_width, args.vert_scale)

gaborBasis = refHeightfield.toGaborBasis()
hf = Heightfield(gaborBasis, refImage.width, refImage.height, args.texel_width, args.vert_scale)
EXRImage.writeImage(np.absolute(hf.values - refHeightfield.values), args.resolution, args.resolution, "Results/GabortransformResiduum.exr")

query = makeQuery(args.x, args.y, args.sigma, args.lambda_, args.light_x, args.light_y)

brdfFunction = WaveBrdfAccel(args.diff_model, gaborBasis, refHeightfield.width, refHeightfield.height, refHeightfield.texelWidth)
result = brdfFunction.genBrdfImage(query, args.resolution)
EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, "Results/Brdf.exr")
