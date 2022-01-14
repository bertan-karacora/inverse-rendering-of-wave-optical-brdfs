import numpy as np
from options import args
from WaveOpticsBrdfWithPybind.exrimage import EXRImage
from WaveOpticsBrdfWithPybind.heightfield import Heightfield

refImage = EXRImage(args.reference)
# print(refImage.values)
# print(EXRImage.writeImage(refImage.values, args.resolution, args.resolution, args.save_path))

refHeightfield = Heightfield(refImage.values, refImage.width, refImage.height, args.texel_width, args.vert_scale)
# print(refHeightfield.values)
# print(refHeightfield.getValue(55.5, 4.0))
# print(EXRImage.writeImage(refHeightfield.values, args.resolution, args.resolution, args.save_path))

gaborKernel = toGaborKernel(ref_heightfield)

# print(gaborKernel)
# ref_brdf = genBrdf(
#     gaborKernel,
#     args.texel_width,
#     args.vert_scale,
#     args.x,
#     args.y,
#     args.sigma,
#     args.method,
#     args.sample_num,
#     args.diff_model,
#     args.lambda_,
#     args.light_x,
#     args.light_y,
#     args.resolution
# )
# print(ref_brdf)

# genImage(ref_brdf, args.save_path, args.resolution)

# #args.heightfield_hypothesis
# hypo = np.arange(100).reshape(10, 10)
