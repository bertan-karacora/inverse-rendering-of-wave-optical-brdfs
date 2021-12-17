import numpy as np
from options import args
from WaveOpticsBrdfWithPybind.genBrdf import genBrdf
from WaveOpticsBrdfWithPybind.visualize import readImage, genImage

ref_heightfield = readImage(args.reference, args.texel_width, args.vert_scale)
print(ref_heightfield)
#genImage(ref_heightfield, args.save_path, args.resolution)

ref_brdf = genBrdf(
    ref_heightfield,
    args.texel_width,
    args.vert_scale,
    args.x,
    args.y,
    args.sigma,
    args.method,
    args.sample_num,
    args.diff_model,
    args.lambda_,
    args.light_x,
    args.light_y,
    args.resolution
)
print(ref_brdf)

genImage(ref_brdf, args.save_path, args.resolution)

#args.heightfield_hypothesis
