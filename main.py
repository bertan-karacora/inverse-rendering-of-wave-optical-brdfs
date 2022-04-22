import numpy as np
import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from options import args

from WaveOpticsBrdfWithPybind.exrimage import EXRImage
from WaveOpticsBrdfWithPybind.heightfield import Heightfield, GaborBasis
from WaveOpticsBrdfWithPybind.gaborkernel import GaborKernel, GaborKernelPrime
from WaveOpticsBrdfWithPybind.heightfielddiff import HeightfieldDiff, GaborBasisDiff
from WaveOpticsBrdfWithPybind.gaborkerneldiff import GaborKernelDiff, GaborKernelPrimeDiff
from WaveOpticsBrdfWithPybind.brdf import initialize, makeQuery, Query, BrdfImage, WaveBrdfAccel

def plot(i):
    axis = np.linspace(0, i, len(log))
    label = "Loss during optimization"
    labels = ["MSE", "log space", "RGB space", "combined", "ground truth"]
    linestyles = ["-", ":", "--", "-.", (0, (3, 5, 1, 5))]
    fig = plt.figure(figsize=(10, 6))
    plt.title(label)
    for j in range(len(labels)):
        plt.plot(axis, np.asarray(log)[:, j + 1], linestyle=linestyles[j], label=labels[j])
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"Results/{now}/loss.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


###################### Setup ##########################
initialize()
query = makeQuery(args.x, args.y, args.sigma, args.lambda_, args.light_x, args.light_y)
ground_truth = EXRImage(args.reference_path)
log = []
since = datetime.datetime.now()
now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
results_path = f"{os.path.abspath(os.path.dirname(__file__))}/Results/{now}"

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

make_dir(results_path)
make_dir(f"{results_path}/Brdf")
make_dir(f"{results_path}/Gradient")
make_dir(f"{results_path}/Hypothesis")

# Cropping
ground_truth.width = ground_truth.height = args.size
ground_truth.values = ground_truth.values[args.crop_x:args.crop_x+args.size, args.crop_y:args.crop_x+args.size]
EXRImage.writeImage(ground_truth.values, args.size, args.size, f"Results/{now}/Ref_Heightfield.exr")

brdf_function = WaveBrdfAccel(args.diff_model, ground_truth.width, ground_truth.height, args.texel_width, args.resolution)
########################################################


# Generate reference BRDF output
ref_heightfield = Heightfield(ground_truth.values, ground_truth.width, ground_truth.height, args.texel_width, args.vert_scale)
ref_gabor_basis = GaborBasis(ref_heightfield)
ref_result = brdf_function.genBrdfImage(query, ref_gabor_basis)
ref_result.r = np.clip(ref_result.r, 0.0, None)
ref_result.g = np.clip(ref_result.g, 0.0, None)
ref_result.b = np.clip(ref_result.b, 0.0, None)
EXRImage.writeImageRGB(ref_result.r, ref_result.g, ref_result.b, args.resolution, args.resolution, f"Results/{now}/Ref_Brdf.exr")

# Generate hypothesis
heightfield = HeightfieldDiff(np.zeros((ref_heightfield.width, ref_heightfield.height)), ref_heightfield.width, ref_heightfield.height, args.texel_width, args.vert_scale)


################### Optimization ########################
m = np.zeros((heightfield.width, heightfield.height))
v = np.zeros((heightfield.width, heightfield.height))
mse_s = mse_hdr_s = mse_rgb_s = mse_truth_s = 1
for i in range(args.iterations + 1):
    args.lr *= args.decay
    values = np.transpose(heightfield.values[0])
    EXRImage.writeImage(values, heightfield.width, heightfield.height, f"Results/{now}/Hypothesis/Hypo_{i}.exr")
    result = brdf_function.genBrdfImageDiff(query, heightfield, ref_result)
    result.r = np.clip(result.r, 0.0, None)
    result.g = np.clip(result.g, 0.0, None)
    result.b = np.clip(result.b, 0.0, None)

    EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, f"Results/{now}/Brdf/Brdf_{i}.exr")
    EXRImage.writeImage(result.grad, heightfield.width, heightfield.height, f"Results/{now}/Gradient/Grad_{i}.exr")

    mse_truth = np.mean((values - ref_heightfield.values)**2) / mse_truth_s
    mse = np.mean((result.r - ref_result.r)**2 + (result.g - ref_result.g)**2 + (result.b - ref_result.b)**2) / mse_s
    mse_hdr = np.mean((np.log(result.r + 1.0) - np.log(ref_result.r + 1.0))**2 + (np.log(result.g + 1.0) - np.log(ref_result.g + 1.0))**2 + (np.log(result.b + 1.0) - np.log(ref_result.b + 1.0))**2) / mse_hdr_s
    mse_rgb = np.mean(((result.r / (1 + result.r))**(1.0 / 2.2) - (ref_result.r / (1 + ref_result.r))**(1.0 / 2.2))**2 + ((result.g / (1 + result.g))**(1.0 / 2.2) - (ref_result.g / (1 + ref_result.g))**(1.0 / 2.2))**2 + ((result.b / (1 + result.b))**(1.0 / 2.2) - (ref_result.b / (1 + ref_result.b))**(1.0 / 2.2))**2) / mse_rgb_s
    if i == 0:
        mse_truth_s = mse_truth
        mse_s = mse
        mse_hdr_s = mse_hdr
        mse_rgb_s = mse_rgb
        log.append([i, 1.0, 1.0, 1.0, 1.0, 1.0])
    else:
        log.append([i, mse, mse_hdr, mse_rgb, 0.5 * mse_hdr + 0.5 * mse_rgb, mse_truth])
    print("MSE (truth): ", mse_truth)
    print("MSE: ", mse)
    print("MSE HDR: ", mse_hdr)
    print("MSE RGB: ", mse_rgb)
    if not(i % 10): plot(i)

    m = args.beta1 * m + (1.0 - args.beta1) * result.grad
    v = args.beta2 * v + (1.0 - args.beta2) * np.square(result.grad)
    mhat = m / (1.0 - args.beta1**(i+1))
    vhat = v / (1.0 - args.beta2**(i+1))
    values = np.subtract(values, np.divide(args.lr * mhat, np.sqrt(vhat) + args.eps))

    heightfield = HeightfieldDiff(values, heightfield.width, heightfield.height, args.texel_width, args.vert_scale)

    time_elapsed = (datetime.datetime.now() - since).seconds
    print("Iteration: ", i, ", time: ", time_elapsed // 60, "m", time_elapsed % 60, "s")
##########################################################
