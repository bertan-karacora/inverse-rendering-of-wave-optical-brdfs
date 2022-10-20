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
    # labels = ["MSE", "CMSLE (ɛ = 0.001)", "CMSLE (ɛ = 1.0)", "RGB space", "gamma corrected", "ground truth"]
    # linestyles = [":", (0, (3, 10, 1, 10, 1, 10)), "--", "-.", (0, (3, 5, 1, 5)), "-"]
    labels = ["CMSLE (ɛ = 0.001)", "CMSLE (ɛ = 1.0)", "ground truth"]
    linestyles = [":", "--", "-"]
    
    fig = plt.figure(figsize=(7, 4))
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
make_dir(f"{results_path}/Ref")

# Ground truth
ground_truth = EXRImage(args.reference_path)

# Cropping
ground_truth.width = ground_truth.height = args.size
ground_truth.values = ground_truth.values[args.crop_x:args.crop_x+args.size, args.crop_y:args.crop_x+args.size]
ground_truth.values = ground_truth.values - np.mean(ground_truth.values)
EXRImage.writeImage(ground_truth.values, args.size, args.size, f"Results/{now}/Ref_Heightfield.exr")

ref_heightfield = Heightfield(ground_truth.values, ground_truth.width, ground_truth.height, args.texel_width, args.vert_scale)
ref_gabor_basis = GaborBasis(ref_heightfield)

brdf_function = WaveBrdfAccel(args.diff_model, ground_truth.width, ground_truth.height, args.texel_width, args.resolution)
########################################################

x = y = args.size / 2.0
light_x = light_y = 0.0
query = makeQuery(x, y, args.sigma, args.lambda_, light_x, light_y)
ref_result = brdf_function.genBrdfImage(query, ref_gabor_basis)
ref_result.r = np.clip(ref_result.r, 0.0, None)
ref_result.g = np.clip(ref_result.g, 0.0, None)
ref_result.b = np.clip(ref_result.b, 0.0, None)
EXRImage.writeImageRGB(ref_result.r, ref_result.g, ref_result.b, args.resolution, args.resolution, f"Results/{now}/Ref/Ref_{x}_{y}.exr")


# Generate hypothesis
hypo_values = np.zeros((ref_heightfield.width, ref_heightfield.height))
heightfield = HeightfieldDiff(ground_truth.values + (np.random.rand(ref_heightfield.width, ref_heightfield.height) - 0.5) * 0.076, ref_heightfield.width, ref_heightfield.height, args.texel_width, args.vert_scale)

################### Optimization ########################
m = np.zeros((heightfield.width, heightfield.height))
v = np.zeros((heightfield.width, heightfield.height))
mse_s = mse_hdr_s = mse_hdr_s_2 = mse_rgb_s = mse_gamma_s = mse_truth_s = 1
for i in range(args.iterations + 1):
    # Generate reference BRDF output
    x = round(np.random.uniform(0.0, args.size), 2)
    y = round(np.random.uniform(0.0, args.size), 2)
    # light_x = round(np.random.uniform(-1.0, 1.0), 2)
    # light_y = round(np.random.uniform(-1.0, 1.0), 2)
    query = makeQuery(x, y, args.sigma, args.lambda_, light_x, light_y)
    ref_result = brdf_function.genBrdfImage(query, ref_gabor_basis)
    ref_result.r = np.clip(ref_result.r, 0.0, None)
    ref_result.g = np.clip(ref_result.g, 0.0, None)
    ref_result.b = np.clip(ref_result.b, 0.0, None)
    EXRImage.writeImageRGB(ref_result.r, ref_result.g, ref_result.b, args.resolution, args.resolution, f"Results/{now}/Ref/Ref_{i}_{x}_{y}.exr")


    args.lr *= (1.0 - args.decay)
    print("lr : ", args.lr)
    values = np.transpose(heightfield.values[0])
    EXRImage.writeImage(values, heightfield.width, heightfield.height, f"Results/{now}/Hypothesis/Hypo_{i}.exr")
    result = brdf_function.genBrdfImageDiff(query, heightfield, ref_result)
    result.r = np.clip(result.r, 0.0, None)
    result.g = np.clip(result.g, 0.0, None)
    result.b = np.clip(result.b, 0.0, None)

    EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, f"Results/{now}/Brdf/Brdf_{i}.exr")
    EXRImage.writeImage(result.grad, heightfield.width, heightfield.height, f"Results/{now}/Gradient/Grad_{i}.exr")

    eps = 0.001
    eps_2 = 1.0
    mse_truth = np.mean((values - ref_heightfield.values)**2) / mse_truth_s
    mse = np.mean((result.r - ref_result.r)**2 + (result.g - ref_result.g)**2 + (result.b - ref_result.b)**2) / mse_s
    mse_hdr = np.mean((np.log(result.r + eps) - np.log(ref_result.r + eps))**2 + (np.log(result.g + eps) - np.log(ref_result.g + eps))**2 + (np.log(result.b + eps) - np.log(ref_result.b + eps))**2) / mse_hdr_s
    mse_hdr_2 = np.mean((np.log(result.r + eps_2) - np.log(ref_result.r + eps_2))**2 + (np.log(result.g + eps_2) - np.log(ref_result.g + eps_2))**2 + (np.log(result.b + eps_2) - np.log(ref_result.b + eps_2))**2) / mse_hdr_s_2
    mse_rgb = np.mean((result.r / (1 + result.r) - ref_result.r / (1 + ref_result.r))**2 + (result.g / (1 + result.g) - ref_result.g / (1 + ref_result.g))**2 + (result.b / (1 + result.b) - ref_result.b / (1 + ref_result.b))**2) / mse_rgb_s
    mse_gamma = np.mean(((result.r / (1 + result.r))**(1.0 / 2.2) - (ref_result.r / (1 + ref_result.r))**(1.0 / 2.2))**2 + ((result.g / (1 + result.g))**(1.0 / 2.2) - (ref_result.g / (1 + ref_result.g))**(1.0 / 2.2))**2 + ((result.b / (1 + result.b))**(1.0 / 2.2) - (ref_result.b / (1 + ref_result.b))**(1.0 / 2.2))**2) / mse_gamma_s
    if i == 0:
        mse_truth_s = mse_truth
        mse_s = mse
        mse_hdr_s = mse_hdr
        mse_hdr_s_2 = mse_hdr_2
        mse_rgb_s = mse_rgb
        mse_gamma_s = mse_gamma
        log.append([i, 1.0, 1.0, 1.0])
    else:
        log.append([i, mse_hdr, mse_hdr_2, mse_truth])
    print("MSE (truth): ", mse_truth)
    print("MSE: ", mse)
    print("MSE HDR: ", mse_hdr)
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
