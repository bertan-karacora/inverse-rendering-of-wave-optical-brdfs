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
EXRImage.writeImageRGB(ref_result.r, ref_result.g, ref_result.b, args.resolution, args.resolution, f"Results/{now}/Ref_Brdf.exr")

# Generate hypothesis
heightfield = HeightfieldDiff(np.zeros((ref_heightfield.width, ref_heightfield.height)), ref_heightfield.width, ref_heightfield.height, args.texel_width, args.vert_scale)


################### Optimization ########################
m = np.zeros((heightfield.width, heightfield.height))
v = np.zeros((heightfield.width, heightfield.height))
for i in range(args.iterations):
    values = np.transpose(heightfield.values[0])
    EXRImage.writeImage(values, heightfield.width, heightfield.height, f"Results/{now}/Hypothesis/Hypo_{i}.exr")
    result = brdf_function.genBrdfImageDiff(query, heightfield, ref_result)
    EXRImage.writeImageRGB(result.r, result.g, result.b, args.resolution, args.resolution, f"Results/{now}/Brdf/Brdf_{i}.exr")
    EXRImage.writeImage(result.grad, heightfield.width, heightfield.height, f"Results/{now}/Gradient/Grad_{i}.exr")

    ref_diff = np.sum(np.square(np.subtract(values, ref_heightfield.values))) / (heightfield.width * heightfield.height)
    loss = np.sum(np.square(np.subtract(result.r, ref_result.r)) + np.square(np.subtract(result.g, ref_result.g)) + np.square(np.subtract(result.b, ref_result.b))) / (args.resolution * args.resolution)
    print("Real diff: ", ref_diff)
    print("Loss: ", loss)
    log.append([i, loss])

    m = args.beta1 * m + (1.0 - args.beta1) * result.grad
    v = args.beta2 * v + (1.0 - args.beta2) * np.square(result.grad)
    mhat = m / (1.0 - args.beta1**(i+1))
    vhat = v / (1.0 - args.beta2**(i+1))
    values = np.subtract(values, np.divide(args.lr * mhat, np.sqrt(vhat) + args.eps))

    heightfield = HeightfieldDiff(values, heightfield.width, heightfield.height, args.texel_width, args.vert_scale)

    time_elapsed = (datetime.datetime.now() - since).seconds
    print("Iteration: ", i, ", time: ", time_elapsed // 60, "m", time_elapsed % 60, "s")
##########################################################
    

def plot():
    axis = np.linspace(0, args.iterations, len(log))
    label = "Loss during iterative optimization"
    labels = ["Mean squared error"]
    fig = plt.figure()
    plt.title(label)
    for i in range(len(labels)):
        plt.plot(axis, np.asarray(log)[:, i + 1], label=labels[i])
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"Results/{now}/loss.pdf", dpi=600)
    plt.close(fig)

plot()
