import argparse
import yaml

parser = argparse.ArgumentParser()


parser.add_argument("--config", type=str, help="Config path.")

parser.add_argument("--reference_path", type=str, help="Reference heightfield path.")

parser.add_argument("--x", type=float, default=0.0, help="Center x of the Gaussian footprint.")
parser.add_argument("--y", type=float, default=0.0, help="Center y of the Gaussian footprint.")
parser.add_argument("--sigma", type=float, default=10.0, help="Size (1 sigma) of the Gaussian footprint.")

parser.add_argument("--method", type=str, default="Wave", choices=["Geom", "Wave"], help="Rendering method.")
parser.add_argument("--sample_num", type=int, default=10000000, help="Number of binning samples. Only valid for geometric optics.")
parser.add_argument("--diff_model", type=str, default="ROHS", choices=["OHS", "GHS", "ROHS", "RGHS", "Kirchhoff"], help="Diffraction model. Expect only subtle difference.")
parser.add_argument("--lambda", type=float, default=0.0, dest="lambda_", metavar="LAMBDA", help="Wavelength in microns. Once set, single wavelength mode is ON.")
parser.add_argument("--light_x", type=float, default=0.0, help="Incoming light's x coordinate (assuming z = 1).")
parser.add_argument("--light_y", type=float, default=0.0, help="Incoming light's y coordinate (assuming z = 1).")

parser.add_argument("--texel_width", type=float, default=1.0, help="Width of a texel in microns on the heightfield.")
parser.add_argument("--vert_scale", type=float, default=1.0, help="Vertical scaling factor of the heightfield.")
parser.add_argument("--resolution", type=int, default=256, help="Output BRDF resolution.")

parser.add_argument("--iterations", type=int, default=10, help="Number of iterations.")


args = parser.parse_args()

if args.config:
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for op in config:
        setattr(args, op, config[op])
