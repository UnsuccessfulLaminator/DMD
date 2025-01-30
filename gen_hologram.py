from argparse import ArgumentParser

# Parse a comma-separated pair into a 2-vector
def vector(s):
    vec = list(map(float, s.split(",")))

    if len(vec) < 2: raise ValueError("Not enough components in vector")
    elif len(vec) > 2: raise ValueError("Too many components in vector")

    return vec

parser = ArgumentParser()
parser.add_argument("width", type = int)
parser.add_argument("height", type = int)
parser.add_argument("out")
parser.add_argument("--freq", "-f", type = vector)
parser.add_argument("--freq-polar", "-fp", type = vector)
parser.add_argument("--radius", "-r", type = float, default = 1)

args = parser.parse_args()

if (args.freq is None) == (args.freq_polar is None):
    print("Must specify exactly one of --freq (-f) or --freq-polar (-fp)")
    exit(1)

import hologram
from PIL import Image
import numpy as np



size = (args.height, args.width)

if args.freq: freq = args.freq
else:
    angle = np.deg2rad(args.freq_polar[1])
    mag = args.freq_polar[0]
    freq = [mag*np.cos(angle), mag*np.sin(angle)]

radius_lim = args.radius*min(size)/2

xs = np.arange(size[1])-size[1]//2
ys = np.arange(size[0])-size[0]//2
xx, yy = np.meshgrid(xs, ys)
phase = np.arctan2(yy, xx)
radius = np.hypot(xx, yy)
ampl = 0.5-0.5*np.cos(2*np.pi*radius/radius_lim)
ampl[radius > radius_lim] = 0

holo = hologram.orthogonal_lee(ampl, phase, freq)

Image.fromarray(holo.astype("u1")*255).save(args.out)
