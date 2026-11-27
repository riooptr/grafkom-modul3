"""Produce perspective-correction and gamma-correct mip experiments."""
import os
import numpy as np
from ppm_io import read_ppm, write_ppm
from srgb import srgb_to_linear, linear_to_srgb
from raster3d import sample_bilinear, render
from transforms import rot_y, scale
from objio import load_obj

os.makedirs('out', exist_ok=True)
tex = read_ppm('web/relief.ppm')
linear = srgb_to_linear(tex.astype(float) / 255)
levels = [linear]
for i in range(3):
    a = levels[-1]
    levels.append(a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2, 3).mean((1, 3)))
for i, a in enumerate(levels):
    write_ppm(f'out/mip_{i}.ppm', (linear_to_srgb(a).clip(0,1)*255).astype(np.uint8))

# Screen-aligned triangle: the bottom edge has w=1 and w=9.
y, x = np.mgrid[0:256, 0:512]
t = x / 511
mask = y >= (128 * (1-t))
invw = (1-t) + t/9
u_pc = (t/9) / invw
for label, u in [('naive', t), ('correct', u_pc)]:
    rgb = sample_bilinear(linear, u, y/256)
    rgb[~mask] = 0
    write_ppm(f'out/perspective_{label}.ppm', (linear_to_srgb(rgb).clip(0,1)*255).astype(np.uint8))
print('u at edge midpoint: naive=0.5, perspective-correct=0.1')
# Fixed LOD comparison on the same distant view; not automatic LOD selection.
mesh = load_obj('web/mymodel.obj'); model=rot_y(35) @ scale((1.4,)*3)
for i in [0,3]:
    a=(linear_to_srgb(levels[i]).clip(0,1)*255).astype(np.uint8)
    image, _, _=render(mesh,w=256,h=256,tex_u8=a,model=model,eye=(8,6,10),center=(0,.7,0),far=50)
    write_ppm(f'out/distant_mip_{i}.ppm',image)
    gray=image.astype(float).mean(2)
    energy=np.mean(np.diff(gray,axis=1)**2) + np.mean(np.diff(gray,axis=0)**2)
    print(f'LOD {i}: adjacent-pixel difference energy={energy:.2f}')
