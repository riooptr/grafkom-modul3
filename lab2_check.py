"""Measure the reference, ray tracer, and culling on identical frames."""
import os
import numpy as np
import raster3d, raytrace
from objio import load_obj
from ppm_io import read_ppm, write_ppm
from transforms import rot_y, scale
from metrics import psnr, ssim

os.makedirs('out', exist_ok=True)
mesh = load_obj('data/pusaka_model.obj')
tex = read_ppm('data/relief.ppm')
model = rot_y(35) @ scale((1.4, 1.4, 1.4))
args = dict(w=160, h=160, tex_u8=tex, model=model)
a, _, count = raster3d.render(mesh, **args)
b, _ = raytrace.render(mesh, **args)
ref = read_ppm('ref/temple_ref.ppm')
write_ppm('out/temple_raster_160.ppm', a)
write_ppm('out/temple_rt_160.ppm', b)
write_ppm('out/raster_ray_diff.ppm', np.abs(a.astype(int) - b.astype(int)).astype(np.uint8))
for name, image in [('reference', ref), ('raytrace', b)]:
    d = np.abs(a.astype(int)-image.astype(int))
    print(f'raster vs {name}: PSNR={psnr(a,image):.4f} dB SSIM={ssim(a,image):.6f} pixels different={np.any(d,axis=2).sum()} maxdiff={d.max()}')
c, _, count_cull = raster3d.render(mesh, cull_back=True, **args)
print(f'cull: depth-passing fragments {count} -> {count_cull}, reduction={(count-count_cull)/count*100:.2f}%')
print(f'culling vs two-sided: PSNR={psnr(a,c):.4f} dB SSIM={ssim(a,c):.6f}')
