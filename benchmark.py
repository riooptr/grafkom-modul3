"""Benchmark both implementations on the same original model and camera."""
import time
from objio import load_obj
from ppm_io import read_ppm, write_ppm
from transforms import rot_y, scale
from metrics import psnr, ssim
import raster3d, raytrace
mesh = load_obj('web/mymodel.obj')
tex = read_ppm('web/relief.ppm')
model = rot_y(35) @ scale((1.4, 1.4, 1.4))
args = dict(w=128, h=128, tex_u8=tex, model=model)
t = time.perf_counter(); a, _, fragments = raster3d.render(mesh, **args); tr = time.perf_counter()-t
t = time.perf_counter(); b, _ = raytrace.render(mesh, **args); tt = time.perf_counter()-t
write_ppm('out/mymodel_raster_128.ppm', a); write_ppm('out/mymodel_ray_128.ppm', b)
print(f'raster {tr:.4f}s; ray {tt:.4f}s; ratio {tt/tr:.2f}; fragments {fragments}')
print(f'PSNR {psnr(a,b):.3f} dB; SSIM {ssim(a,b):.6f}')
