"""Reproduce perspective interpolation and a fixed-LOD mip experiment."""
import os
import numpy as np
from objio import Mesh, load_obj
from ppm_io import read_ppm, write_ppm
from srgb import srgb_to_linear, linear_to_srgb
from raster3d import render
from transforms import rot_y, scale


def main():
    os.makedirs('out', exist_ok=True)
    tex = read_ppm('data/relief.ppm')
    levels = [srgb_to_linear(tex.astype(float) / 255)]
    for _ in range(3):
        a = levels[-1]
        levels.append(a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2, 3).mean((1, 3)))
    encoded = [(linear_to_srgb(a).clip(0, 1)*255).astype(np.uint8) for a in levels]
    for i, image in enumerate(encoded):
        write_ppm(f'out/mip_{i}.ppm', image)

    # A world-space floor (y=-0.5): edge vertices at view depths 1 and 9.
    # fovy=90, looking down -Z, gives clip.w = -view.z.
    pos = np.array([[-.8, -.5, -1], [7.2, -.5, -9], [-7.2, -.5, -9]])
    uv = np.array([[0, 0], [1, 0], [0, 1]])
    mesh = Mesh(pos, uv, [[0, 1, 2]], [[0, 1, 2]])
    common = dict(w=512, h=512, tex_u8=tex, eye=(0,0,0), center=(0,0,-1),
                  fovy=90, near=.5, far=20, ambient=1)
    images = []
    for label, correct in [('naive', False), ('correct', True)]:
        image, _, _ = render(mesh, perspective_correct=correct, **common)
        images.append(image)
        write_ppm(f'out/perspective_{label}.ppm', image)
    diff = np.abs(images[0].astype(int) - images[1].astype(int)).astype(np.uint8)
    write_ppm('out/perspective_diff.ppm', diff)
    midpoint = (0.5*0/1 + 0.5*1/9) / (0.5/1 + 0.5/9)
    print(f'E4 midpoint: naive={0.5:.3f}, perspective-correct={midpoint:.3f}')

    # Fixed level 3 is chosen deliberately; this is not automatic LOD.
    temple = load_obj('data/pusaka_model.obj')
    model = rot_y(35) @ scale((1.4,)*3)
    gray_levels = []
    for i in (0, 3):
        image, depth, _ = render(temple, w=256, h=256, tex_u8=encoded[i], model=model,
                                 eye=(8,6,10), center=(0,.7,0), far=50)
        write_ppm(f'out/distant_mip_{i}.ppm', image)
        gray = image.astype(float).mean(axis=2)
        # Use only adjacent foreground pixels so the silhouette does not dominate.
        mask = np.isfinite(depth)
        gx = np.diff(gray, axis=1); gy = np.diff(gray, axis=0)
        mx = mask[:, 1:] & mask[:, :-1]; my = mask[1:] & mask[:-1]
        energy = (np.sum(gx[mx]**2) + np.sum(gy[my]**2)) / (mx.sum()+my.sum())
        gray_levels.append(image)
        print(f'LOD {i}: foreground adjacent-pixel energy={energy:.2f}')
    delta = np.abs(gray_levels[0].astype(int) - gray_levels[1].astype(int)).astype(np.uint8)
    write_ppm('out/distant_mip_diff.ppm', delta)


if __name__ == '__main__':
    main()
