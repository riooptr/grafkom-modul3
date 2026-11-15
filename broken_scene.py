"""Module 3 -- DIAGNOSE-AND-FIX: three broken renders, each fixed by one change.

  python broken_scene.py --bug inside-out   # winding / backface convention
  python broken_scene.py --bug zfight       # near plane 1e-3 -> depth crush
  python broken_scene.py --bug stretched    # matrix multiply order swapped
  python broken_scene.py --all              # evidence table for all three

Your Lab 2 task: from the EVIDENCE (the render, the depth image, the printed
matrices/values) -- write the diagnosis BEFORE reading the fix -- identify the
cause, apply the one change, and prove it (SSIM to the reference; a depth
histogram for z-fighting).
"""
from __future__ import annotations

import argparse
import os
import json

import numpy as np

from transforms import perspective, look_at, rot_y, scale, translate
from objio import load_obj, Mesh
from ppm_io import read_ppm, write_ppm
from metrics import psnr, ssim
import raster3d


def _load():
    mesh = load_obj("data/pusaka_model.obj")
    tex = read_ppm("data/relief.ppm")
    return mesh, tex


GOOD_MODEL = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))


def bug_inside_out(mesh, tex, w=192, h=192):
    # BUG: every triangle's winding flipped, rendered WITH back-face culling on
    # -> the outward faces are now "back-facing" and get culled; you see through
    # the temple to its far interior. FIX: re-wind the faces (or, if you cannot
    # touch the mesh, disable culling).
    mesh = outward_fallback(mesh)
    flipped = Mesh(mesh.positions, mesh.uvs,
                   mesh.faces[:, ::-1].copy(),
                   None if mesh.face_uvs is None else mesh.face_uvs[:, ::-1].copy())
    bad, db, fbad = raster3d.render(flipped, w=w, h=h, tex_u8=tex, model=GOOD_MODEL,
                                    cull_back=True)
    good, dg, fgood = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=GOOD_MODEL,
                                      cull_back=True)
    fin = np.isfinite(db) & np.isfinite(dg)
    dz = float(np.mean(np.abs(db[fin] - dg[fin])))
    return bad, good, (f"flipped winding + back-face cull: the outward faces are now "
                       f"'back-facing' and culled, so the render shows the temple's "
                       f"interior surfaces instead. Visible depth differs by "
                       f"{dz:.3f} NDC on average; {fbad} vs {fgood} fragments.")


def _zfight_scene():
    """Two quads that CROSS: A is flat at z=-9; B is A tilted ~1.2 deg about its
    own centre, so B's top is a few cm in front of A and B's bottom a few cm
    behind. They intersect along a near-horizontal line. With a float depth
    buffer the crossing is a crisp edge; with a low-precision buffer and a tiny
    `near`, the two depths collapse to one code across a wide band and draw
    order (A first) wins it -- z-fighting."""
    cy, cz = 1.0, -9.0
    quadA = np.array([[-3.0, -1, 0], [3, -1, 0], [3, 3, 0], [-3, 3, 0]]) + [0, 0, cz]
    a = np.radians(1.2)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    quadB = (quadA - [0, cy, cz]) @ R.T + [0, cy, cz]
    pos = np.vstack([quadA, quadB])
    faces = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]
    return Mesh(pos, np.zeros((0, 2)), faces, [])


TRI_COLS = [(230, 130, 40), (230, 130, 40), (60, 120, 210), (60, 120, 210)]  # A orange, B blue


def bug_zfight(mesh, tex, w=192, h=192):
    m = _zfight_scene()
    cam = dict(eye=(0, 1, 6), center=(0, 1, -1), cull_back=False, far=200,
               tri_colors=TRI_COLS, light_dir=(0, 0.2, 1))
    # low-precision (16-bit) depth buffer; the float-depth render is the truth
    bad, _, _ = raster3d.render(m, w=w, h=h, near=1e-3, depth_bits=16, **cam)
    good, _, _ = raster3d.render(m, w=w, h=h, near=1.0, depth_bits=16, **cam)
    ref, _, _ = raster3d.render(m, w=w, h=h, near=1.0, depth_bits=None, **cam)
    fight_bad = int(np.sum(np.abs(bad.astype(int) - ref.astype(int)).max(2) > 60))
    fight_good = int(np.sum(np.abs(good.astype(int) - ref.astype(int)).max(2) > 60))
    ev = (f"near=1e-3 + 16-bit depth: {fight_bad} pixels wrong vs the float-depth "
          f"reference (the two quads collapse to one depth code and draw order "
          f"decides).  near=1.0 + 16-bit depth: only {fight_good} pixels differ "
          f"-> raising `near` is the fix.")
    return bad, good, ev


def bug_stretched(mesh, tex, w=192, h=192):
    # Both products use EXACTLY the same nonuniform scale and rotation factors.
    # The original starter instead appended an extra scale to commuting factors.
    T,R,S = translate((.15,0,-.1)),rot_y(35),scale((1.7,1.1,.9))
    wrong_model = T @ S @ R
    correct_model = T @ R @ S
    bad, _, _ = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=wrong_model)
    good, _, _ = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=correct_model)
    ev = ("model matrix rows (bad):\n" + np.array2string(wrong_model[:3], precision=2) +
          "\nvs good (same factors, T@R@S):\n" + np.array2string(correct_model[:3], precision=2))
    return bad, good, ev


BUGS = {"inside-out": bug_inside_out, "zfight": bug_zfight, "stretched": bug_stretched}


def outward_fallback(mesh):
    """Correct only the KNOWN 15 closed boxes of make_model.py, not arbitrary OBJ.

    Its original faces are inward despite the generator's outward-CCW comment.
    Keep historical input/ref files intact and orient a diagnostic copy instead.
    """
    if len(mesh.positions) != 120 or mesh.n_tris != 180:
        raise ValueError('outward_fallback expects the documented 15-box starter')
    faces,uvs=mesh.faces.copy(),mesh.face_uvs.copy()
    for first in range(0,len(faces),12):
        t=mesh.positions[faces[first:first+12]]
        volume=np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6
        if volume < 0:
            faces[first:first+12]=faces[first:first+12,::-1]
            uvs[first:first+12]=uvs[first:first+12,::-1]
    return Mesh(mesh.positions,mesh.uvs,faces,uvs)


def diagnostic_renders(name,mesh,tex,w=192,h=192):
    """Bad/fixed/reference RGB and true depth images for the Lab 2 evidence.

    Inside-out and matrix-order references are controlled intended configurations.
    Z-fighting references use full float precision, not the quantized fixed frame.
    """
    if name == 'inside-out':
        canonical=outward_fallback(mesh)
        flipped=Mesh(canonical.positions,canonical.uvs,canonical.faces[:,::-1].copy(),
                     canonical.face_uvs[:,::-1].copy())
        fixed=Mesh(flipped.positions,flipped.uvs,flipped.faces[:,::-1].copy(),
                   flipped.face_uvs[:,::-1].copy())
        kw=dict(w=w,h=h,tex_u8=tex,model=GOOD_MODEL,cull_back=True)
        a,da,_=raster3d.render(flipped,**kw)
        b,db,_=raster3d.render(fixed,**kw)
        ref,dr,_=raster3d.render(canonical,**kw)
        meta={'cause':'all face and UV corner indices reversed',
              'fix':'reverse BOTH geometry and UV corner order once',
              'model_matrix':GOOD_MODEL.tolist(),'cull_back':True}
    elif name == 'stretched':
        T,R,S=translate((.15,0,-.1)),rot_y(35),scale((1.7,1.1,.9))
        bad_M,good_M=T@S@R,T@R@S
        kw=dict(w=w,h=h,tex_u8=tex)
        a,da,_=raster3d.render(mesh,model=bad_M,**kw)
        b,db,_=raster3d.render(mesh,model=good_M,**kw)
        ref,dr,_=raster3d.render(mesh,model=good_M,**kw)
        meta={'cause':'T@S@R instead of T@R@S with the same factors',
              'fix':'swap only R and S',
              'bad_matrix':bad_M.tolist(),'fixed_matrix':good_M.tolist()}
    elif name == 'zfight':
        m=_zfight_scene()
        kw=dict(w=w,h=h,eye=(0,1,6),center=(0,1,-1),cull_back=False,
                far=200,tri_colors=TRI_COLS,light_dir=(0,.2,1))
        a,da,_=raster3d.render(m,near=.001,depth_bits=16,**kw)
        b,db,_=raster3d.render(m,near=1,depth_bits=16,**kw)
        ref,dr,_=raster3d.render(m,near=1,depth_bits=None,**kw)
        ref_bad,_,_=raster3d.render(m,near=.001,depth_bits=None,**kw)
        assert np.array_equal(ref,ref_bad), 'Near change altered geometry visibility'
        meta={'cause':'depth quantization ties with near=0.001 at far=200',
              'fix':'near=1.0; keep far=200 and depth_bits=16',
              'depth_bits':16,'far':200,'bad_near':.001,'fixed_near':1.0,
              'wrong_pixels_before':int((np.abs(a.astype(int)-ref.astype(int)).max(2)>60).sum()),
              'wrong_pixels_after':int((np.abs(b.astype(int)-ref.astype(int)).max(2)>60).sum())}
    else:
        raise ValueError(name)
    return {'bad':a,'fixed':b,'reference':ref,'bad_depth':da,'fixed_depth':db,
            'reference_depth':dr,'metadata':meta}


def run(name, mesh, tex):
    diagnostic = diagnostic_renders(name,mesh,tex)
    bad,good,reference=diagnostic['bad'],diagnostic['fixed'],diagnostic['reference']
    os.makedirs("out", exist_ok=True)
    write_ppm(f"out/bug_{name}_bad.ppm", bad)
    write_ppm(f"out/bug_{name}_fixed.ppm", good)
    write_ppm(f'out/bug_{name}_reference.ppm',reference)
    print(f"\n== bug: {name} ==")
    print('evidence:',json.dumps(diagnostic['metadata'],indent=2))
    print(f'SSIM(bad, reference) = {ssim(bad,reference):.6f}')
    print(f'SSIM(fixed, reference) = {ssim(good,reference):.6f}')
    print(f'wrote out/bug_{name}_bad.ppm, out/bug_{name}_fixed.ppm, out/bug_{name}_reference.ppm')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bug", choices=list(BUGS))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    mesh, tex = _load()
    if args.all or not args.bug:
        for k in BUGS:
            run(k, mesh, tex)
    else:
        run(args.bug, mesh, tex)


if __name__ == "__main__":
    main()
