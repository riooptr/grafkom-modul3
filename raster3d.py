"""Module 3 -- the from-scratch rasteriser: MVP -> near clip -> w-divide ->
viewport -> barycentric z-buffer fill -> perspective-correct texture -> Lambert.

NumPy only. This is Module 2's edge-function triangle plus a depth test and
perspective-correct attribute interpolation.

Run:
  python raster3d.py --model data/pusaka_model.obj --w 512 --h 512
  python raster3d.py --demo-triangle --w 64 --h 64
  python raster3d.py --demo-overlap  --w 64 --h 64
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from transforms import perspective, look_at, rot_y, translate, scale
from objio import load_obj
from ppm_io import read_ppm, write_ppm
from srgb import srgb_to_linear, linear_to_srgb
from texture_sampling import (sample_bilinear, sample_nearest, sample_trilinear,
                              build_mipmaps)


# ---------------------------------------------------------------- near clipping
def clip_triangle_near(vc, attrs):
    """Sutherland-Hodgman clip one triangle (clip-space vc: (3,4)) against the
    near plane  z + w >= 0.  attrs: (3, k) per-vertex attributes to interpolate.
    Returns a list of (vc', attrs') triangles (0, 1 or 2)."""
    def inside(p):
        return p[2] + p[3] >= 0.0
    poly = [(vc[i], attrs[i]) for i in range(3)]
    out = []
    n = len(poly)
    for i in range(n):
        cur_v, cur_a = poly[i]
        prv_v, prv_a = poly[(i - 1) % n]
        ci, pi = inside(cur_v), inside(prv_v)
        if ci:
            if not pi:
                t = (prv_v[2] + prv_v[3]) / ((prv_v[2] + prv_v[3]) - (cur_v[2] + cur_v[3]))
                out.append((prv_v + t * (cur_v - prv_v), prv_a + t * (cur_a - prv_a)))
            out.append((cur_v, cur_a))
        elif pi:
            t = (prv_v[2] + prv_v[3]) / ((prv_v[2] + prv_v[3]) - (cur_v[2] + cur_v[3]))
            out.append((prv_v + t * (cur_v - prv_v), prv_a + t * (cur_a - prv_a)))
    if len(out) < 3:
        return []
    tris = []
    for k in range(1, len(out) - 1):
        vv = np.stack([out[0][0], out[k][0], out[k + 1][0]])
        aa = np.stack([out[0][1], out[k][1], out[k + 1][1]])
        tris.append((vv, aa))
    return tris


# --------------------------------------------------------------- the rasteriser
def render(mesh, *, w, h, tex_u8=None, model=None, tri_colors=None,
           eye=(2.6, 2.0, 3.4), center=(0, 0.7, 0), fovy=45.0,
           near=1.0, far=14.0, light_dir=(0.4, 0.9, 0.6), ambient=0.15,
           cull_back=False, depth_bits=None, perspective_correct=True,
           texture_filter='bilinear', mip_levels=None, stats=None):
    """depth_bits: if set, the stored depth is quantised to that many bits
    (a low-precision depth buffer) -- used to make z-fighting reproducible."""
    if w <= 0 or h <= 0 or not 0 < near < far:
        raise ValueError('positive framebuffer and 0 < near < far required')
    if depth_bits is not None and not 1 <= depth_bits <= 32:
        raise ValueError('depth_bits must be in [1,32]')
    if texture_filter not in ('nearest','bilinear','trilinear'):
        raise ValueError('unknown texture filter')
    if stats is None:
        stats = {}
    stats.update(triangles_input=mesh.n_tris, triangles_after_near_clip=0,
                 triangles_culled=0, covered_fragments=0, depth_passes=0,
                 sampled_lod_sum=0.0, sampled_lod_max=0.0, texture_samples=0)
    if model is None:
        model = np.eye(4)
    eye_pos = np.asarray(eye, float)
    P = perspective(fovy, w / h, near, far)
    V = look_at(eye, center)
    world = (model @ np.c_[mesh.positions, np.ones(len(mesh.positions))].T).T[:, :3]
    clip = (P @ V @ model @ np.c_[mesh.positions, np.ones(len(mesh.positions))].T).T  # (Vn,4)

    tris_pos = mesh.faces                      # (F,3) position idx
    tris_uv = mesh.triangle_uvs()              # (F,3,2)
    L = np.asarray(light_dir, float); L = L / np.linalg.norm(L)

    tex_lin = None
    if tex_u8 is not None:
        tex_lin = srgb_to_linear(tex_u8.astype(np.float64) / 255.0)
        if texture_filter == 'trilinear' and mip_levels is None:
            mip_levels = build_mipmaps(tex_u8,4)
    elif texture_filter == 'trilinear':
        raise ValueError('trilinear filtering needs a texture')

    color = np.zeros((h, w, 3), np.float64)
    depth = np.full((h, w), np.inf)
    filled = 0

    for fi in range(len(tris_pos)):
        vidx = tris_pos[fi]
        vc = clip[vidx]                        # (3,4)
        # flat normal + Lambert term in world space
        wp = world[vidx]
        nrm = np.cross(wp[1] - wp[0], wp[2] - wp[0])
        nl = np.linalg.norm(nrm)
        if nl < 1e-12:
            continue
        nrm = nrm / nl
        # orient the normal toward the camera (shade the side we actually see)
        if nrm @ (eye_pos - wp[0]) < 0:
            nrm = -nrm
        shade = ambient + (1 - ambient) * max(0.0, float(nrm @ L))

        uv = tris_uv[fi]                       # (3,2)
        # attrs to carry through clipping: [u, v, 1] (1 -> becomes 1/w below)
        attrs = np.c_[uv, np.ones(3)]
        for (cvc, cattr) in (clip_triangle_near(vc, attrs) or []):
            stats['triangles_after_near_clip'] += 1
            wv = cvc[:, 3]
            ndc = cvc[:, :3] / wv[:, None]
            sx = (ndc[:, 0] * 0.5 + 0.5) * w
            sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h
            sz = ndc[:, 2]
            inv_w = 1.0 / wv
            uvw = cattr[:, :2] * inv_w[:, None]      # u/w, v/w

            minx = max(0, int(np.floor(sx.min())))
            maxx = min(w - 1, int(np.ceil(sx.max())))
            miny = max(0, int(np.floor(sy.min())))
            maxy = min(h - 1, int(np.ceil(sy.max())))
            if minx > maxx or miny > maxy:
                continue

            ax, ay = sx[0], sy[0]
            bx, by = sx[1], sy[1]
            cx, cy = sx[2], sy[2]
            # area = edge function of C about edge A->B (2x signed triangle area)
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if area == 0:
                continue
            if cull_back and area > 0:          # screen y-down: front faces have area<0
                stats['triangles_culled'] += 1
                continue

            ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
            px = xs + 0.5; py = ys + 0.5
            # edge functions, normalised by `area` -> barycentric weights.
            #   lA = edge(B,C,P)/area   lB = edge(C,A,P)/area   lC = edge(A,B,P)/area
            lA = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) / area
            lB = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) / area
            lC = 1.0 - lA - lB
            inside = (lA >= 0) & (lB >= 0) & (lC >= 0)
            if not inside.any():
                continue
            stats['covered_fragments'] += int(inside.sum())

            l0, l1, l2 = lA, lB, lC          # vertex 0=A, 1=B, 2=C
            frag_z = l0 * sz[0] + l1 * sz[1] + l2 * sz[2]
            # Near clipping already changed the polygon. The far plane can be
            # rejected per fragment here because z_NDC is affine in screen space.
            in_depth_range = (frag_z >= -1-1e-12) & (frag_z <= 1+1e-12)
            if depth_bits is not None:
                q = float(2 ** depth_bits - 1)
                frag_z = np.round((frag_z * 0.5 + 0.5) * q) / q * 2.0 - 1.0
            inv_w_i = l0 * inv_w[0] + l1 * inv_w[1] + l2 * inv_w[2]

            reg = depth[miny:maxy + 1, minx:maxx + 1]
            win = inside & in_depth_range & (frag_z < reg)
            if not win.any():
                continue
            reg[win] = frag_z[win]

            if tex_lin is not None:
                safe_inv_w = np.where(np.abs(inv_w_i) > 1e-15, inv_w_i, 1.0)
                if perspective_correct:
                    u_i = (l0 * uvw[0, 0] + l1 * uvw[1, 0] + l2 * uvw[2, 0]) / safe_inv_w
                    v_i = (l0 * uvw[0, 1] + l1 * uvw[1, 1] + l2 * uvw[2, 1]) / safe_inv_w
                else:
                    u_i = l0*cattr[0,0] + l1*cattr[1,0] + l2*cattr[2,0]
                    v_i = l0*cattr[0,1] + l1*cattr[1,1] + l2*cattr[2,1]
                if texture_filter == 'trilinear':
                    # Quotient rule for rational UV interpolation. Each lambda
                    # derivative is constant across this projected triangle.
                    dx = np.array([-(cy-by),-(ay-cy),-(by-ay)])/area
                    dy = np.array([(cx-bx),(ax-cx),(bx-ax)])/area
                    if perspective_correct:
                        rx,ry = dx@inv_w,dy@inv_w
                        ux = (dx@uvw[:,0]-u_i*rx)/safe_inv_w
                        vx = (dx@uvw[:,1]-v_i*rx)/safe_inv_w
                        uy = (dy@uvw[:,0]-u_i*ry)/safe_inv_w
                        vy = (dy@uvw[:,1]-v_i*ry)/safe_inv_w
                    else:
                        ux,vx=dx@cattr[:,0],dx@cattr[:,1]
                        uy,vy=dy@cattr[:,0],dy@cattr[:,1]
                    th,tw = tex_lin.shape[:2]
                    rho = np.maximum(np.hypot(ux*(tw-1),vx*(th-1)),
                                     np.hypot(uy*(tw-1),vy*(th-1)))
                    lod = np.broadcast_to(np.log2(np.maximum(rho,1.0)),u_i.shape)
                    used_lod = np.clip(lod,0,len(mip_levels)-1)
                    tex_rgb = sample_trilinear(mip_levels,u_i,v_i,lod)
                    stats['sampled_lod_sum'] += float(used_lod[win].sum())
                    stats['sampled_lod_max'] = max(stats['sampled_lod_max'],float(used_lod[win].max()))
                elif texture_filter == 'nearest':
                    tex_rgb = sample_nearest(tex_lin,u_i,v_i)
                else:
                    tex_rgb = sample_bilinear(tex_lin,u_i,v_i)
                stats['texture_samples'] += int(win.sum())
            elif tri_colors is not None:
                tex_rgb = np.ones((*lA.shape, 3)) * srgb_to_linear(
                    np.asarray(tri_colors[fi], float) / 255.0)
            else:
                tex_rgb = np.ones((*lA.shape, 3)) * 0.7

            lit = np.clip(tex_rgb * shade, 0, 1)
            cregion = color[miny:maxy + 1, minx:maxx + 1]
            cregion[win] = linear_to_srgb(lit)[win]
            filled += int(win.sum())

    stats['depth_passes'] = filled
    return (np.clip(color * 255, 0, 255).astype(np.uint8), depth, filled)


# ------------------------------------------------------------------------ demos
def _demo_triangle(w, h):
    """One textured triangle -- reproduces Fully Worked Example Step 2."""
    from objio import Mesh
    pos = np.array([[-0.8, -0.6, -2.0], [0.9, -0.7, -3.5], [-0.2, 0.9, -2.5]])
    uv = np.array([[0, 0], [1, 0], [0.5, 1.0]])
    m = Mesh(pos, uv, [[0, 1, 2]], [[0, 1, 2]])
    tex = read_ppm("data/relief.ppm") if os.path.exists("data/relief.ppm") else None
    P = perspective(60, w / h, 1.0, 50.0)
    V = np.eye(4)
    clip, ndc = _project(P, V, np.eye(4), pos)
    scr = _viewport(ndc, w, h)
    print(f"triangle screen verts: "
          f"({scr[0,0]:.0f},{scr[0,1]:.0f}) ({scr[1,0]:.0f},{scr[1,1]:.0f}) "
          f"({scr[2,0]:.0f},{scr[2,1]:.0f})   uv: (0,0) (1,0) (0.5,1)")
    col, dep, filled = render(m, w=w, h=h, tex_u8=tex, near=1.0, far=50.0,
                              eye=(0, 0, 0), center=(0, 0, -1), cull_back=False,
                              light_dir=(0, 0, 1))
    fin = np.isfinite(dep)
    if fin.any():
        print(f"filled fragments: {filled}   depth range "
              f"[{dep[fin].min():.2f}, {dep[fin].max():.2f}]")
    # perspective-correct vs naive UV at the centroid
    invw = 1.0 / clip[:, 3]
    uvw = uv * invw[:, None]
    pc = (uvw.mean(0)) / invw.mean()
    print(f"perspective-correct UV at centroid: ({pc[0]:.3f}, {pc[1]:.3f})   "
          f"(naive linear: ({uv[:,0].mean():.3f}, {uv[:,1].mean():.3f}))")
    os.makedirs("out", exist_ok=True)
    write_ppm("out/tri_raster.ppm", col)
    print("wrote out/tri_raster.ppm  (texture: data/relief.ppm, bilinear)")


def _demo_overlap(w, h):
    from objio import Mesh
    front = [[-0.7, -0.7, -1.5], [0.7, -0.7, -1.5], [0.0, 0.7, -1.5]]
    back = [[-0.2, -0.7, -3.0], [1.2, -0.7, -3.0], [0.5, 0.7, -3.0]]
    pos = np.array(front + back)
    m = Mesh(pos, np.zeros((0, 2)), [[0, 1, 2], [3, 4, 5]], [])
    col, dep, _ = render(m, w=w, h=h, tex_u8=None, near=1.0, far=50.0,
                         eye=(0, 0, 0), center=(0, 0, -1), cull_back=False,
                         light_dir=(0, 0, 1))
    # re-render back only to count its unobstructed pixels
    mb = Mesh(np.array(back), np.zeros((0, 2)), [[0, 1, 2]], [])
    _, depb, fb = render(mb, w=w, h=h, near=1.0, far=50.0, eye=(0, 0, 0),
                         center=(0, 0, -1), cull_back=False, light_dir=(0, 0, 1))
    mf = Mesh(np.array(front), np.zeros((0,2)), [[0,1,2]], [])
    _,depf,_ = render(mf,w=w,h=h,near=1,far=50,eye=(0,0,0),
                      center=(0,0,-1),light_dir=(0,0,1))
    overlap = np.isfinite(depf) & np.isfinite(depb)
    back_wins = int(np.sum(overlap & (dep >= depb-1e-12)))
    back_visible = int(np.sum(np.isfinite(depb) & ~np.isfinite(depf)))
    print(f'front depth NDC={np.median(depf[np.isfinite(depf)]):.3f}; '
          f'back depth NDC={np.median(depb[np.isfinite(depb)]):.3f}')
    print(f'pixels where BACK won in overlap: {back_wins}/{int(overlap.sum())}')
    print(f'pixels where BACK is visible (no overlap): {back_visible}')
    os.makedirs("out", exist_ok=True)
    write_ppm("out/overlap.ppm", col)
    print("wrote out/overlap.ppm")


def _project(P, V, M, verts):
    from transforms import project
    return project(P, V, M, verts)


def _viewport(ndc, w, h):
    from transforms import viewport
    return viewport(ndc, w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--out", default="out/temple_raster.ppm")
    ap.add_argument('--texture', default='data/relief.ppm')
    ap.add_argument('--mipmaps', action='store_true')
    ap.add_argument('--naive-uv', action='store_true')
    ap.add_argument('--cull-back', action='store_true')
    ap.add_argument("--demo-triangle", action="store_true")
    ap.add_argument("--demo-overlap", action="store_true")
    args = ap.parse_args()

    if args.demo_triangle:
        _demo_triangle(args.w, args.h); return
    if args.demo_overlap:
        _demo_overlap(args.w, args.h); return

    if not args.model:
        raise SystemExit("give --model, or --demo-triangle / --demo-overlap")
    mesh = load_obj(args.model)
    tex = read_ppm(args.texture) if os.path.exists(args.texture) else None
    model = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))
    col, dep, filled = render(mesh, w=args.w, h=args.h, tex_u8=tex, model=model,
                              texture_filter='trilinear' if args.mipmaps else 'bilinear',
                              perspective_correct=not args.naive_uv,cull_back=args.cull_back)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    write_ppm(args.out, col)
    fin = np.isfinite(dep)
    dvis = np.where(fin, (dep + 1) * 0.5, 1.0)
    parent,filename = os.path.split(args.out)
    depth_name = (filename.replace('raster','depth') if 'raster' in filename
                  else os.path.splitext(filename)[0]+'_depth.ppm')
    depth_path = os.path.join(parent,depth_name)
    write_ppm(depth_path,
              np.repeat((np.clip(dvis, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2))
    print(f"model: {mesh.n_tris} triangles   framebuffer {args.w}x{args.h}")
    print(f"filled fragments: {filled}   depth(ndc) range "
          f"[{dep[fin].min():.3f}, {dep[fin].max():.3f}]" if fin.any() else "nothing drawn")
    print(f"wrote {args.out} and {depth_path}")


if __name__ == "__main__":
    main()
