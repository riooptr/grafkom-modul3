"""Module 3 -- Fully Worked Example, Step 1: the running point (2,3,4) through
every coordinate space.

Model order matches Example E1: translate (1,0,-1) FIRST, then yaw 90 deg,
i.e.  M = Ry(90) @ T(1,0,-1).

Run:  python worked_example.py
"""
from __future__ import annotations

import numpy as np
import argparse

from transforms import (translate, rot_y, look_at, perspective, project,
                        viewport, perspective_divide_2d)

np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', help='Trace an actual vertex from an OBJ model')
parser.add_argument('--vertex', type=int, default=0, help='Zero-based OBJ vertex index')
parser.add_argument('--width', type=int, default=800)
parser.add_argument('--height', type=int, default=600)
args = parser.parse_args()
if args.width <= 0 or args.height <= 0:
    parser.error('width and height must be positive')
if args.model:
    from objio import load_obj
    mesh = load_obj(args.model)
    if not 0 <= args.vertex < len(mesh.positions):
        parser.error('vertex index outside model')
    p = mesh.positions[args.vertex]
    print(f'model input = {args.model}, vertex index = {args.vertex} (zero-based)')
else:
    p = np.array([2.0, 3.0, 4.0])
print(f"object p        = {np.r_[p, 1.0]}")

M = rot_y(90) @ translate((1, 0, -1))
world = (M @ np.r_[p, 1.0])
print(f"model  M = Ry(90) . T(1,0,-1)        ; world  = M p       = {world}")

V = look_at(eye=(0, 0, 8), center=(0, 0, 0))
eye = V @ world
print(f"view   V = lookAt(eye=[0,0,8], ctr=0); eye    = V world   = {eye}   "
      f"(z={eye[2]:.3f}: {'in front' if eye[2] < 0 else 'BEHIND'}, dist {abs(eye[2]):.3f})")

aspect = args.width / args.height if args.model else 1.0
P = perspective(60, aspect, 1.0, 50.0)
clip = P @ eye
print(f"proj   P = perspective(60deg, {aspect:.6f}, 1, 50)")
print(f"clip            = P eye              = {clip}")
ndc = clip[:3] / clip[3]
print(f"ndc  = clip/clip.w                   = {ndc}")

W, H = args.width, args.height
scr = viewport(ndc[None, :], W, H)[0]
print(f"screen ({W}x{H}) = [{scr[0]:.0f} {scr[1]:.0f}]  depth {(ndc[2] + 1) * 0.5:.3f}")

if not args.model:
    print()
    print("--- Example E1 cross-check (bare similar-triangles projection) ---")
    print(f"world point (pre-view) = {world[:3]}   -> after Ry(90) of (3,3,3): (3,3,-3)")
    proj2d = perspective_divide_2d(world[:3], d=2)
    print(f"perspective divide d=2 of (3,3,-3)   = {proj2d}   (expected [2. 2.])")
