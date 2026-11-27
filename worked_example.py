"""Trace the first vertex of web/mymodel.obj through the viewer pipeline."""
import numpy as np
from objio import load_obj
from transforms import rot_y, scale, look_at, perspective, viewport

np.set_printoptions(precision=6, suppress=True)
mesh = load_obj('web/mymodel.obj')
p = np.r_[mesh.positions[0], 1.0]
M = rot_y(35) @ scale((1.4, 1.4, 1.4))
V = look_at((2.6, 2.0, 3.4), (0, .7, 0))
P = perspective(45, 1, 1, 14)
world, view = M @ p, V @ M @ p
clip = P @ view
ndc = clip[:3] / clip[3]
print('object:', p)
print('M:\n', M)
print('world:', world)
print('V:\n', V)
print('view:', view)
print('clip:', clip)
print('ndc:', ndc)
print('screen (512x512):', viewport(ndc[None, :], 512, 512)[0])
print('depth [0,1]:', (ndc[2]+1)/2)
