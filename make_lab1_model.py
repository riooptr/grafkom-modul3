"""Reproducible Lab 1 blockout inspired by Gapura Bajang Ratu.

This is a stylized exercise asset made with Codex assistance, not a measured
reconstruction or a claim that the student manually modeled it in Blender.
Run with the project's Python: python make_lab1_model.py
"""
from pathlib import Path
import json

import numpy as np
from PIL import Image

from ppm_io import write_ppm

ROOT = Path(__file__).resolve().parent


def brick_texture(size=256):
    """Original deterministic brick pattern; no downloaded image assets."""
    y, x = np.mgrid[:size, :size]
    row = y // 32
    mortar = (y % 32 < 3) | ((x + (row % 2) * 32) % 64 < 3)
    grain = 5 * np.sin(x * 0.71 + y * 1.13) + 3 * np.cos(x * .17 - y * .37)
    stagger = ((x + (row % 2) * 32) // 64 + row) % 3
    rgb = np.stack([161 + grain + 8 * stagger,
                    75 + grain + 4 * stagger,
                    43 + grain + 2 * stagger], axis=-1)
    rgb[mortar] = (94, 65, 47)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def build_gate():
    positions, uvs, faces, face_uvs, components = [], [], [], [], []

    def add_box(name, x0, x1, y0, y1, z0, z1):
        p = np.array([[x0,y0,z0], [x1,y0,z0], [x1,y0,z1], [x0,y0,z1],
                      [x0,y1,z0], [x1,y1,z0], [x1,y1,z1], [x0,y1,z1]], float)
        center = p.mean(axis=0)
        quads = [(0,1,2,3), (4,7,6,5), (0,4,5,1),
                 (3,2,6,7), (0,3,7,4), (1,5,6,2)]
        first = len(faces)
        for quad in quads:
            q = p[list(quad)]
            normal = np.cross(q[1]-q[0], q[2]-q[0])
            if np.dot(normal, q.mean(axis=0)-center) < 0:
                q = q[[0,3,2,1]]
            start = len(positions)
            positions.extend(q.tolist())
            # Four distinct UV corners for EVERY face, including roof and sides.
            # OBJ UV origin is bottom-left. Loader flips v once for NumPy images.
            uvs.extend([(0.02,0.02), (.98,.02), (.98,.98), (.02,.98)])
            for tri in [(0,1,2), (0,2,3)]:
                ids = tuple(start+i for i in tri)
                faces.append(ids)
                face_uvs.append(ids)
        components.append({'name': name, 'first_triangle': first,
                           'triangle_count': len(faces)-first})

    # A roofed gateway: two separate jambs and a lintel preserve an open passage.
    for i, (half, y0, y1) in enumerate([(1.25,0,.12),(1.12,.12,.25),(1.0,.25,.40)]):
        add_box(f'base_{i}', -half, half, y0, y1, -.70, .70)
    for side in [-1,1]:
        xa, xb = sorted([side*.36, side*.86])
        for j in range(6):
            add_box(f'jamb_{side}_{j}', xa, xb, .4+j*.22, .4+(j+1)*.22,
                    -.38, .38)
        for j, yy in enumerate([.40,.62,1.50,1.72]):
            add_box(f'jamb_moulding_{side}_{j}', xa-.07, xb+.07,
                    yy, yy+.065, -.44, .44)
        # Lower side wings and their caps.
        lo, hi = sorted([side*.92, side*1.40])
        add_box(f'wing_{side}', lo, hi, .25, .74, -.28, .28)
        add_box(f'wing_cap_{side}', lo-.04, hi+.04, .74, .83, -.33, .33)
    add_box('lintel', -.90,.90,1.72,1.96,-.43,.43)
    add_box('lintel_moulding', -.99,.99,1.96,2.06,-.49,.49)
    for j in range(6):
        half = .89 - .12*j
        yy = 2.06 + j*.17
        add_box(f'roof_tier_{j}', -half, half, yy, yy+.12,
                -half*.55, half*.55)
        add_box(f'roof_lip_{j}', -half-.035,half+.035,yy+.12,yy+.17,
                -half*.55-.025,half*.55+.025)
    add_box('finial', -.09,.09,3.08,3.26,-.09,.09)
    for side in [-1,1]:
        for j in range(5):
            # Stairs are part of the original blockout rather than extra padding.
            za, zb = sorted([side*.68, side*(1.28-.10*j)])
            add_box(f'stair_{side}_{j}', -.32,.32,.08*j,.08*(j+1),za,zb)
    # Deliberately asymmetric plaque, on the +x/front side: a mirror sentinel.
    add_box('asymmetric_front_plaque', .48,.73,1.05,1.28,.38,.48)
    return (np.array(positions), np.array(uvs), np.array(faces),
            np.array(face_uvs), components)


def main():
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    pos, uv, faces, fuv, parts = build_gate()
    with (data/'mymodel.obj').open('w', encoding='utf-8', newline='\n') as f:
        f.write('# Stylized Bajang Ratu-inspired gate; Codex-assisted exercise asset\n')
        f.write('mtllib mymodel.mtl\no BajangRatu_Blockout\n')
        for p in pos:
            f.write('v ' + ' '.join(f'{v:.6f}' for v in p) + '\n')
        for t in uv:
            f.write(f'vt {t[0]:.6f} {t[1]:.6f}\n')
        f.write('usemtl procedural_brick\n')
        for face, ft in zip(faces, fuv):
            f.write('f ' + ' '.join(f'{v+1}/{u+1}' for v,u in zip(face,ft))+'\n')
    (data/'mymodel.mtl').write_text(
        'newmtl procedural_brick\nKa 0.15 0.15 0.15\nKd 1 1 1\n'
        'Ks 0 0 0\nd 1\nillum 1\nmap_Kd lab1_brick.png\n', encoding='utf-8')
    tex = brick_texture()
    write_ppm(str(data/'lab1_brick.ppm'), tex)
    Image.fromarray(tex).save(data/'lab1_brick.png')
    metadata = {'subject':'Stylized blockout inspired by Gapura Bajang Ratu',
                'provenance':'Procedural geometry and texture authored with Codex assistance',
                'not_measured_reconstruction':True,
                'vertices':len(pos),'triangles':len(faces),'uvs':len(uv),
                'components':parts,'axis_convention':'right-handed; +y up; front +z',
                'asset_license':'CC0-1.0 (new geometry and procedural texture only)'}
    (data/'lab1_model.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    (data/'ASSET_LICENSE.md').write_text(
        '# Aset baru Lab 1\n\n'
        '`mymodel.obj`, `mymodel.mtl`, `lab1_brick.png`, dan `lab1_brick.ppm` '
        'dibuat secara prosedural dengan bantuan Codex melalui `make_lab1_model.py`. '
        'Tidak memakai foto/texture pihak ketiga. Aset baru ini didedikasikan dengan '
        'CC0-1.0: https://creativecommons.org/publicdomain/zero/1.0/\n\n'
        'Model merupakan interpretasi sederhana untuk praktikum, bukan rekonstruksi '
        'arkeologis terukur atau model yang diklaim dibuat manual oleh mahasiswa. '
        'Lisensi ini tidak mengubah lisensi starter, Blender/House.blend, atau sumber bacaan.\n',
        encoding='utf-8')
    print(f'Wrote data/mymodel.obj: {len(pos)} vertices, {len(faces)} triangles, {len(parts)} boxes')
    print('Wrote material, 256x256 brick PNG/PPM, provenance, and CC0 asset notice')


if __name__ == '__main__':
    main()
